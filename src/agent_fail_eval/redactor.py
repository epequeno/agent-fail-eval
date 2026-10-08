"""Session content redaction for safe sharing.

Redacts sensitive content from session transcripts and evaluation reports:
- File paths (absolute, relative, tilde-expanded) → deterministic labels
- Project names and CWD paths → anonymized labels
- Code blocks in text and tool results → [CODE_BLOCK] placeholder
- API keys, tokens, and secrets → [REDACTED]
- Session IDs → hashed

Redaction is deterministic (same input → same label) but irreversible.
Applied at the SessionRecord level so all downstream extraction and
reporting automatically produce clean output.
"""

from __future__ import annotations

import hashlib
import re
from typing import ClassVar

from agent_fail_eval.models import (
    BLOCK_TEXT,
    BLOCK_TOOL_CALL,
    BLOCK_TOOL_RESULT,
    CompactionBoundary,
    SessionRecord,
)

# Pattern matches file paths:
# - Absolute: /Users/..., /home/..., /tmp/...
# - Tilde: ~/code/..., ~/.config/...
# - Project-like: --project-name-- directories in paths
PATH_RE = re.compile(
    r"(?:/Users/|/home/|/tmp/|/var/|/etc/|~/|\./|\.\./)[^\s\"'<>|]+"
)
# Be more aggressive for common sensitive patterns
HOME_PATH_RE = re.compile(r"(?:/Users/\w+|/home/\w+|~)/[^\s\"'<>|\[\](){}]+")

# Project directory names in paths (e.g., --my-project--)
PROJECT_DIR_RE = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*--")

# Secrets: API keys, tokens, bearer auth
SECRET_RE = re.compile(
    r"(?:sk-[a-zA-Z0-9]{20,}"                # OpenAI-style keys
    r"|(?:api[_-]?key|api[_-]?secret|token|secret|password|auth)\s*[:=]\s*[\"'`]?[^\s\"'`\n]{8,}[\"'`]?"  # key=value
    r"|Bearer\s+[a-zA-Z0-9._\-]{20,}"         # Bearer tokens
    r"|[a-zA-Z0-9+/]{40,}={0,2})"             # Base64-like tokens
)

# Code block markers: triple backticks, indented code
CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")

# Known content-free markers
PADDING_RE = re.compile(r"dependency output line \d+", re.IGNORECASE)


class SessionRedactor:
    """Redacts sensitive content from session records.

    Deterministic: uses SHA-256 hashing with a per-session salt so
    the same input always maps to the same label within a redaction session,
    but labels cannot be reversed to original values without the salt.

    Usage:
        redactor = SessionRedactor()
        # Redact individual records
        clean_record = redactor.redact_record(record)
        # Redact a compaction boundary (summary + dropped records)
        clean_boundary = redactor.redact_boundary(boundary)
    """

    # Characters for generating readable labels
    LABEL_CHARS: ClassVar[str] = "abcdefghijklmnopqrstuvwxyz0123456789"

    def __init__(self, salt: str = "") -> None:
        """Initialize redactor with an optional salt.

        Args:
            salt: Optional salt for the hash function. A random salt is
                  generated if not provided, ensuring different redaction
                  runs produce different labels.
        """
        import secrets

        self._salt = salt or secrets.token_hex(16)
        self._path_map: dict[str, str] = {}
        self._project_map: dict[str, str] = {}
        self._counter: int = 0

    def _hash_label(self, value: str, prefix: str) -> str:
        """Generate a deterministic label for a value.

        Args:
            value: The value to hash.
            prefix: Label prefix (e.g., 'path', 'proj').

        Returns:
            A label like 'path_a1b2c3d4' or 'proj_e5f6'.
        """
        h = hashlib.sha256(f"{self._salt}:{value}".encode()).hexdigest()[:8]
        return f"{prefix}_{h}"

    def _next_project_label(self) -> str:
        """Generate next sequential project label."""
        self._counter += 1
        return f"project_{self._counter:03d}"

    def redact_path(self, path: str) -> str:
        """Redact a file path to a deterministic label.

        Preserves the file extension when possible for readability.
        """
        if path in self._path_map:
            return self._path_map[path]

        # Extract extension for context
        ext = ""
        if "." in path.rsplit("/", 1)[-1]:
            ext = path.rsplit(".", 1)[-1]
            if len(ext) > 10 or "/" in ext:
                ext = ""  # Not a real extension
            elif ext:
                ext = f".{ext}"

        label = f"[file_{self._hash_label(path, 'f')[:8]}]{ext}"
        self._path_map[path] = label
        return label

    def redact_project(self, project: str) -> str:
        """Redact a project name or CWD path."""
        if project in self._project_map:
            return self._project_map[project]

        label = self._next_project_label()
        self._project_map[project] = label
        return label

    def redact_project_from_path(self, path_str: str) -> str:
        """Redact project directory markers in a path string.

        Replaces --project-name-- patterns with [project_NNN].
        """
        def _replace(m: re.Match) -> str:
            return self.redact_project(m.group(0))

        return PROJECT_DIR_RE.sub(_replace, path_str)

    def redact_text(self, text: str) -> str:
        """Redact sensitive content from arbitrary text.

        Handles: file paths, secrets, code blocks.
        """
        if not text:
            return text

        result = text

        # 1. Redact secrets first (highest priority)
        result = SECRET_RE.sub("[REDACTED]", result)

        # 2. Redact home directory paths
        def _replace_path(m: re.Match) -> str:
            return self.redact_path(m.group(0))

        result = HOME_PATH_RE.sub(_replace_path, result)
        result = PATH_RE.sub(_replace_path, result)

        # 3. Redact project directory patterns in remaining text
        result = self.redact_project_from_path(result)

        # 4. Redact code blocks (triple backtick fenced)
        result = CODE_FENCE_RE.sub("[CODE_BLOCK]", result)

        return result

    def redact_record(self, record: SessionRecord) -> SessionRecord:
        """Redact sensitive content from a session record.

        Returns a new SessionRecord with redacted fields.
        The original record is not modified.
        """
        import copy

        # Shallow copy the data dict
        clean_data = copy.deepcopy(record.data) if record.data else {}

        # Redact CWD
        clean_cwd = self.redact_project(record.cwd) if record.cwd else None

        # Redact message content
        clean_message = None
        if record.message:
            clean_message = copy.deepcopy(record.message)
            content = clean_message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        self._redact_content_block(block)

        # Redact compaction summary
        clean_summary = self.redact_text(record.summary) if record.summary else None

        # Redact file lists
        clean_read_files = None
        if record.read_files:
            clean_read_files = [self.redact_path(f) for f in record.read_files]
        clean_modified_files = None
        if record.modified_files:
            clean_modified_files = [self.redact_path(f) for f in record.modified_files]

        return SessionRecord(
            type=record.type,
            id=record.id,
            parent_id=record.parent_id,
            timestamp=record.timestamp,
            cwd=clean_cwd,
            version=record.version,
            message=clean_message,
            provider=record.provider,
            model_id=record.model_id,
            thinking_level=record.thinking_level,
            summary=clean_summary,
            first_kept_entry_id=record.first_kept_entry_id,
            tokens_before=record.tokens_before,
            read_files=clean_read_files,
            modified_files=clean_modified_files,
            from_hook=record.from_hook,
            custom_type=record.custom_type,
            data=clean_data,
        )

    def _redact_content_block(self, block: dict) -> None:
        """Redact a single content block in-place."""
        bt = block.get("type", "")

        if bt == BLOCK_TEXT:
            if "text" in block and block["text"]:
                block["text"] = self.redact_text(block["text"])

        elif bt == BLOCK_TOOL_CALL:
            # Redact arguments (they often contain file paths)
            if "arguments" in block and block["arguments"]:
                if isinstance(block["arguments"], dict):
                    block["arguments"] = self._redact_dict_values(
                        block["arguments"]
                    )
                elif isinstance(block["arguments"], str):
                    block["arguments"] = self.redact_text(block["arguments"])

            # Redact tool result content if present
            if "content" in block and block["content"]:
                if isinstance(block["content"], list):
                    for sub_block in block["content"]:
                        if isinstance(sub_block, dict):
                            self._redact_content_block(sub_block)
                elif isinstance(block["content"], str):
                    block["content"] = self.redact_text(block["content"])

        elif bt == BLOCK_TOOL_RESULT:
            if "content" in block and block["content"]:
                if isinstance(block["content"], list):
                    for sub_block in block["content"]:
                        if isinstance(sub_block, dict):
                            self._redact_content_block(sub_block)
                elif isinstance(block["content"], str):
                    block["content"] = self.redact_text(block["content"])

    def _redact_dict_values(self, d: dict) -> dict:
        """Redact all string values in a dict."""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = self.redact_text(v)
            elif isinstance(v, dict):
                result[k] = self._redact_dict_values(v)
            elif isinstance(v, list):
                result[k] = [
                    self.redact_text(item) if isinstance(item, str) else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    def redact_boundary(self, boundary: CompactionBoundary) -> CompactionBoundary:
        """Redact a compaction boundary and its dropped records.

        Returns a new CompactionBoundary with redacted summary and
        dropped records. The original boundary is not modified.
        """
        # Redact summary
        clean_summary = self.redact_text(boundary.summary)

        # Redact all dropped records
        clean_dropped = [self.redact_record(r) for r in boundary.dropped_records]

        # Redact file lists
        clean_read_files = (
            [self.redact_path(f) for f in boundary.read_files]
            if boundary.read_files
            else None
        )
        clean_modified_files = (
            [self.redact_path(f) for f in boundary.modified_files]
            if boundary.modified_files
            else None
        )

        return CompactionBoundary(
            compaction_id=boundary.compaction_id,
            timestamp=boundary.timestamp,
            summary=clean_summary,
            first_kept_entry_id=boundary.first_kept_entry_id,
            tokens_before=boundary.tokens_before,
            file_position=boundary.file_position,
            first_kept_position=boundary.first_kept_position,
            dropped_records=clean_dropped,
            newly_dropped_count=boundary.newly_dropped_count,
            read_files=clean_read_files,
            modified_files=clean_modified_files,
            from_hook=boundary.from_hook,
        )

    def redact_session_path(self, path: str) -> str:
        """Redact a session file path to just a redacted session ID."""
        import os

        # Extract the filename stem
        stem = os.path.splitext(os.path.basename(path))[0]
        # Use the path as the hash input for determinism
        return self._hash_label(path, "session")

    def redact_project_name(self, path_str: str) -> str:
        """Extract and redact project name from a path.

        Uses the same extraction logic as _extract_project_name in report.py
        but returns a redacted label instead.
        """
        from pathlib import Path

        path = Path(path_str) if not isinstance(path_str, Path) else path_str

        if isinstance(path_str, Path):
            parent = path_str.parent.name
        else:
            import os
            parent = os.path.basename(os.path.dirname(str(path_str)))

        parts = parent.strip("-").split("-")
        if "code" in parts:
            idx = parts.index("code")
            extracted = "-".join(parts[idx + 1:]) or parent
        else:
            extracted = parent.strip("-")

        return self.redact_project(extracted)
