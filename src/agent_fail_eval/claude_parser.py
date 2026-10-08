"""Session log parser for Claude Code JSONL files.

Claude Code stores sessions as JSONL at:
    ~/Library/Application Support/21st-desktop/claude-sessions/<hash>/projects/<project>/<session-id>.jsonl

Record types in Claude Code format:
    - ``user``      — user message (role="user" in message.content)
    - ``assistant`` — assistant message (role="assistant" in message.content)
    - ``system``    — system events; ``subtype: "compact_boundary"`` marks compaction
    - ``queue-operation`` — queue metadata (skipped)

Compaction mechanism:
    A ``system`` record with ``subtype: "compact_boundary"`` marks the event.
    ``logicalParentUuid`` is the UUID of the last record before compaction
    (may not exist in the file — it was dropped). The record immediately
    after the boundary is a synthetic ``user`` message whose text content
    is the compaction summary (starts with "This session is being continued
    from a previous conversation that ran out of context...").

    Everything before the compact_boundary record is "dropped" by this
    compaction. The first kept record is the summary message immediately
    after the boundary.

Mapping to the shared SessionRecord/ParsedSession model:
    Claude Code → pi-agent-compatible SessionRecord/CompactionBoundary.
    All downstream tools (FactExtractor, CompactionLossEvaluator,
    VulnerabilityWindowDetector) work without modification.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent_fail_eval.models import (
    COMPACTION,
    CUSTOM_MESSAGE,
    MESSAGE,
    CompactionBoundary,
    ParsedSession,
    SessionRecord,
)

if TYPE_CHECKING:
    from agent_fail_eval.redactor import SessionRedactor


# Prefix that identifies the compaction-summary synthetic user message.
_SUMMARY_PREFIX = "This session is being continued from a previous conversation"


class ClaudeCodeLoader:
    """Loads and parses Claude Code session JSONL files."""

    def load(
        self, path: str | Path, *, redactor: SessionRedactor | None = None
    ) -> ParsedSession:
        """Parse a Claude Code session JSONL file into a ParsedSession.

        Raises ValueError if the file cannot be parsed as valid JSONL
        or contains no records.
        """
        path = Path(path)
        records: list[SessionRecord] = []
        session_id = path.stem  # filename without .jsonl
        cwd = ""

        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rec = self._parse_record(raw)
                if rec is None:
                    continue
                if not session_id and rec.message:
                    session_id = raw.get("sessionId", path.stem)
                if not cwd:
                    cwd = raw.get("cwd", "")
                records.append(rec)

        if not records:
            raise ValueError(f"No parseable records in {path}")

        if redactor is not None:
            records = [redactor.redact_record(r) for r in records]
            cwd = redactor.redact_project(cwd) if cwd else cwd

        # Extract session_id from the first record that has one
        if records and not session_id:
            for r in records:
                if r.data.get("sessionId"):
                    session_id = r.data["sessionId"]
                    break

        boundaries = self._extract_boundaries(records)

        return ParsedSession(
            session_id=session_id,
            cwd=cwd or "/unknown",
            started_at=datetime.min,
            records=records,
            compaction_boundaries=boundaries,
        )

    def _parse_record(self, raw: dict) -> SessionRecord | None:
        """Parse a raw Claude Code JSON record into a SessionRecord.

        Returns None for records that should be skipped (queue-operation).
        """
        rec_type = raw.get("type", "")

        if rec_type == "queue-operation":
            return None

        if rec_type == "system":
            subtype = raw.get("subtype", "")
            if subtype == "compact_boundary":
                return self._parse_compaction(raw)
            # Other system records become custom_message
            return SessionRecord(
                type=CUSTOM_MESSAGE,
                id=raw.get("uuid"),
                parent_id=raw.get("parentUuid"),
                timestamp=raw.get("timestamp"),
                custom_type=subtype or "system",
                data=raw,
            )

        if rec_type in ("user", "assistant"):
            return self._parse_message(raw)

        # Unknown record type — preserve as custom_message
        return SessionRecord(
            type=CUSTOM_MESSAGE,
            id=raw.get("uuid"),
            parent_id=raw.get("parentUuid"),
            timestamp=raw.get("timestamp"),
            custom_type=rec_type,
            data=raw,
        )

    def _parse_message(self, raw: dict) -> SessionRecord:
        """Parse a user/assistant message record."""
        message = raw.get("message", {})
        # Claude Code stores role inside message; normalize to match model
        role = message.get("role", raw.get("type", ""))
        if role == "user":
            message = {**message, "role": "user"}
        elif role == "assistant":
            message = {**message, "role": "assistant"}

        # toolResult messages in Claude Code have type "user" with
        # tool_result content blocks
        content = message.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    message = {**message, "role": "toolResult"}
                    break

        return SessionRecord(
            type=MESSAGE,
            id=raw.get("uuid"),
            parent_id=raw.get("parentUuid"),
            timestamp=raw.get("timestamp"),
            cwd=raw.get("cwd"),
            message=message,
            data=raw,
        )

    def _parse_compaction(self, raw: dict) -> SessionRecord:
        """Parse a compact_boundary system record into a compaction SessionRecord."""
        meta = raw.get("compactMetadata", {})
        return SessionRecord(
            type=COMPACTION,
            id=raw.get("uuid"),
            parent_id=None,  # compact_boundary has parentUuid=None
            timestamp=raw.get("timestamp"),
            summary=None,  # summary is in the next record, filled later
            first_kept_entry_id=None,  # will be set to next record's UUID
            tokens_before=meta.get("preTokens"),
            from_hook=meta.get("trigger") == "auto",
            data=raw,
        )

    def _extract_boundaries(
        self, records: list[SessionRecord]
    ) -> list[CompactionBoundary]:
        """Extract compaction boundaries with their dropped segments.

        For each compact_boundary record:
        - Everything before it (from the previous boundary or session start)
          is the dropped segment.
        - The summary is the text content of the next user message after
          the boundary (the synthetic "This session is being continued..."
          message).
        - The first kept record is that summary message.
        """
        boundaries: list[CompactionBoundary] = []
        prev_first_kept_pos = 0

        # Build ID → index map
        id_to_index: dict[str, int] = {}
        for i, r in enumerate(records):
            if r.id:
                id_to_index[r.id] = i

        compaction_indices = [
            (i, r) for i, r in enumerate(records) if r.type == COMPACTION
        ]

        for comp_idx, comp_rec in compaction_indices:
            # The summary is the next message record after the boundary
            summary = ""
            first_kept_pos = comp_idx + 1

            if first_kept_pos < len(records):
                next_rec = records[first_kept_pos]
                # Extract summary text from the synthetic user message
                if next_rec.is_message and next_rec.role == "user":
                    text = next_rec.text_content
                    if text.startswith(_SUMMARY_PREFIX):
                        summary = text

            # Dropped segment: from previous boundary to this boundary
            dropped = records[prev_first_kept_pos:comp_idx]

            # Set the compaction record's summary and first_kept_entry_id
            comp_rec.summary = summary
            first_kept_id = records[first_kept_pos].id if first_kept_pos < len(records) else None
            comp_rec.first_kept_entry_id = first_kept_id

            boundary = CompactionBoundary(
                compaction_id=comp_rec.id or "",
                timestamp=comp_rec.timestamp or "",
                summary=summary,
                first_kept_entry_id=first_kept_id or "",
                tokens_before=comp_rec.tokens_before or 0,
                file_position=comp_idx,
                first_kept_position=first_kept_pos,
                dropped_records=dropped,
                newly_dropped_count=len(dropped),
                from_hook=comp_rec.from_hook,
            )
            boundaries.append(boundary)

            # Next compaction drops from this boundary's first kept position
            prev_first_kept_pos = first_kept_pos

        return boundaries

    def scan_directory(
        self,
        dir_path: str | Path,
        *,
        redactor: SessionRedactor | None = None,
    ) -> list[tuple[Path, ParsedSession]]:
        """Scan a directory for Claude Code session files with compaction.

        Returns a list of (path, parsed_session) for sessions that contain
        at least one compaction event.
        """
        dir_path = Path(dir_path)
        results: list[tuple[Path, ParsedSession]] = []

        for jsonl_path in sorted(dir_path.rglob("*.jsonl")):
            if "subagents" in str(jsonl_path):
                continue
            try:
                session = self.load(jsonl_path, redactor=redactor)
                if session.compaction_boundaries:
                    results.append((jsonl_path, session))
            except (ValueError, json.JSONDecodeError):
                continue

        return results
