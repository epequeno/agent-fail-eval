"""Tests for session redaction."""

import pytest

from agent_fail_eval.models import (
    BLOCK_TEXT,
    BLOCK_TOOL_CALL,
    COMPACTION,
    MESSAGE,
    SESSION,
    SessionRecord,
)
from agent_fail_eval.redactor import SessionRedactor


def make_record(
    record_type: str = MESSAGE,
    role: str = "assistant",
    text: str = "",
    tool_args: dict | None = None,
    cwd: str = "",
    summary: str = "",
    read_files: list[str] | None = None,
    modified_files: list[str] | None = None,
) -> SessionRecord:
    """Build a minimal SessionRecord for testing."""
    message = None
    if record_type == MESSAGE:
        content_blocks = []
        if text:
            content_blocks.append({"type": BLOCK_TEXT, "text": text})
        if tool_args:
            content_blocks.append(
                {"type": BLOCK_TOOL_CALL, "name": "read", "arguments": tool_args}
            )
        message = {"role": role, "content": content_blocks}

    return SessionRecord(
        type=record_type,
        id="test-id",
        cwd=cwd,
        message=message,
        summary=summary,
        read_files=read_files,
        modified_files=modified_files,
    )


class TestPathRedaction:
    """File path redaction is deterministic and irreversible."""

    def test_absolute_paths_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="Read file /Users/alice/code/myproject/src/main.py",
            cwd="/Users/alice/code/myproject",
        )
        clean = r.redact_record(record)
        assert "/Users/" not in clean.text_content
        assert "file_" in clean.text_content
        assert clean.cwd != "/Users/alice/code/myproject"

    def test_tilde_paths_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="Config at ~/.config/agent/settings.toml",
        )
        clean = r.redact_record(record)
        assert "~/" not in clean.text_content
        assert "file_" in clean.text_content

    def test_home_paths_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="Check /home/bob/docs/notes.md",
        )
        clean = r.redact_record(record)
        assert "/home/" not in clean.text_content
        assert "file_" in clean.text_content

    def test_path_deterministic(self):
        """Same path → same label within one redactor."""
        r = SessionRedactor(salt="test")
        record = make_record(
            text="File: /Users/alice/src/a.py and /Users/alice/src/a.py",
        )
        clean = r.redact_record(record)
        # Count occurrences of the same redacted label
        text = clean.text_content
        import re
        labels = re.findall(r"\[file_f_[a-f0-9]+\]", text)
        assert len(labels) == 2
        assert labels[0] == labels[1]

    def test_extension_preserved(self):
        """File extension is preserved for readability."""
        r = SessionRedactor(salt="test")
        record = make_record(
            text="Open /Users/alice/src/main.py",
        )
        clean = r.redact_record(record)
        assert ".py" in clean.text_content

    def test_cwd_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            cwd="/Users/alice/code/my-secret-project",
        )
        clean = r.redact_record(record)
        assert clean.cwd.startswith("project_")
        assert "my-secret-project" not in clean.cwd


class TestSecretRedaction:
    """API keys, tokens, and secrets are redacted."""

    def test_openai_key_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="API key: sk-abcdefghijklmnopqrstuvwxyz123456",
        )
        clean = r.redact_record(record)
        assert "sk-" not in clean.text_content
        assert "[REDACTED]" in clean.text_content

    def test_bearer_token_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
        )
        clean = r.redact_record(record)
        assert "Bearer" not in clean.text_content
        assert "[REDACTED]" in clean.text_content

    def test_key_value_secret_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text='api_key="super-secret-value-12345678"',
        )
        clean = r.redact_record(record)
        assert "[REDACTED]" in clean.text_content

    def test_short_strings_not_falsely_redacted(self):
        """Short strings that look like tokens should not be redacted."""
        r = SessionRedactor(salt="test")
        record = make_record(
            text="The variable name is api_key for clarity.",
        )
        clean = r.redact_record(record)
        assert "api_key" in clean.text_content
        # The short value "for clarity" is not a secret


class TestCodeBlockRedaction:
    """Triple-backtick code blocks are redacted."""

    def test_fenced_code_block_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="Here is the code:\n```python\ndef secret():\n    return 'hidden'\n```\nEnd.",
        )
        clean = r.redact_record(record)
        assert "```" not in clean.text_content
        assert "[CODE_BLOCK]" in clean.text_content
        assert "End." in clean.text_content

    def test_multiple_code_blocks(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="Block1:\n```\ncode1\n```\nBlock2:\n```\ncode2\n```",
        )
        clean = r.redact_record(record)
        assert clean.text_content.count("[CODE_BLOCK]") == 2


class TestProjectRedaction:
    """Project names are anonymized sequentially."""

    def test_project_names_sequential(self):
        r = SessionRedactor(salt="test")
        p1 = r.redact_project("my-secret-project")
        p2 = r.redact_project("another-project")
        assert p1 == "project_001"
        assert p2 == "project_002"

    def test_project_deterministic(self):
        r = SessionRedactor(salt="test")
        a = r.redact_project("foo")
        b = r.redact_project("foo")
        assert a == b

    def test_project_from_path(self):
        """Project extraction from session paths."""
        r = SessionRedactor(salt="test")
        path = "/Users/example/.pi/agent/sessions/--Users-example-code-my-project--/session.jsonl"
        label = r.redact_project_name(path)
        assert label.startswith("project_")


class TestSummaryRedaction:
    """Compaction summaries are redacted."""

    def test_summary_paths_redacted(self):
        r = SessionRedactor(salt="test")
        summary = "Modified /Users/alice/src/main.py and /home/bob/config.toml"
        record = make_record(record_type=COMPACTION, summary=summary)
        clean = r.redact_record(record)
        assert "/Users/" not in clean.summary
        assert "/home/" not in clean.summary

    def test_summary_plain_text_untouched(self):
        r = SessionRedactor(salt="test")
        summary = "The agent completed the task successfully."
        record = make_record(record_type=COMPACTION, summary=summary)
        clean = r.redact_record(record)
        assert clean.summary == summary


class TestFileListRedaction:
    """read_files and modified_files lists are redacted."""

    def test_read_files_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            read_files=["/Users/alice/src/main.py", "/tmp/output.log"],
        )
        clean = r.redact_record(record)
        for f in clean.read_files:
            assert "/Users/" not in f
            assert "/tmp/" not in f
            assert "file_" in f

    def test_modified_files_redacted(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            modified_files=["/Users/alice/src/main.py"],
        )
        clean = r.redact_record(record)
        for f in clean.modified_files:
            assert "/Users/" not in f


class TestIdempotency:
    """Redaction is idempotent — redacting redacted content is safe."""

    def test_double_redaction_safe(self):
        r = SessionRedactor(salt="test")
        record = make_record(
            text="File: /Users/alice/src/main.py, key: sk-abc123def456",
            cwd="/Users/alice/code/project",
        )
        clean1 = r.redact_record(record)
        clean2 = r.redact_record(clean1)
        # Text should be stable (no paths/secrets left to redact)
        assert clean2.text_content == clean1.text_content
        # Project names are re-mapped (project_001 → project_002), which is safe
        assert clean2.cwd.startswith("project_")
        assert "[REDACTED]" not in clean2.cwd  # Not falsely redacted as secret


class TestNonMessageRecords:
    """Session and model_change records pass through safely."""

    def test_session_record_passes(self):
        r = SessionRedactor(salt="test")
        record = make_record(record_type=SESSION, cwd="/Users/alice/code/project")
        clean = r.redact_record(record)
        assert clean.cwd.startswith("project_")
        assert clean.type == SESSION

    def test_no_crash_on_none_fields(self):
        r = SessionRedactor(salt="test")
        record = SessionRecord(type="custom_message", custom_type="status")
        clean = r.redact_record(record)
        assert clean.type == "custom_message"