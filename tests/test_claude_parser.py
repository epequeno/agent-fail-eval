"""Tests for the Claude Code session parser."""

import json
from pathlib import Path

import pytest

from agent_fail_eval.claude_parser import ClaudeCodeLoader
from agent_fail_eval.models import (
    BLOCK_TEXT,
    COMPACTION,
    CUSTOM_MESSAGE,
    MESSAGE,
    CompactionBoundary,
    ParsedSession,
    SessionRecord,
)


# ---------------------------------------------------------------------------
# Helpers — build Claude Code JSONL records
# ---------------------------------------------------------------------------

def user_msg(text: str, uuid: str = "u1", parent: str | None = None) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "sessionId": "test-session",
        "cwd": "/tmp/project",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    }


def assistant_msg(text: str, uuid: str = "a1", parent: str | None = None) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:01.000Z",
        "sessionId": "test-session",
        "cwd": "/tmp/project",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


def compact_boundary(
    uuid: str = "c1",
    logical_parent: str = "prev-uuid",
    pre_tokens: int = 150000,
    trigger: str = "auto",
) -> dict:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": uuid,
        "parentUuid": None,
        "logicalParentUuid": logical_parent,
        "timestamp": "2026-01-01T00:30:00.000Z",
        "sessionId": "test-session",
        "cwd": "/tmp/project",
        "content": "Conversation compacted",
        "compactMetadata": {"trigger": trigger, "preTokens": pre_tokens},
    }


def summary_msg(text: str, uuid: str = "s1", parent: str = "c1") -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:30:01.000Z",
        "sessionId": "test-session",
        "cwd": "/tmp/project",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        },
    }


COMPACTION_SUMMARY = (
    "This session is being continued from a previous conversation "
    "that ran out of context. The summary below covers the earlier "
    "portion of the conversation.\n\nSummary of work done."
)


def write_session(path: Path, records: list[dict]) -> Path:
    """Write records as JSONL to a file."""
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

class TestParseRecord:
    """Individual record type parsing."""

    def setup_method(self):
        self.loader = ClaudeCodeLoader()

    def test_user_message(self):
        rec = self.loader._parse_record(user_msg("hello world"))
        assert rec.type == MESSAGE
        assert rec.message["role"] == "user"
        assert "hello world" in rec.text_content

    def test_assistant_message(self):
        rec = self.loader._parse_record(assistant_msg("I'll help with that"))
        assert rec.type == MESSAGE
        assert rec.message["role"] == "assistant"
        assert "I'll help with that" in rec.text_content

    def test_compact_boundary(self):
        rec = self.loader._parse_record(compact_boundary())
        assert rec.type == COMPACTION
        assert rec.timestamp == "2026-01-01T00:30:00.000Z"
        assert rec.tokens_before == 150000
        assert rec.from_hook is True  # trigger=auto

    def test_compact_boundary_manual_trigger(self):
        rec = self.loader._parse_record(compact_boundary(trigger="manual"))
        assert rec.type == COMPACTION
        assert rec.from_hook is False

    def test_queue_operation_skipped(self):
        raw = {"type": "queue-operation", "operation": "dequeue"}
        assert self.loader._parse_record(raw) is None

    def test_other_system_record(self):
        raw = {"type": "system", "subtype": "info", "uuid": "x1"}
        rec = self.loader._parse_record(raw)
        assert rec.type == CUSTOM_MESSAGE
        assert rec.custom_type == "info"

    def test_unknown_type_preserved(self):
        raw = {"type": "unknown_type", "uuid": "z1"}
        rec = self.loader._parse_record(raw)
        assert rec.type == CUSTOM_MESSAGE
        assert rec.custom_type == "unknown_type"

    def test_tool_result_message_classified_as_toolresult(self):
        raw = {
            "type": "user",
            "uuid": "tr1",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result"}
                ],
            },
        }
        rec = self.loader._parse_record(raw)
        assert rec.type == MESSAGE
        assert rec.message["role"] == "toolResult"

    def test_uuid_and_parent_preserved(self):
        rec = self.loader._parse_record(user_msg("hi", uuid="abc", parent="def"))
        assert rec.id == "abc"
        assert rec.parent_id == "def"


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

class TestBoundaryExtraction:
    """Compaction boundary extraction with dropped segments."""

    def setup_method(self):
        self.loader = ClaudeCodeLoader()

    def _make_records(self, n_user: int = 5, n_assistant: int = 5) -> list[SessionRecord]:
        records = []
        for i in range(n_user):
            records.append(self.loader._parse_record(
                user_msg(f"user message {i}", uuid=f"u{i}")
            ))
        for i in range(n_assistant):
            records.append(self.loader._parse_record(
                assistant_msg(f"assistant message {i}", uuid=f"a{i}", parent=f"u{i}")
            ))
        return records

    def test_single_boundary(self):
        records = self._make_records()
        comp = self.loader._parse_record(compact_boundary(uuid="c1"))
        summ = self.loader._parse_record(summary_msg(COMPACTION_SUMMARY, uuid="s1", parent="c1"))
        records.extend([comp, summ])
        records.append(self.loader._parse_record(user_msg("after compaction", uuid="u_after", parent="s1")))

        boundaries = self.loader._extract_boundaries(records)
        assert len(boundaries) == 1
        b = boundaries[0]
        assert len(b.dropped_records) == 10  # 5 user + 5 assistant
        assert b.tokens_before == 150000
        assert b.summary == COMPACTION_SUMMARY
        assert b.first_kept_entry_id == "s1"

    def test_multiple_boundaries(self):
        # First segment
        records = self._make_records(n_user=3, n_assistant=3)
        # First compaction
        comp1 = self.loader._parse_record(compact_boundary(uuid="c1"))
        summ1 = self.loader._parse_record(summary_msg(COMPACTION_SUMMARY, uuid="s1", parent="c1"))
        records.extend([comp1, summ1])
        # More records
        for i in range(4):
            records.append(self.loader._parse_record(
                user_msg(f"post1-{i}", uuid=f"p1u{i}", parent="s1" if i == 0 else f"p1u{i-1}")
            ))
        # Second compaction
        comp2 = self.loader._parse_record(compact_boundary(uuid="c2", logical_parent="p1u3"))
        summ2 = self.loader._parse_record(summary_msg(COMPACTION_SUMMARY + " v2", uuid="s2", parent="c2"))
        records.extend([comp2, summ2])
        # More records after
        records.append(self.loader._parse_record(user_msg("final", uuid="f1", parent="s2")))

        boundaries = self.loader._extract_boundaries(records)
        assert len(boundaries) == 2
        # First boundary drops 6 records (3 user + 3 assistant)
        assert len(boundaries[0].dropped_records) == 6
        # Second boundary drops from s1 to c2 (summ1 + 4 post records = 5)
        assert len(boundaries[1].dropped_records) == 5

    def test_no_boundaries(self):
        records = self._make_records()
        boundaries = self.loader._extract_boundaries(records)
        assert boundaries == []

    def test_dropped_segment_includes_messages(self):
        records = self._make_records(n_user=2, n_assistant=2)
        comp = self.loader._parse_record(compact_boundary(uuid="c1"))
        summ = self.loader._parse_record(summary_msg(COMPACTION_SUMMARY, uuid="s1", parent="c1"))
        records.extend([comp, summ])

        boundaries = self.loader._extract_boundaries(records)
        b = boundaries[0]
        dropped_msgs = b.dropped_messages
        assert len(dropped_msgs) == 4
        roles = {m.role for m in dropped_msgs}
        assert "user" in roles
        assert "assistant" in roles

    def test_summary_not_found(self):
        """If the next record after boundary isn't a summary, summary is empty."""
        records = self._make_records(n_user=3)
        comp = self.loader._parse_record(compact_boundary(uuid="c1"))
        # Next record is NOT a summary — it's a regular assistant message
        records.append(comp)
        records.append(self.loader._parse_record(assistant_msg("not a summary", uuid="a_after", parent="c1")))

        boundaries = self.loader._extract_boundaries(records)
        assert len(boundaries) == 1
        assert boundaries[0].summary == ""

    def test_from_hook_auto(self):
        records = self._make_records(n_user=1)
        comp = self.loader._parse_record(compact_boundary(trigger="auto"))
        summ = self.loader._parse_record(summary_msg(COMPACTION_SUMMARY))
        records.extend([comp, summ])
        boundaries = self.loader._extract_boundaries(records)
        assert boundaries[0].from_hook is True

    def test_from_hook_manual(self):
        records = self._make_records(n_user=1)
        comp = self.loader._parse_record(compact_boundary(trigger="manual"))
        summ = self.loader._parse_record(summary_msg(COMPACTION_SUMMARY))
        records.extend([comp, summ])
        boundaries = self.loader._extract_boundaries(records)
        assert boundaries[0].from_hook is False


# ---------------------------------------------------------------------------
# Full file loading
# ---------------------------------------------------------------------------

class TestLoad:
    """End-to-end file loading."""

    def test_load_valid_session(self, tmp_path):
        records = [
            user_msg("hello", uuid="u1"),
            assistant_msg("hi there", uuid="a1", parent="u1"),
            compact_boundary(uuid="c1"),
            summary_msg(COMPACTION_SUMMARY, uuid="s1", parent="c1"),
            user_msg("continue", uuid="u2", parent="s1"),
        ]
        path = write_session(tmp_path / "test-session.jsonl", records)

        loader = ClaudeCodeLoader()
        session = loader.load(path)

        assert session.session_id == "test-session"
        assert len(session.compaction_boundaries) == 1
        assert session.compaction_boundaries[0].summary == COMPACTION_SUMMARY
        assert len(session.records) == 5

    def test_load_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loader = ClaudeCodeLoader()
        with pytest.raises(ValueError, match="No parseable records"):
            loader.load(path)

    def test_load_session_with_no_compaction(self, tmp_path):
        records = [
            user_msg("hello", uuid="u1"),
            assistant_msg("hi", uuid="a1", parent="u1"),
        ]
        path = write_session(tmp_path / "no-compaction.jsonl", records)

        loader = ClaudeCodeLoader()
        session = loader.load(path)
        assert len(session.compaction_boundaries) == 0

    def test_cwd_extracted_from_records(self, tmp_path):
        records = [
            user_msg("hello", uuid="u1"),
        ]
        path = write_session(tmp_path / "cwd-test.jsonl", records)
        loader = ClaudeCodeLoader()
        session = loader.load(path)
        assert session.cwd == "/tmp/project"

    def test_queue_operations_skipped(self, tmp_path):
        records = [
            {"type": "queue-operation", "operation": "dequeue", "sessionId": "x"},
            user_msg("hello", uuid="u1"),
        ]
        path = write_session(tmp_path / "queue.jsonl", records)
        loader = ClaudeCodeLoader()
        session = loader.load(path)
        assert len(session.records) == 1  # only the user message
        assert session.records[0].type == MESSAGE


# ---------------------------------------------------------------------------
# scan_directory
# ---------------------------------------------------------------------------

class TestScanDirectory:
    """Directory scanning for sessions with compaction."""

    def test_returns_only_sessions_with_compaction(self, tmp_path):
        # Session with compaction
        records_with = [
            user_msg("hello", uuid="u1"),
            assistant_msg("hi", uuid="a1", parent="u1"),
            compact_boundary(uuid="c1"),
            summary_msg(COMPACTION_SUMMARY, uuid="s1", parent="c1"),
        ]
        write_session(tmp_path / "with-compaction.jsonl", records_with)

        # Session without compaction
        records_without = [
            user_msg("hello", uuid="u2"),
            assistant_msg("hi", uuid="a2", parent="u2"),
        ]
        write_session(tmp_path / "no-compaction.jsonl", records_without)

        loader = ClaudeCodeLoader()
        results = loader.scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0][0].name == "with-compaction.jsonl"

    def test_skips_subagent_files(self, tmp_path):
        records = [
            user_msg("hello"),
            compact_boundary(),
            summary_msg(COMPACTION_SUMMARY),
        ]
        subagent_dir = tmp_path / "subagents"
        subagent_dir.mkdir()
        write_session(subagent_dir / "agent-abc.jsonl", records)

        loader = ClaudeCodeLoader()
        results = loader.scan_directory(tmp_path)
        assert len(results) == 0

    def test_handles_corrupt_files_gracefully(self, tmp_path):
        # Valid session
        records = [
            user_msg("hello"),
            compact_boundary(),
            summary_msg(COMPACTION_SUMMARY),
        ]
        write_session(tmp_path / "good.jsonl", records)

        # Corrupt file
        (tmp_path / "bad.jsonl").write_text("not json\n")

        loader = ClaudeCodeLoader()
        results = loader.scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0][0].name == "good.jsonl"
