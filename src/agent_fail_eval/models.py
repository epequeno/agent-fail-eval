"""Data models for parsed pi agent sessions.

Record types in session JSONL files:
    session, message, model_change, thinking_level_change, compaction, custom_message

The file is chronologically ordered. parentId chains are fully resolvable.
Compaction events contain firstKeptEntryId — the first record kept after compaction.
Everything before it is dropped and replaced by the summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# --- Record type constants ---

SESSION = "session"
MESSAGE = "message"
MODEL_CHANGE = "model_change"
THINKING_LEVEL_CHANGE = "thinking_level_change"
COMPACTION = "compaction"
CUSTOM_MESSAGE = "custom_message"

RecordType = Literal[
    "session",
    "message",
    "model_change",
    "thinking_level_change",
    "compaction",
    "custom_message",
]

# Message role constants

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_RESULT = "toolResult"

# Content block type constants

BLOCK_TEXT = "text"
BLOCK_TOOL_CALL = "toolCall"
BLOCK_TOOL_RESULT = "toolResult"
BLOCK_THINKING = "thinking"


@dataclass
class SessionRecord:
    """A single record from a session JSONL file.

    The raw dict is preserved in ``data`` for access to any fields
    not explicitly modeled. Typed fields cover the common cases
    needed for failure analysis.
    """

    type: RecordType
    id: str | None = None
    parent_id: str | None = None
    timestamp: str | None = None

    # session
    cwd: str | None = None
    version: int | None = None

    # message
    message: dict[str, Any] | None = None

    # model_change
    provider: str | None = None
    model_id: str | None = None

    # thinking_level_change
    thinking_level: str | None = None

    # compaction
    summary: str | None = None
    first_kept_entry_id: str | None = None
    tokens_before: int | None = None
    read_files: list[str] | None = None
    modified_files: list[str] | None = None
    from_hook: bool | None = None

    # custom_message
    custom_type: str | None = None

    # Raw data preserved
    data: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_message(self) -> bool:
        return self.type == MESSAGE

    @property
    def is_compaction(self) -> bool:
        return self.type == COMPACTION

    @property
    def role(self) -> str | None:
        """Message role: user, assistant, or toolResult."""
        if self.message is None:
            return None
        return self.message.get("role")

    @property
    def content_blocks(self) -> list[dict[str, Any]]:
        """Content blocks from a message record."""
        if self.message is None:
            return []
        content = self.message.get("content", [])
        if isinstance(content, str):
            return [{"type": BLOCK_TEXT, "text": content}]
        return content if isinstance(content, list) else []

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Tool call blocks from a message record."""
        return [b for b in self.content_blocks if b.get("type") == BLOCK_TOOL_CALL]

    @property
    def text_content(self) -> str:
        """Concatenated text blocks from a message record."""
        parts = [
            b.get("text", "")
            for b in self.content_blocks
            if b.get("type") == BLOCK_TEXT and b.get("text")
        ]
        return "\n".join(parts)

    @property
    def thinking_content(self) -> str | None:
        """Thinking block text from an assistant message, if present."""
        for b in self.content_blocks:
            if b.get("type") == BLOCK_THINKING:
                return b.get("thinking")
        return None

    def tool_name(self) -> str | None:
        """Tool name for toolResult messages."""
        if self.message is None:
            return None
        return self.message.get("toolName")

    def tool_call_id(self) -> str | None:
        """Tool call ID for toolResult messages."""
        if self.message is None:
            return None
        return self.message.get("toolCallId")

    def is_error(self) -> bool | None:
        """Whether a toolResult message is an error."""
        if self.message is None:
            return None
        return self.message.get("isError")


@dataclass
class CompactionBoundary:
    """A compaction event and the content it dropped.

    Attributes:
        compaction_id: ID of the compaction record.
        timestamp: When compaction occurred.
        summary: The LLM-generated summary that replaced the dropped content.
        first_kept_entry_id: ID of the first record kept after compaction.
        tokens_before: Token count that triggered compaction.
        file_position: Index of the compaction record in the session's record list.
        first_kept_position: Index of firstKeptEntryId in the record list.
        dropped_records: Records from the previous boundary (or session start)
            to this compaction's firstKeptEntryId. This is the content at risk.
        newly_dropped_count: Number of records that were still visible before
            this compaction but are now dropped (excludes already-dropped records
            from prior compactions).
        read_files: Files read before compaction (from details).
        modified_files: Files modified before compaction (from details).
        from_hook: Whether compaction was triggered by a hook.
    """

    compaction_id: str
    timestamp: str
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    file_position: int
    first_kept_position: int
    dropped_records: list[SessionRecord]
    newly_dropped_count: int
    read_files: list[str] | None = None
    modified_files: list[str] | None = None
    from_hook: bool | None = None

    @property
    def dropped_messages(self) -> list[SessionRecord]:
        """Message records in the dropped segment."""
        return [r for r in self.dropped_records if r.is_message]

    @property
    def dropped_tool_calls(self) -> list[dict[str, Any]]:
        """All tool calls in the dropped segment."""
        calls = []
        for r in self.dropped_messages:
            calls.extend(r.tool_calls)
        return calls

    @property
    def summary_preview(self) -> str:
        """First 500 chars of the summary."""
        return self.summary[:500] + ("..." if len(self.summary) > 500 else "")


@dataclass
class ParsedSession:
    """A fully parsed pi agent session.

    Records are in chronological order (as stored in the JSONL file).
    Compaction boundaries are extracted and linked to their dropped segments.
    """

    session_id: str
    cwd: str
    started_at: datetime
    records: list[SessionRecord]
    compaction_boundaries: list[CompactionBoundary]

    @property
    def message_records(self) -> list[SessionRecord]:
        """All message records in chronological order."""
        return [r for r in self.records if r.is_message]

    @property
    def total_messages(self) -> int:
        return len(self.message_records)

    @property
    def total_records(self) -> int:
        return len(self.records)

    @property
    def models_used(self) -> list[str]:
        """All models used during the session."""
        return [
            r.model_id
            for r in self.records
            if r.type == MODEL_CHANGE and r.model_id
        ]

    @property
    def thinking_levels(self) -> list[str]:
        """All thinking levels set during the session."""
        return [
            r.thinking_level
            for r in self.records
            if r.type == THINKING_LEVEL_CHANGE and r.thinking_level
        ]

    @property
    def total_tokens_compacted(self) -> int:
        """Sum of tokensBefore across all compaction events."""
        return sum(b.tokens_before for b in self.compaction_boundaries)

    def visible_history_at(self, index: int) -> list[SessionRecord]:
        """Records visible to the agent at the given record index.

        Before any compaction: all records from start to index.
        After compaction C: [C's summary as a synthetic record] +
            records from C's firstKeptEntryId to index.

        If multiple compactions occurred, only the most recent one
        before ``index`` applies.
        """
        if index < 0 or index >= len(self.records):
            return []

        # Find the last compaction boundary before this index
        active_boundary: CompactionBoundary | None = None
        for b in self.compaction_boundaries:
            if b.file_position <= index:
                active_boundary = b
            else:
                break

        if active_boundary is None:
            return self.records[: index + 1]

        # Post-compaction: summary replaces everything before firstKeptEntryId
        summary_record = SessionRecord(
            type=COMPACTION,
            id=active_boundary.compaction_id,
            timestamp=active_boundary.timestamp,
            summary=active_boundary.summary,
        )
        kept = self.records[active_boundary.first_kept_position : index + 1]
        return [summary_record] + kept
