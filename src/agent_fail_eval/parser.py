"""Session log parser for pi agent JSONL files.

Loads a .jsonl session file, reconstructs the chronological history,
and extracts compaction boundaries with their dropped segments.

The file is chronologically ordered; parentId chains are fully resolvable.
No reordering is needed — file order IS the timeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent_fail_eval.models import (
    COMPACTION,
    MESSAGE,
    MODEL_CHANGE,
    SESSION,
    THINKING_LEVEL_CHANGE,
    CompactionBoundary,
    ParsedSession,
    SessionRecord,
)

if TYPE_CHECKING:
    from agent_fail_eval.redactor import SessionRedactor


class SessionLoader:
    """Loads and parses pi agent session JSONL files."""
    def load(
        self, path: str | Path, *, redactor: SessionRedactor | None = None
    ) -> ParsedSession:
        """Parse a session JSONL file into a ParsedSession.

        Args:
            path: Path to the .jsonl session file.
            redactor: Optional SessionRedactor to sanitize records and
                      boundaries before returning. All extracted facts
                      and downstream reports will operate on redacted data.

        Returns:
            ParsedSession with records in chronological order and
            compaction boundaries extracted.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file has no session record or is empty.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        records: list[SessionRecord] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                records.append(self._parse_record(raw))

        if not records:
            raise ValueError(f"Empty session file: {path}")

        # Extract session metadata from the first session record
        session_record = next(
            (r for r in records if r.type == SESSION), None
        )
        if session_record is None:
            raise ValueError(f"No session record found in: {path}")

        # Redact records if a redactor is provided
        if redactor is not None:
            records = [redactor.redact_record(r) for r in records]

        compaction_boundaries = self._extract_boundaries(records)

        started_at = self._parse_timestamp(session_record.timestamp)

        # Build session with redacted cwd
        raw_cwd = session_record.cwd or ""
        clean_cwd = redactor.redact_project(raw_cwd) if redactor else raw_cwd
        session_id = session_record.id or path.stem

        return ParsedSession(
            session_id=session_id,
            cwd=clean_cwd,
            started_at=started_at,
            records=records,
            compaction_boundaries=compaction_boundaries,
        )

    def _parse_record(self, raw: dict) -> SessionRecord:
        """Parse a raw JSON dict into a SessionRecord."""
        record_type = raw.get("type", "")

        details = raw.get("details", {}) or {}

        return SessionRecord(
            type=record_type,
            id=raw.get("id"),
            parent_id=raw.get("parentId"),
            timestamp=raw.get("timestamp"),
            cwd=raw.get("cwd"),
            version=raw.get("version"),
            message=raw.get("message"),
            provider=raw.get("provider"),
            model_id=raw.get("modelId"),
            thinking_level=raw.get("thinkingLevel"),
            summary=raw.get("summary"),
            first_kept_entry_id=raw.get("firstKeptEntryId"),
            tokens_before=raw.get("tokensBefore"),
            read_files=details.get("readFiles") if isinstance(details, dict) else None,
            modified_files=(
                details.get("modifiedFiles") if isinstance(details, dict) else None
            ),
            from_hook=raw.get("fromHook"),
            custom_type=raw.get("customType"),
            data=raw,
        )

    def _extract_boundaries(
        self, records: list[SessionRecord]
    ) -> list[CompactionBoundary]:
        """Extract compaction boundaries with their dropped segments.

        For each compaction event:
        - firstKeptEntryId marks the first record kept after compaction
        - Everything before it is "dropped" and replaced by the summary
        - For multi-compaction sessions, each compaction drops from the
          previous boundary to its own firstKeptEntryId

        The ``dropped_records`` list includes records from the previous
        compaction's firstKeptEntryId (or session start) to this
        compaction's firstKeptEntryId. This includes the previous
        compaction's summary record if one exists.
        """
        # Build ID → index map for O(1) lookup
        id_to_index: dict[str, int] = {}
        for i, r in enumerate(records):
            if r.id:
                id_to_index[r.id] = i

        compaction_records = [
            (i, r) for i, r in enumerate(records) if r.type == COMPACTION
        ]

        boundaries: list[CompactionBoundary] = []
        prev_first_kept_pos = 0  # Start of session or previous boundary

        for comp_idx, comp_rec in compaction_records:
            fke_id = comp_rec.first_kept_entry_id
            if fke_id is None:
                continue

            fke_pos = id_to_index.get(fke_id, -1)
            if fke_pos == -1:
                # firstKeptEntryId not found — skip this compaction
                continue

            # Dropped segment: from previous boundary to firstKeptEntryId
            dropped = records[prev_first_kept_pos:fke_pos]

            # Newly dropped = records that were still visible before this
            # compaction but are now dropped. This is the same as `dropped`
            # because we track from the previous boundary.
            newly_dropped = len(dropped)

            boundary = CompactionBoundary(
                compaction_id=comp_rec.id or "",
                timestamp=comp_rec.timestamp or "",
                summary=comp_rec.summary or "",
                first_kept_entry_id=fke_id,
                tokens_before=comp_rec.tokens_before or 0,
                file_position=comp_idx,
                first_kept_position=fke_pos,
                dropped_records=dropped,
                newly_dropped_count=newly_dropped,
                read_files=comp_rec.read_files,
                modified_files=comp_rec.modified_files,
                from_hook=comp_rec.from_hook,
            )
            boundaries.append(boundary)

            # Next compaction drops from this boundary's firstKeptEntryId
            prev_first_kept_pos = fke_pos

        return boundaries

    def _parse_timestamp(self, ts: str | None) -> datetime:
        """Parse an ISO timestamp string into a datetime."""
        if ts is None:
            return datetime.min
        # Pi timestamps look like "2026-02-26T17:06:50.557Z"
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return datetime.min

    def scan_directory(
        self,
        dir_path: str | Path,
        *,
        redactor: SessionRedactor | None = None,
    ) -> list[tuple[Path, ParsedSession]]:
        """Scan a directory for session files and parse those with compaction.

        Returns a list of (path, parsed_session) for sessions that contain
        at least one compaction event. Sessions without compaction are skipped.

        Args:
            dir_path: Path to the sessions directory.
            redactor: Optional SessionRedactor to sanitize loaded sessions.
        """
        dir_path = Path(dir_path)
        results: list[tuple[Path, ParsedSession]] = []

        for jsonl_path in sorted(dir_path.rglob("*.jsonl")):
            try:
                session = self.load(jsonl_path, redactor=redactor)
                if session.compaction_boundaries:
                    results.append((jsonl_path, session))
            except (ValueError, json.JSONDecodeError):
                continue

        return results
