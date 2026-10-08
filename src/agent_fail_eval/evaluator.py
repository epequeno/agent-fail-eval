"""Compaction-loss evaluator.

Takes a parsed session with compaction boundaries and evaluates which facts
were lost by compaction, and whether retrieval-over-history or REPL/history-query
would recover them.

Strategy comparison:
    Full history    — oracle baseline; all facts available
    Compaction only — only the summary; facts not in summary are lost
    Retrieval       — summary + keyword search over dropped segment
    REPL            — summary + code-based search: full-text matching,
                      verbose-output filtering, structural queries

Fact types extracted from dropped segments:
    - User constraints/decisions (from user messages)
    - File paths referenced (from tool call arguments)
    - Error states (from toolResult with isError=True)
    - Key assistant findings (from assistant text and tool result content)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from agent_fail_eval.models import (
    BLOCK_TEXT,
    BLOCK_TOOL_CALL,
    CompactionBoundary,
    ParsedSession,
    SessionRecord,
)
from agent_fail_eval.parser import SessionLoader

# ---------------------------------------------------------------------------
# Noise filters — reduce false positives in fact extraction
# ---------------------------------------------------------------------------

# Conversational acknowledgments — not constraints
_ACK_RE = re.compile(
    r"^(yes|ok|okay|sure|got it|makes sense|gotcha|understood|"
    r"right|exactly|perfect|great|cool|nice|thanks?|yep|yeah|"
    r"please proceed|please continue|please go ahead|"
    r"what'?s next|continue|go ahead|proceed|"
    r"looks good|sounds good|that works|let'?s do it|"
    r"i think we'?re surfacing"
    r")\b.*",
    re.IGNORECASE,
)

_ACK_MAX_LENGTH = 60

# Build/compilation progress — not errors
_BUILD_PROGRESS_RE = re.compile(
 r"^(Compiling|Building for|Checking|Finished|Downloading|Downloaded|"
 r"Installing|Installed|Updating|Updated|Resolving|Documenting|"
 r"Packaging|Generating|Running\s+\d+\s+test)"
 r"\b",
 re.IGNORECASE,
)

# Informational notes — not errors
_INFO_NOTE_RE = re.compile(
 r"^(Note that|Note:|Please note|Tip:|Warning: not an error)",
 re.IGNORECASE,
)

# Code line detection — for filtering bare code fragments
_CODE_KEYWORD_RE = re.compile(
    r"^(let |const |var |fn |def |pub |impl |struct |enum |mod |use |"
    r"import |export |async |await |guard |match |return |assert_|"
    r"case |switch |if |for |while |loop |trait |type |static |"
    r"#\[|//[/!]?|/\*)"
)

# Natural English words (lowercase, ≥4 chars, alpha only, no underscores)
_NATURAL_WORD_RE = re.compile(r"\b[a-z]{4,}\b")

# Common words excluded from summary-matching (in addition to _extract_search_terms stopwords)
_SUMMARY_STOPWORDS = frozenset({
    "please", "just", "like", "also", "some", "more", "only",
    "here", "there", "what", "when", "where", "which", "while",
    "about", "after", "before", "would", "could", "should",
    "their", "these", "those", "your", "been", "does", "done",
    "want", "need", "make", "into", "they", "them", "then",
    "from", "have", "will", "this", "that", "with", "they",
})


def _is_conversational_noise(text: str) -> bool:
    """True if text is a short conversational acknowledgment, not a constraint."""
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) > _ACK_MAX_LENGTH:
        return False
    return bool(_ACK_RE.match(stripped))


def _is_code_fragment(text: str) -> bool:
    """True if text looks like a bare code line with no standalone meaning."""
    stripped = text.strip()
    if not stripped:
        return True
    # Strong code signals: starts with code keyword
    if _CODE_KEYWORD_RE.match(stripped):
        return True
    # Ends with semicolon (code statement) and is short
    if stripped.endswith(";") and len(stripped) < 200:
        return True
    # Contains namespace separator (::) — strong code signal
    if "::" in stripped and len(stripped) < 200:
        return True
    # High density of parens/brackets/braces — code expression
    bracket_count = stripped.count("(") + stripped.count(")") + \
        stripped.count("[") + stripped.count("]") + \
        stripped.count("{") + stripped.count("}")
    if bracket_count >= 3 and len(stripped) < 200:
        return True
    # Low ratio of natural English words → likely code
    words = stripped.split()
    if words and len(stripped) < 200:
        natural = _NATURAL_WORD_RE.findall(stripped.lower())
        natural = [w for w in natural if w not in _SUMMARY_STOPWORDS]
        if len(natural) / max(1, len(words)) < 0.2:
            return True
    return False


def _is_build_progress(text: str) -> bool:
    """True if text is build/compilation progress output, not an error."""
    return bool(_BUILD_PROGRESS_RE.match(text.strip()))


def _is_info_note(text: str) -> bool:
    """True if text is an informational note, not an error."""
    return bool(_INFO_NOTE_RE.match(text.strip()))

class FactCategory(str, Enum):
    """Categories of facts that can be lost by compaction."""

    USER_CONSTRAINT = "user_constraint"
    FILE_REFERENCE = "file_reference"
    ERROR_STATE = "error_state"
    ASSISTANT_FINDING = "assistant_finding"

class Strategy(str, Enum):
    """Context-management strategies to compare."""

    FULL_HISTORY = "full_history"
    COMPACTION_ONLY = "compaction_only"
    RETRIEVAL = "retrieval"
    REPL = "repl"


class FactStatus(str, Enum):
    """Whether a fact is available under a given strategy."""

    AVAILABLE = "available"
    LOST = "lost"
    RECOVERED = "recovered"


@dataclass
class Fact:
    """A testable fact extracted from the dropped segment.

    A fact is something the agent knew before compaction that might
    be needed later. Each fact has a category, a value (the fact itself),
    and a source (the record it was extracted from).
    """

    category: FactCategory
    value: str
    source_record_id: str | None
    source_index: int  # Position in dropped_records
    # Search terms used for retrieval simulation
    search_terms: list[str] = field(default_factory=list)


@dataclass
class FactResult:
    """The status of a single fact under a single strategy."""

    fact: Fact
    strategy: Strategy
    status: FactStatus
    evidence: str | None = None  # Where it was found (for retrieval)


@dataclass
class CompactionLossReport:
    """Full report for one compaction boundary.

    Attributes:
        boundary: The compaction boundary being evaluated.
        facts: All facts extracted from the dropped segment.
        results: Per-fact, per-strategy status.
    """

    boundary: CompactionBoundary
    facts: list[Fact]
    results: list[FactResult]

    @property
    def facts_by_category(self) -> dict[FactCategory, list[Fact]]:
        """Facts grouped by category."""
        grouped: dict[FactCategory, list[Fact]] = {}
        for f in self.facts:
            grouped.setdefault(f.category, []).append(f)
        return grouped

    def results_for(self, strategy: Strategy) -> list[FactResult]:
        """All fact results for a given strategy."""
        return [r for r in self.results if r.strategy == strategy]

    def summary_stats(self) -> dict[str, dict[str, int]]:
        """Per-strategy: available, lost, recovered counts."""
        stats: dict[str, dict[str, int]] = {}
        for strategy in Strategy:
            results = self.results_for(strategy)
            stats[strategy.value] = {
                "available": sum(1 for r in results if r.status == FactStatus.AVAILABLE),
                "lost": sum(1 for r in results if r.status == FactStatus.LOST),
                "recovered": sum(1 for r in results if r.status == FactStatus.RECOVERED),
                "total": len(results),
            }
        return stats


class FactExtractor:
    """Extracts testable facts from a compaction boundary's dropped segment."""

    # Minimum text length to consider as a "finding"
    MIN_FINDING_LENGTH = 50

    # Patterns for file path extraction
    FILE_PATH_RE = re.compile(
        r"(?:/Users/|/home/|~/)[^\s\"'<>|]+", re.IGNORECASE
    )

    # Patterns for code identifiers (function/variable names)
    IDENTIFIER_RE = re.compile(r"\b([a-z_][a-zA-Z0-9_]{3,})\s*[=:({]")

    def extract(self, boundary: CompactionBoundary) -> list[Fact]:
        """Extract all facts from a compaction boundary's dropped segment."""
        facts: list[Fact] = []

        for i, record in enumerate(boundary.dropped_records):
            if not record.is_message:
                continue

            role = record.role
            if role == "user":
                facts.extend(self._extract_user_constraints(record, i))
            elif role == "assistant":
                facts.extend(self._extract_assistant_findings(record, i))
            elif role == "toolResult":
                facts.extend(self._extract_tool_result_content(record, i))

            # Extract file references from tool calls (any role)
            facts.extend(self._extract_file_references(record, i))

            # Extract error states from tool results
            if record.is_error():
                facts.extend(self._extract_errors(record, i))

        return facts

    def _extract_user_constraints(
        self, record: SessionRecord, index: int
    ) -> list[Fact]:
        """Extract user decisions, constraints, and preferences.

        These are the most critical facts to lose — they represent
        explicit user intent that may be needed later in the session.
        """
        facts: list[Fact] = []
        text = record.text_content.strip()

        if len(text) < 10:
            return facts

        # Split on sentence boundaries for granular facts
        sentences = re.split(r"[.!?]\s+|\n\n+", text)
        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 10:
                continue
            # Skip conversational acknowledgments — not constraints
            if _is_conversational_noise(s):
                continue
            # Extract search terms: significant words
            terms = self._extract_search_terms(s)
            facts.append(
                Fact(
                    category=FactCategory.USER_CONSTRAINT,
                    value=s[:500],
                    source_record_id=record.id,
                    source_index=index,
                    search_terms=terms,
                )
            )

        return facts

    def _extract_file_references(
        self, record: SessionRecord, index: int
    ) -> list[Fact]:
        """Extract file paths from tool call arguments.

        These represent files the agent interacted with — knowing
        which files were touched is critical for later context.
        """
        facts: list[Fact] = []
        seen: set[str] = set()

        for tc in record.tool_calls:
            args = tc.get("arguments", {})
            args_str = str(args)

            paths = self.FILE_PATH_RE.findall(args_str)
            for path in paths:
                # Normalize and deduplicate
                path = path.rstrip(".,;)")
                if path in seen:
                    continue
                seen.add(path)

                # Extract filename for easier matching
                filename = path.rsplit("/", 1)[-1] if "/" in path else path
                terms = [path, filename]
                # Also extract without extension
                if "." in filename:
                    terms.append(filename.rsplit(".", 1)[0])

                facts.append(
                    Fact(
                        category=FactCategory.FILE_REFERENCE,
                        value=path,
                        source_record_id=record.id,
                        source_index=index,
                        search_terms=terms,
                    )
                )

        return facts

    def _extract_tool_result_content(
        self, record: SessionRecord, index: int
    ) -> list[Fact]:
        """Extract significant content from tool results.

        Tool results often contain critical information: security constraints
        in dependency docs, API response details, configuration values.
        For verbose outputs, extract individual lines or short sections that
        contain distinctive terms (not just padding).
        """
        facts: list[Fact] = []
        text = record.text_content.strip()

        if not text or text == "(no output)":
            return facts

        # For short tool results, extract the whole thing as a fact
        if len(text) < self.MIN_FINDING_LENGTH:
            return facts

        if len(text) <= 500:
            terms = self._extract_search_terms(text)
            facts.append(
                Fact(
                    category=FactCategory.ASSISTANT_FINDING,
                    value=text[:500],
                    source_record_id=record.id,
                    source_index=index,
                    search_terms=terms,
                )
            )
            return facts

        # For verbose tool results, extract significant lines
        # (lines with distinctive content, not repetitive padding)
        lines = text.split("\n")
        seen_content: set[str] = set()

        for line in lines:
            line = line.strip()
            if len(line) < self.MIN_FINDING_LENGTH:
                continue
            # Skip repetitive padding lines
            if "dependency output line" in line.lower():
                continue
            # Skip lines that are just config options
            if re.match(r"^\s*line \d+:", line):
                continue
            # Skip bare code fragments
            if _is_code_fragment(line):
                continue
            # Deduplicate
            content_key = line[:100].lower()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            terms = self._extract_search_terms(line)
            facts.append(
                Fact(
                    category=FactCategory.ASSISTANT_FINDING,
                    value=line[:500],
                    source_record_id=record.id,
                    source_index=index,
                    search_terms=terms,
                )
            )

        return facts

    def _extract_errors(
        self, record: SessionRecord, index: int
    ) -> list[Fact]:
        """Extract error states from tool results.

        Errors represent failed attempts that the agent should
        remember to avoid repeating.
        """
        facts: list[Fact] = []
        text = record.text_content.strip()

        if not text or text == "(no output)":
            return facts
        # Skip build/compilation progress — not an error
        if _is_build_progress(text):
            return facts
        # Skip informational notes — not an error
        if _is_info_note(text):
            return facts

        # Truncate long error outputs
        value = text[:500]
        terms = self._extract_search_terms(text)

        facts.append(
            Fact(
                category=FactCategory.ERROR_STATE,
                value=value,
                source_record_id=record.id,
                source_index=index,
                search_terms=terms,
            )
        )

        return facts

    def _extract_assistant_findings(
        self, record: SessionRecord, index: int
    ) -> list[Fact]:
        """Extract significant assistant text as findings.

        These are the agent's own analysis, conclusions, and summaries
        of what it discovered during the dropped segment.
        """
        facts: list[Fact] = []
        text = record.text_content.strip()

        if len(text) < self.MIN_FINDING_LENGTH:
            return facts

        # Split into paragraphs/sentences for granular facts
        paragraphs = re.split(r"\n\n+", text)
        for para in paragraphs:
            p = para.strip()
            if len(p) < self.MIN_FINDING_LENGTH:
                continue
            # Skip bare code fragments — not meaningful findings
            if _is_code_fragment(p):
                continue
            terms = self._extract_search_terms(p)
            facts.append(
                Fact(
                    category=FactCategory.ASSISTANT_FINDING,
                    value=p[:500],
                    source_record_id=record.id,
                    source_index=index,
                    search_terms=terms,
                )
            )

        return facts

    def _extract_search_terms(self, text: str) -> list[str]:
        """Extract significant search terms from text.

        Used for retrieval simulation — these are the terms that
        a keyword search would match against the dropped segment.
        """
        # Extract identifiers (function/variable names)
        identifiers = self.IDENTIFIER_RE.findall(text)
        # Extract file paths
        paths = self.FILE_PATH_RE.findall(text)
        # Extract quoted strings
        quoted = re.findall(r'"([^"]{3,})"', text)
        # Extract backtick-quoted code
        backtick = re.findall(r"`([^`]{3,})`", text)

        terms = set()
        for idf in identifiers:
            # Filter common words
            if idf.lower() not in {
                "the", "and", "for", "not", "but", "with", "that", "this",
                "from", "have", "will", "was", "are", "can", "all", "use",
                "using", "need", "make", "into", "they", "them", "then",
            }:
                terms.add(idf)
        for p in paths:
            terms.add(p)
            filename = p.rsplit("/", 1)[-1]
            terms.add(filename)
        for q in quoted + backtick:
            terms.add(q)

        return list(terms)[:20]  # Cap to avoid explosion


class CompactionLossEvaluator:
    """Evaluates compaction loss across strategies.

    For each fact extracted from the dropped segment:
    - Full history: always available (oracle)
    - Compaction only: available if the fact appears in the summary
    - Retrieval: available if found via keyword search over dropped segment
    """

    def __init__(self) -> None:
        self.extractor = FactExtractor()

    def evaluate_boundary(
        self, boundary: CompactionBoundary
    ) -> CompactionLossReport:
        """Evaluate a single compaction boundary."""
        facts = self.extractor.extract(boundary)
        results: list[FactResult] = []

        # Pre-compute summary check and build retrieval index once
        summary_lower = boundary.summary.lower()
        retrieval_index = self._build_retrieval_index(boundary)

        for fact in facts:
            for strategy in Strategy:
                status, evidence = self._evaluate_fact(
                    fact, strategy, boundary, summary_lower, retrieval_index
                )
                results.append(
                    FactResult(
                        fact=fact,
                        strategy=strategy,
                        status=status,
                        evidence=evidence,
                    )
                )

        return CompactionLossReport(
            boundary=boundary,
            facts=facts,
            results=results,
        )

    def evaluate_session(
        self, session: ParsedSession
    ) -> list[CompactionLossReport]:
        """Evaluate all compaction boundaries in a session."""
        return [
            self.evaluate_boundary(b) for b in session.compaction_boundaries
        ]

    def _evaluate_fact(
        self,
        fact: Fact,
        strategy: Strategy,
        boundary: CompactionBoundary,
        summary_lower: str,
        retrieval_index: dict,
    ) -> tuple[FactStatus, str | None]:
        """Evaluate a single fact under a single strategy.

        Returns (status, evidence) where evidence is the text that
        confirms the fact is available.
        """
        if strategy == Strategy.FULL_HISTORY:
            # Oracle: always available since the fact came from the history
            return FactStatus.AVAILABLE, None

        if strategy == Strategy.COMPACTION_ONLY:
            if self._fact_in_summary(fact, summary_lower):
                return FactStatus.AVAILABLE, boundary.summary[:200]
            return FactStatus.LOST, None

        if strategy == Strategy.RETRIEVAL:
            if self._fact_in_summary(fact, summary_lower):
                return FactStatus.AVAILABLE, boundary.summary[:200]

            evidence = self._retrieve_from_index(fact, retrieval_index)
            if evidence:
                return FactStatus.RECOVERED, evidence

            return FactStatus.LOST, None

        if strategy == Strategy.REPL:
            if self._fact_in_summary(fact, summary_lower):
                return FactStatus.AVAILABLE, boundary.summary[:200]

            evidence = self._repl_retrieve(fact, retrieval_index)
            if evidence:
                return FactStatus.RECOVERED, evidence

            return FactStatus.LOST, None

        return FactStatus.LOST, None

    def _fact_in_summary(self, fact: Fact, summary_lower: str) -> bool:
        """Check if a fact is represented in the compaction summary.

        Uses different matching strategies per fact category:
        - File references: check if the path or filename appears in summary
        - User constraints: check if key search terms appear in summary
        - Errors: check if error text or key terms appear in summary
        - Assistant findings: check if key terms appear in summary
        """
        if fact.category == FactCategory.FILE_REFERENCE:
            for term in fact.search_terms:
                if term.lower() in summary_lower:
                    return True
            if fact.value.lower() in summary_lower:
                return True
            return False

        # Filter common terms before matching — they match trivially
        significant_terms = [
            t for t in fact.search_terms
            if t.lower() not in _SUMMARY_STOPWORDS and len(t) >= 4
        ]
        if not significant_terms:
            # No significant terms — fall back to substring match
            return fact.value[:100].lower() in summary_lower

        matches = sum(
            1 for term in significant_terms if term.lower() in summary_lower
        )
        # Require 50% of significant terms to match (was 30%)
        threshold = max(2, len(significant_terms) // 2)
        return matches >= threshold

    def _build_retrieval_index(
        self, boundary: CompactionBoundary
    ) -> dict:
        """Build an inverted index from dropped records for O(1) retrieval.

        Returns a dict with:
        - "records": list of (record_text_lower, original_text) tuples
        - For each record, the full lowercased text is pre-computed once.

        Retrieval then iterates records (still O(records) per fact) but
        avoids re-extracting and re-lowercasing text for every fact.
        """
        records_data: list[tuple[str, str]] = []
        for record in boundary.dropped_records:
            if not record.is_message:
                continue
            text_parts = []
            for block in record.content_blocks:
                if block.get("type") == BLOCK_TEXT:
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == BLOCK_TOOL_CALL:
                    text_parts.append(str(block.get("arguments", {})))
            original = " ".join(text_parts)
            if not original.strip():
                continue
            records_data.append((original.lower(), original[:300]))
        return {"records": records_data}

    def _retrieve_from_index(
        self, fact: Fact, index: dict
    ) -> str | None:
        """Retrieve the best matching record text for a fact using the index.

        Scans pre-computed record texts for term matches.
        """
        best_score = 0
        best_text: str | None = None

        fact_terms_lower = [t.lower() for t in fact.search_terms]
        fact_value_lower = fact.value.lower()

        for record_text_lower, original_text in index["records"]:
            score = sum(
                1 for term in fact_terms_lower if term in record_text_lower
            )

            if fact.category == FactCategory.FILE_REFERENCE:
                if fact_value_lower in record_text_lower:
                    score += 5

            if score > best_score:
                best_score = score
                best_text = original_text

        min_score = 1
        if (
            fact.category == FactCategory.ASSISTANT_FINDING
            and len(fact.search_terms) > 5
        ):
            min_score = 2

        if best_score >= min_score:
            return best_text
        return None

    def _repl_retrieve(
        self, fact: Fact, index: dict
    ) -> str | None:
        """Simulate REPL/history-query retrieval for a lost fact.

        The REPL strategy simulates what a code-writing model would do:
        1. If the fact has search terms, use them (same as retrieval)
        2. If not, search using the fact's VALUE directly (full-text matching)
        3. For verbose tool results, filter padding before searching
        4. For user constraints, search by structural role and content patterns

        This addresses the gap where keyword retrieval fails because it
        can't extract distinctive terms from plain English text.
        """
        fact_value_lower = fact.value.lower()

        # Strategy 1: If fact has search terms, use retrieval's approach first
        if fact.search_terms:
            result = self._retrieve_from_index(fact, index)
            if result:
                return result

        # Strategy 2: Full-text matching using the fact's value
        # A REPL model would do: `if "do not commit" in record_text`
        # Use a significant substring of the value (not the whole thing,
        # which might be truncated or have formatting differences)
        value_substrings = self._extract_value_substrings(fact.value)

        for record_text_lower, original_text in index["records"]:
            for sub in value_substrings:
                if sub in record_text_lower:
                    return original_text

        # Strategy 3: For verbose tool results, filter padding then search
        # A REPL model would do: `lines = [l for l in text if not is_padding(l)]`
        # then search the filtered lines
        for record_text_lower, original_text in index["records"]:
            if len(original_text) > 500:
                filtered = self._filter_verbose_output(record_text_lower)
                for sub in value_substrings:
                    if sub in filtered:
                        return original_text

        # Strategy 4: Pattern-based matching for user constraints
        # User constraints often contain imperative language
        if fact.category == FactCategory.USER_CONSTRAINT:
            patterns = self._extract_constraint_patterns(fact.value)
            for record_text_lower, original_text in index["records"]:
                for pattern in patterns:
                    if pattern in record_text_lower:
                        return original_text

        return None

    def _extract_value_substrings(self, value: str) -> list[str]:
        """Extract searchable substrings from a fact's value.

        Instead of relying on extracted search terms (identifiers, paths),
        use the fact's own text broken into meaningful substrings.
        This simulates a REPL model doing `if "substring" in text`.
        """
        value = value.strip()
        if not value:
            return []

        substrings: list[str] = []

        # If the value is short enough, use it directly
        if len(value) <= 80:
            substrings.append(value.lower())
            return substrings

        # Break into sentences/phrases
        sentences = re.split(r"[.!?]\s+|\n+", value)
        for s in sentences:
            s = s.strip()
            if len(s) >= 10:
                # Use the first ~60 chars as a search substring
                substrings.append(s[:60].lower())

        # Also try the first 60 chars of the whole value
        if not substrings:
            substrings.append(value[:60].lower())

        return substrings

    def _filter_verbose_output(self, text_lower: str) -> str:
        """Filter padding lines from verbose tool output.

        Simulates a REPL model doing:
            lines = [l for l in text.split("\\n")
                     if not re.match(r"^\\s*line \\d+", l)]
        """
        lines = text_lower.split("\n")
        filtered = []
        for line in lines:
            stripped = line.strip()
            # Skip repetitive padding patterns
            if re.match(r"^\s*line \d+:", stripped):
                continue
            if "dependency output line" in stripped:
                continue
            if re.match(r"^\s*\d+_param=\d+", stripped):
                continue
            # Keep lines with actual content
            if len(stripped) > 5:
                filtered.append(stripped)
        return "\n".join(filtered)

    def _extract_constraint_patterns(self, value: str) -> list[str]:
        """Extract imperative patterns from user constraint text.

        User constraints often use imperative language that a REPL model
        could search for structurally, even without distinctive terms.
        """
        value_lower = value.lower()
        patterns: list[str] = []

        # Common constraint patterns
        imperative_patterns = [
            r"do not\s+\w+",
            r"don'?t\s+\w+",
            r"never\s+\w+",
            r"must\s+\w+",
            r"always\s+\w+",
            r"i want\s+",
            r"please\s+\w+",
            r"need to\s+\w+",
            r"make sure\s+",
            r"ensure\s+\w+",
        ]

        for pattern in imperative_patterns:
            match = re.search(pattern, value_lower)
            if match:
                patterns.append(match.group())

        # Also extract any word sequences of 3+ words
        words = value_lower.split()
        if len(words) >= 3:
            # Use 3-word sliding window
            for i in range(len(words) - 2):
                window = " ".join(words[i : i + 3])
                if len(window) > 10:
                    patterns.append(window)

        return patterns[:10]  # Cap to avoid explosion
