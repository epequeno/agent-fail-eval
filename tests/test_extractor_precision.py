"""Tests for extractor precision filters."""

import pytest

from agent_fail_eval.evaluator import (
    FactCategory,
    FactExtractor,
    Fact,
    CompactionLossEvaluator,
)
from agent_fail_eval.models import (
    BLOCK_TEXT,
    MESSAGE,
    CompactionBoundary,
    SessionRecord,
)
from agent_fail_eval.evaluator import (
    _is_conversational_noise,
    _is_code_fragment,
    _is_build_progress,
    _is_info_note,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user_record(text: str, rid: str = "u1") -> SessionRecord:
    return SessionRecord(
        type=MESSAGE,
        id=rid,
        message={"role": "user", "content": [{"type": BLOCK_TEXT, "text": text}]},
    )


def make_assistant_record(text: str, rid: str = "a1") -> SessionRecord:
    return SessionRecord(
        type=MESSAGE,
        id=rid,
        message={"role": "assistant", "content": [{"type": BLOCK_TEXT, "text": text}]},
    )


def make_error_record(text: str, rid: str = "e1") -> SessionRecord:
    return SessionRecord(
        type=MESSAGE,
        id=rid,
        message={
            "role": "toolResult",
            "content": [{"type": BLOCK_TEXT, "text": text}],
            "isError": True,
        },
    )


def make_tool_result_record(text: str, rid: str = "t1") -> SessionRecord:
    return SessionRecord(
        type=MESSAGE,
        id=rid,
        message={
            "role": "toolResult",
            "content": [{"type": BLOCK_TEXT, "text": text}],
            "isError": False,
        },
    )


def make_boundary(records: list[SessionRecord]) -> CompactionBoundary:
    return CompactionBoundary(
        compaction_id="c1",
        timestamp="2026-01-01T00:00:00Z",
        summary="test summary",
        first_kept_entry_id="kept1",
        tokens_before=10000,
        file_position=0,
        first_kept_position=1,
        dropped_records=records,
        newly_dropped_count=len(records),
    )


# ---------------------------------------------------------------------------
# Conversational noise filter
# ---------------------------------------------------------------------------

class TestConversationalNoise:
    """Conversational acknowledgments are filtered from user constraints."""

    def test_yes_please(self):
        assert _is_conversational_noise("yes please") is True

    def test_ok_great(self):
        assert _is_conversational_noise("ok great, what's next") is True

    def test_please_proceed(self):
        assert _is_conversational_noise("please proceed") is True

    def test_okay_sounds_good(self):
        assert _is_conversational_noise("sounds good") is True

    def test_genuine_constraint_not_filtered(self):
        assert _is_conversational_noise("never deploy without running tests") is False

    def test_genuine_instruction_not_filtered(self):
        assert _is_conversational_noise("use the env var AWS_PROFILE=arch to get credentials") is False

    def test_longer_constraint_not_filtered(self):
        assert _is_conversational_noise("go ahead and commit but don't push yet, there is more work to be done") is False

    def test_empty_string(self):
        assert _is_conversational_noise("") is True


# ---------------------------------------------------------------------------
# Code fragment filter
# ---------------------------------------------------------------------------

class TestCodeFragment:
    """Bare code fragments are filtered from assistant findings."""

    def test_rust_let(self):
        assert _is_code_fragment("let resource = event.payload.resource.as_deref().unwrap_or(\"\");") is True

    def test_rust_match(self):
        assert _is_code_fragment("match event.payload.resource.as_deref() { Some(r) => r, None => \"\" }") is True

    def test_js_export(self):
        assert _is_code_fragment("export default defineConfig({ plugins: [react()], clearScreen: false })") is True

    def test_swift_guard(self):
        assert _is_code_fragment("guard let lastUserMsg = messages.last, lastUserMsg.isUser else { return }") is True

    def test_function_call(self):
        assert _is_code_fragment("Ok(response::ok(pharmacy_to_json(pharmacy_item, Some(coas))))") is True

    def test_natural_language_not_filtered(self):
        assert _is_code_fragment("The build failed because of a missing dependency in the Rust Lambda function") is False

    def test_heading_not_filtered(self):
        assert _is_code_fragment("### Plan 15 — Pricing Restructure (Three-Tier Model)") is False

    def test_instruction_not_filtered(self):
        assert _is_code_fragment("Now add the same retry+timeout to the OpenRouter provider:") is False

    def test_empty_string(self):
        assert _is_code_fragment("") is True

    def test_natural_language_paragraph(self):
        assert _is_code_fragment(
            "This is a significant finding about the architecture. "
            "The system uses a multi-tenant approach with JWT isolation."
        ) is False


# ---------------------------------------------------------------------------
# Build progress filter
# ---------------------------------------------------------------------------

class TestBuildProgress:
    """Build/compilation progress is filtered from error states."""

    def test_compiling(self):
        assert _is_build_progress("Compiling libc v0.2.186") is True

    def test_building(self):
        assert _is_build_progress("Building for debugging... [0/4] Write sources") is True

    def test_checking(self):
        assert _is_build_progress("Checking aho-corasick v1.1.4") is True

    def test_genuine_error_not_filtered(self):
        assert _is_build_progress("Error: Re-import failed: Importing keystore") is False

    def test_tool_not_found_not_filtered(self):
        assert _is_build_progress("Tool not found") is False

    def test_traceback_not_filtered(self):
        assert _is_build_progress("Traceback (most recent call last):") is False


# ---------------------------------------------------------------------------
# Info note filter
# ---------------------------------------------------------------------------

class TestInfoNote:
    """Informational notes are filtered from error states."""

    def test_note_that(self):
        assert _is_info_note("Note that running a local entrypoint in detached mode only keeps the last triggered function alive") is True

    def test_note_colon(self):
        assert _is_info_note("Note: this feature requires Python 3.11 or later") is True

    def test_genuine_error_not_filtered(self):
        assert _is_info_note("Validation failed for tool 'bash': command required") is False

    def test_could_not_find_not_filtered(self):
        assert _is_info_note("Could not find the exact text in the file") is False


# ---------------------------------------------------------------------------
# FactExtractor integration — noise filtering reduces fact count
# ---------------------------------------------------------------------------

class TestExtractorNoiseFiltering:
    """The extractor filters noise from each category."""

    def setup_method(self):
        self.extractor = FactExtractor()

    def test_user_constraints_skip_acks(self):
        records = [
            make_user_record("yes please", rid="u1"),
            make_user_record("never deploy without running tests", rid="u2"),
        ]
        boundary = make_boundary(records)
        facts = self.extractor.extract(boundary)
        uc_facts = [f for f in facts if f.category == FactCategory.USER_CONSTRAINT]
        # Only the real constraint, not the ack
        assert len(uc_facts) == 1
        assert "never deploy" in uc_facts[0].value

    def test_assistant_findings_skip_code(self):
        records = [
            make_assistant_record(
                "let resource = event.payload.resource.as_deref().unwrap_or(\"\");\n\n"
                "The system uses a multi-tenant approach with JWT isolation for security.",
                rid="a1",
            ),
        ]
        boundary = make_boundary(records)
        facts = self.extractor.extract(boundary)
        af_facts = [f for f in facts if f.category == FactCategory.ASSISTANT_FINDING]
        # Only the natural language paragraph, not the code line
        assert len(af_facts) == 1
        assert "multi-tenant" in af_facts[0].value

    def test_errors_skip_build_progress(self):
        records = [
            make_error_record("Compiling libc v0.2.186\n    Compiling syn v2.0.117", rid="e1"),
            make_error_record("Validation failed for tool 'bash': command required", rid="e2"),
        ]
        boundary = make_boundary(records)
        facts = self.extractor.extract(boundary)
        err_facts = [f for f in facts if f.category == FactCategory.ERROR_STATE]
        # Only the genuine error, not the build progress
        assert len(err_facts) == 1
        assert "Validation failed" in err_facts[0].value

    def test_errors_skip_info_notes(self):
        records = [
            make_error_record(
                "Note that running a local entrypoint in detached mode only keeps the last triggered function alive",
                rid="e1",
            ),
            make_error_record("Could not find the exact text in the file", rid="e2"),
        ]
        boundary = make_boundary(records)
        facts = self.extractor.extract(boundary)
        err_facts = [f for f in facts if f.category == FactCategory.ERROR_STATE]
        assert len(err_facts) == 1
        assert "Could not find" in err_facts[0].value

    def test_tool_result_content_skips_code(self):
        # Use >500 chars to go through the verbose (per-line) path
        code_lines = "\n".join([
            "let resource = event.payload.resource.as_deref().unwrap_or(\"\");",
            "configs_table.table_arn + \"/index/ShareLinkIndex\",",
            "let mut guard = self.db.lock().map_err(|e| e.to_string())?;",
            "export default defineConfig({ plugins: [react()], clearScreen: false, });",
            "guard let lastUserMsg = messages.last, lastUserMsg.isUser else { return }",
            "Ok(response::ok(pharmacy_to_json(pharmacy_item, Some(coas))))",
        ])
        natural_line = "This is a significant finding about the architecture with natural language words here that provides real insight into the system design."
        text = code_lines + "\n" + natural_line
        assert len(text) > 500  # ensure verbose path
        records = [make_tool_result_record(text, rid="t1")]
        boundary = make_boundary(records)
        facts = self.extractor.extract(boundary)
        af_facts = [f for f in facts if f.category == FactCategory.ASSISTANT_FINDING]
        # Code lines filtered, only the natural-language line kept
        assert len(af_facts) >= 1
        assert any("significant finding" in f.value or "natural language" in f.value for f in af_facts)
        assert not any("let resource" in f.value for f in af_facts)


# ---------------------------------------------------------------------------
# _fact_in_summary precision — higher threshold reduces false positives
# ---------------------------------------------------------------------------

class TestFactInSummaryPrecision:
    """The summary-matching threshold is stricter (50% not 30%)."""

    def setup_method(self):
        self.evaluator = CompactionLossEvaluator()

    def _make_fact(self, terms: list[str], value: str = "test fact") -> Fact:
        return Fact(
            category=FactCategory.USER_CONSTRAINT,
            value=value,
            source_record_id="r1",
            source_index=0,
            search_terms=terms,
        )

    def test_low_term_overlap_not_in_summary(self):
        # 10 terms, only 2 in summary — old threshold (3) would match,
        # new threshold (5) does not
        terms = ["deploy", "region", "approved", "polymarket", "production",
                 "staging", "testing", "development", "security", "compliance"]
        fact = self._make_fact(terms, "deploy to approved region")
        summary = "deploy the application to production environment"
        # "deploy" matches, but only 1/10 significant terms — below threshold
        assert self.evaluator._fact_in_summary(fact, summary.lower()) is False

    def test_high_term_overlap_in_summary(self):
        # 4 terms, 3 in summary — 3 >= 2 (50% of 4) → True
        terms = ["deploy", "region", "approved", "polymarket"]
        fact = self._make_fact(terms, "deploy to approved region")
        summary = "deploy to the approved region for polymarket"
        assert self.evaluator._fact_in_summary(fact, summary.lower()) is True

    def test_common_terms_filtered(self):
        # Terms include common words that would trivially match
        terms = ["please", "just", "here", "deploy", "region"]
        fact = self._make_fact(terms, "deploy to region")
        # "please", "just", "here" are filtered out, leaving 2 significant terms
        summary = "please just deploy here to another place"
        # Only "deploy" matches among significant terms → 1 < 1 (max(2, 1)) → False
        assert self.evaluator._fact_in_summary(fact, summary.lower()) is False

    def test_file_reference_still_exact_match(self):
        fact = Fact(
            category=FactCategory.FILE_REFERENCE,
            value="/Users/example/code/project/src/main.rs",
            source_record_id="r1",
            source_index=0,
            search_terms=["/Users/example/code/project/src/main.rs", "main.rs", "main"],
        )
        summary = "Modified main.rs to fix the build error"
        assert self.evaluator._fact_in_summary(fact, summary.lower()) is True

    def test_file_reference_absent(self):
        fact = Fact(
            category=FactCategory.FILE_REFERENCE,
            value="/Users/example/temp/report.txt",
            source_record_id="r1",
            source_index=0,
            search_terms=["/Users/example/temp/report.txt", "report.txt", "report"],
        )
        summary = "Implemented thread-weave architecture for the agent"
        assert self.evaluator._fact_in_summary(fact, summary.lower()) is False

    def test_no_significant_terms_falls_back_to_substring(self):
        fact = self._make_fact([], "a very specific constraint text")
        # No search terms → falls back to substring match
        summary = "the user said a very specific constraint text earlier"
        assert self.evaluator._fact_in_summary(fact, summary.lower()) is True

    def test_no_significant_terms_substring_absent(self):
        fact = self._make_fact([], "a very specific constraint text")
        summary = "completely unrelated summary content"
        assert self.evaluator._fact_in_summary(fact, summary.lower()) is False
