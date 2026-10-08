# agent-fail-eval

Coding-agent failure evaluation toolkit. Parses pi agent session logs to detect, evaluate, and report on long-session degradation: compaction loss, context rot, strategy-comparison failures, and harness-adversarial attacks.

## Quick Start

```bash
cd agent-fail-eval
uv sync

# Parse a single session and print a summary
uv run afe parse ~/.pi/agent/sessions/--some-project--/2026-XX-XX...jsonl

# Scan all sessions for compaction events
uv run afe scan ~/.pi/agent/sessions/

# Evaluate compaction loss on a specific session (with strategy comparison)
uv run afe eval ~/.pi/agent/sessions/--some-project--/2026-XX-XX...jsonl --compaction 0

# Generate an aggregate report across all sessions
uv run afe report ~/.pi/agent/sessions/ -o results/compaction-loss-report.json

# Run compaction-hiding attack fixtures (H5a)
uv run afe attack

# Detect compaction-induced vulnerability windows across sessions
uv run afe vuln ~/.pi/agent/sessions/ -o results/vuln-window-report.json


# Generate a stratified fact sample for precision/recall labeling
uv run afe label sample ~/.pi/agent/sessions/ -o results/label-sample.json

# Interactive labeling (resumable — run this to annotate the sample)
uv run afe label annotate results/label-sample.json

# Compute precision/recall after labeling
uv run afe label stats results/label-sample.json -o results/label-stats.json
```

## Commands

| Command | Description |
|---------|-------------|
| `afe parse <file> [--compaction N] [--harness H]` | Parse a session and print summary with compaction boundaries |
| `afe scan <dir> [--harness H]` | Find all sessions with compaction events |
| `afe eval <file> [--compaction N] [-v]` | Evaluate compaction loss: per-fact, per-strategy failure report |
| `afe report <dir> [-o file.json] [--harness H]` | Aggregate compaction-loss report across all sessions |
| `afe attack` | Run compaction-hiding attack fixtures (H5a) with 4 variants |
| `afe vuln <dir> [--generate] [-o file] [--harness H]` | Detect compaction-induced vulnerability windows: dropped constraints that create adversarial-attack surface |
| `afe label sample <dir> [-o file]` | Generate a stratified sample of 150 facts for human labeling |
| `afe label annotate <file> [-o file]` | Interactive labeling session over a sample (resumable) |
| `afe label stats <file> [-o file]` | Compute precision/recall of the heuristic extractor vs human labels |

## What It Does

1. **Parses** pi-agent and Claude Code `.jsonl` session logs into structured data (SessionRecord, CompactionBoundary, ParsedSession) — use `--harness pi` (default) or `--harness claude`
2. **Reconstructs** the chronological message history with parentId chains
3. **Identifies** compaction boundaries (`firstKeptEntryId`) and extracts dropped segments
4. **Extracts** heuristic facts from dropped content: user constraints, file references, error states, assistant findings, tool result content
5. **Evaluates** fact survival across four strategies: full history (oracle), compaction only, keyword retrieval, full-history forensic query
6. **Reports** per-session, per-boundary, and aggregate failure reports with JSON export
7. **Demonstrates** compaction-omission reporting with synthetic adversarial fixtures: constraint buried in verbose output, early placement, single mention, control
8. **Labels** a stratified sample of extracted facts to measure heuristic extractor precision/recall — the headline numbers are heuristic output and need human-labeled validation before external claims

## Strategy Comparison

| Strategy | What it simulates | Current result |
|----------|------------------|----------------|
| Full history | Oracle baseline — all extracted facts available | 100% availability (trivially — facts came from this history) |
| Compaction only | Only the compaction summary | 13% availability |
| Retrieval | Summary + keyword search over dropped segment | 69% recovery of facts absent from summaries; 22% recovery for user constraints |
| Full-history forensic query | Code-based search over dropped history using full-text matching, verbose filtering, and structural queries | 100% recovery — **but this is circular**: search terms are derived from the extracted facts themselves, so it cannot miss. This is an internal sanity-check upper bound, not evidence of a working deployed mitigation. Do not cite next to real numbers in external material. |

## Key Results (as of 2026-07-09)

From 115 real pi-agent sessions, 213 compaction boundaries, 170,209 heuristic facts (after noise filtering):

- Compaction summaries omit **86%** of extracted facts overall
- Strongest product-relevant signal: 1,531 user constraints extracted; **96.5%** absent from summaries; keyword retrieval recovers only **28.1%** of lost user constraints
- Error states: 1,319 extracted; **93.5%** absent from summaries; keyword retrieval recovers **39.7%**
- Full-history forensic query "recovers" **100%** of extracted facts — **this is a circular upper bound**, not a real mitigation result. The query terms are derived from the facts themselves, so it cannot miss by construction. Do not cite as evidence of a working mitigation.
- Vulnerability windows: **17 of 115 sessions (14.8%)** have at least one compaction boundary that drops a behavioral or approval constraint absent from the summary — 21 windows, 26 dropped constraints. Report at `results/vuln-window-report.json`.
- **Cross-harness validation**: compaction loss confirmed in a second harness. Claude Code (65 sessions, 223 boundaries, 28,847 facts): **68.5%** of facts omitted by compaction summaries; user constraints **65.2%** omitted; retrieval recovers **60.8%** of lost facts. Vulnerability windows in **44.6%** of sessions (29/65), 51 windows, 78 dropped constraints. Full comparison at `results/cross-harness-comparison.json`.

## Fact Labeling (Precision/Recall Validation)

The `afe label` tool generates a stratified sample of 150 facts (category × compaction-status) for human labeling to measure extractor precision/recall.

**Stratified sample** (deterministic, seed=42):

| Category | lost | recovered | in_summary | total |
|----------|------|-----------|------------|-------|
| user_constraint | 30 | 15 | 10 | 55 |
| error_state | 25 | 10 | 10 | 45 |
| file_reference | 0 | 10 | 10 | 20 |
| assistant_finding | 15 | 10 | 5 | 30 |
| **Total** | **70** | **45** | **35** | **150** |

**Labeling schema** (per fact):
1. `is_meaningful` — genuine discrete information vs noise/fragment
2. `category_correct` — does the assigned category fit the content
3. `in_summary_gt` — human judgment: is this fact in the compaction summary? (ground truth, independent of evaluator keyword matching)
4. `would_loss_matter` — low/medium/high impact if lost at a later decision point
5. `notes` — optional free text

Sample labeled (150/150, human-validated). Stats at `results/label-stats.json`.

**Labeling results** (150 facts, 102 meaningful):

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Meaningful facts | 102/150 (68%) | 48 extracted facts are noise/fragments — extractor precision proxy |
| Category correct | 100/150 (67%) | 1/3 of facts misclassified by the heuristic category |
| Lost precision | 92.9% (92/99) | 7 false alarms (evaluator says lost but fact is in summary) |
| Lost recall | **100%** (92/92) | Every truly-lost fact is caught — reported omission rates are exact |

Per-category breakdown (meaningful facts only):

| Category | N | Lost (GT) | Precision | Recall | High/Med/Low |
|----------|---|----------|-----------|--------|--------------|
| user_constraint | 47 | 46 | 100% | 100% | 6/27/13 |
| error_state | 24 | 24 | 100% | 100% | 1/15/8 |
| file_reference | 20 | 11 | 61.1% | 100% | 0/6/5 |
| assistant_finding | 11 | 11 | 100% | 100% | 0/8/3 |

**Key findings:**
- **Lost recall is 100%** — the evaluator catches every truly-lost fact. The reported omission rates (86% overall, 96.5% user constraints, 93.5% error states) are now exact, not just lower bounds.
- **Lost precision is 92.9%** — 7 false alarms, all from `file_reference` category (keyword-matching finds the filename in the summary even though the fact was classified as "lost" by the bucket).
- **Extractor precision is 68%** — 32% of extracted "facts" are noise (conversational acks, bare code fragments, build output). The `assistant_finding` category is worst: 20/30 are bare code lines with no standalone meaning.
- **Category accuracy is 67%** — requests/questions misclassified as user constraints, build output misclassified as error states, math data misclassified as constraints.
- **7 high-impact truly-lost facts** (6 user constraints + 1 error state) — these are the product-critical losses: don't-push constraints, region requirements, credential configs, feature requirements, decision rules, and a keystore failure that blocks release.

**Extractor precision improvements (v2):** noise filters reduced total facts from 284,947 to 170,209 (40% noise removed). Conversational acknowledgments filtered from user constraints, bare code fragments filtered from assistant findings, build progress and informational notes filtered from error states. The `_fact_in_summary` threshold was raised from 30% to 50% term overlap, and common stopwords are now filtered from summary matching.


## Architecture

```
src/agent_fail_eval/
├── models.py      — SessionRecord, CompactionBoundary, ParsedSession
├── parser.py      — SessionLoader (pi-agent JSONL → ParsedSession)
├── claude_parser.py — ClaudeCodeLoader (Claude Code JSONL → ParsedSession)
├── evaluator.py   — FactExtractor, CompactionLossEvaluator (4 strategies)
├── report.py      — ReportGenerator, AggregateReport, print_report
├── attack.py      — FixtureGenerator, AttackEvaluator (4 H5a variants)
├── vuln_window.py — ConstraintClassifier, VulnerabilityWindowDetector, AdversarialAssertionGenerator
├── redactor.py    — SessionRedactor (path/secret/code redaction for safe sharing)
├── label.py        — sample generation, interactive labeling, precision/recall stats
└── cli.py         — afe parse / scan / eval / report / attack / vuln / label


## Limitations

This is a **single-operator case study**, not a controlled experiment. The results demonstrate the methodology and show consistent patterns across two harnesses, but have known limitations:

- **Single operator** — all 180 sessions (115 pi-agent + 65 Claude Code) are from one developer. Compaction behavior may differ across operators with different session lengths, tool usage patterns, or project structures. Widening to additional operators requires only their session data — the parsers and evaluator run without code changes.
- **Single labeler** — the 150-fact precision/recall validation was labeled by one person. Inter-rater reliability (Cohen's κ) requires a second labeler on the same sample. The `afe label annotate` tool supports this (resumable, same JSON format).
- **Heuristic fact extractor** — the FactExtractor uses keyword and regex patterns, not semantic understanding. Extractor precision is 68% (32% noise); category accuracy is 67% (1/3 misclassified). The noise filters (v2) reduce false positives but do not eliminate them. A model-based extractor would likely improve precision but would require its own validation.
- **Heuristic constraint classifier** — the vulnerability window detector uses keyword taxonomy (behavioral_rule, approval_gate, scope_limit, format_requirement). It misclassifies some error messages and non-constraint text. An error-message negative filter reduces but does not eliminate false positives. The reported vulnerability window counts (17/115 pi sessions, 29/65 Claude Code) are not human-validated.
- **No downstream task validation** — the methodology measures fact survival after compaction, not whether lost facts actually cause task failures. A lost user constraint that is never needed again is harmless; one that is needed at the next decision point is critical. The labeling includes a `would_loss_matter` field (low/medium/high) as a proxy, but actual task-level outcome tracking is a different kind of study.
- **Claude Code partial category coverage** — the FactExtractor's `file_reference` and `error_state` categories depend on pi-agent-specific record fields. Claude Code sessions produce only `user_constraint` and `assistant_finding` facts. Claude Code omission rates are therefore not directly comparable to pi-agent rates at the category level.
- **REPL strategy is circular** — the full-history forensic query "recovers" 100% of facts by construction (search terms are derived from the facts themselves). This is an internal sanity-check upper bound, not evidence of a working mitigation. Do not cite next to real retrieval numbers.

## Business Plan Connection


This is the Phase 0 Evaluator Prototype from the business plan. It provides prototype evidence that:

- Real pi-agent compaction summaries omit many extracted facts, including user constraints and error states
- Retrieval-over-history has a measurable recovery gap, especially for user constraints
- Full-history forensic query is a circular upper bound, not a proven deployed mitigation
- Synthetic harness-adversarial omission fixtures are product-visible and reportable
- Competitor teardown confirms the wedge is defensible — no existing tool measures compaction loss, fact survival, or context-strategy comparison

## Research Connections

- rlm-switch — routing between passthrough and REPL
- rlm-agents — context externalization for agent histories
- harness-adversarial — H5a compaction information hiding
- business-plan — product strategy, validation approach, and competitor teardown
