# agent-fail-eval

Coding-agent failure evaluation toolkit. Parses pi-agent and Claude Code session logs to detect, evaluate, and report on long-session degradation: compaction loss, context rot, strategy-comparison failures, and vulnerability windows.

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
| Compaction only | Only the compaction summary | 14% availability |
| Retrieval | Summary + keyword search over dropped segment | 64% recovery of facts absent from summaries; 28% recovery for user constraints |
| Full-history forensic query | Code-based search over dropped history using full-text matching, verbose filtering, and structural queries | 100% recovery — **but this is circular**: search terms are derived from the extracted facts themselves, so it cannot miss. This is an internal sanity-check upper bound, not evidence of a working deployed mitigation. Do not cite next to real numbers in external material. |

## Key Results (as of 2026-10-08)

From 115 real pi-agent sessions, 213 compaction boundaries, 170,209 heuristic facts (after noise filtering):

- Compaction summaries omit **86%** of extracted facts overall
- Strongest product-relevant signal: 1,531 user constraints extracted; **96.5%** absent from summaries; keyword retrieval recovers only **28.1%** of lost user constraints
- Error states: 1,319 extracted; **93.0%** absent from summaries; keyword retrieval recovers **39.4%**
- Full-history forensic query "recovers" **100%** of extracted facts — **this is a circular upper bound**, not a real mitigation result. The query terms are derived from the facts themselves, so it cannot miss by construction. Do not cite as evidence of a working mitigation.
- Vulnerability windows: **17 of 115 sessions (14.8%)** have at least one compaction boundary that drops a behavioral or approval constraint absent from the summary — 21 windows, 26 dropped constraints. Aggregate counts in `results/cross-harness-comparison.json`.
- **Cross-harness validation**: compaction loss confirmed in a second harness. Claude Code (65 sessions, 223 boundaries, 27,945 facts): **73.6%** of facts omitted by compaction summaries; user constraints **72.7%** omitted; keyword retrieval recovers **62.6%** of lost facts (**69.3%** of lost user constraints). Vulnerability windows in **44.6%** of sessions (29/65), 51 windows, 76 dropped constraints. Full comparison at `results/cross-harness-comparison.json`.


## Methodology: What Counts as a Fact, and What the Numbers Mean

The headline numbers — 96.5% user constraint omission, 86% overall fact loss — are only as meaningful as the definition of "fact." This section defends that definition, explains why keyword retrieval is the baseline, and scopes what the omission rate does and does not claim.

### What is a "fact"?

A **fact** is a discrete piece of information extracted from the compaction-dropped segment that the agent knew before compaction and might need after it. The extractor (`FactExtractor` in `evaluator.py`) pulls four categories from dropped records:

| Category | Extracted from | What it captures | Example |
|----------|--------------|------------------|---------|
| `user_constraint` | User messages | Decisions, requirements, preferences, scope limits | "Don't push to main — use a feature branch" |
| `file_reference` | Tool call arguments | File paths the agent read or modified | `/Users/example/project/src/main.rs` |
| `error_state` | Tool results with `isError=True` | Build failures, test failures, runtime errors | "error: expected `&str`, found `String`" |
| `assistant_finding` | Assistant text + tool result content | Key conclusions, API details, config values | "The rate limit is 100 req/min with a 429 backoff of 60s" |

Extraction is **heuristic** (regex patterns, keyword filters, sentence splitting), not semantic. User messages are split into sentences; each sentence ≥10 chars that isn't a conversational acknowledgment becomes a `user_constraint`. File paths matching `/Users/` or `/home/` or `~/` in tool call arguments become `file_reference` facts. Tool results with `isError=True` become `error_state` facts. Assistant text and tool result content ≥50 chars become `assistant_finding` facts, with bare code fragments filtered out.

Each fact carries `search_terms` — significant words extracted from the fact value, used for the retrieval strategy simulation. These are the terms that a naive keyword-search system would index.

**Why heuristic, not model-based?** The goal is to measure information survival after compaction, not to build a perfect fact extractor. A heuristic extractor is reproducible, auditable, and has no model-specific biases. Its precision is measurable (68% of extracted facts are meaningful; 32% are noise), and the noise is evenly distributed across the survival analysis — it doesn't inflate the omission rate, because noise facts are just as likely to be omitted as real ones. A model-based extractor would need its own validation pipeline, and its errors would be harder to characterize. The heuristic approach trades precision for transparency.

### Why keyword retrieval is a weak baseline (and why that's the point)

The **retrieval strategy** simulates the simplest possible RAG system: after compaction, the agent has the summary plus a keyword search over the dropped segment's records. Matching uses term overlap — if ≥1 of the fact's search terms appears in a dropped record, the fact is "recovered." No embeddings, no reranking, no semantic matching, no query expansion.

This is deliberately weak. The point is to establish a **floor** — the minimum a retrieval-augmented system could plausibly do. If even simple keyword retrieval recovers a fact, any real system (embeddings, reranking, LLM-based search) would too. The gap between this floor and 100% is the opportunity for better context management.

| Harness | Facts lost after compaction | Recovered by keyword retrieval | Still lost |
|---------|---------------------------|-------------------------------|-----------|
| pi-agent | 86% | 64% of lost facts | 36% of lost facts unrecoverable |
| Claude Code | 73.6% | 63% of lost facts | 37% of lost facts unrecoverable |

For user constraints specifically, keyword retrieval recovers only **28.1%** of lost constraints (pi-agent). This is the worst-performing category — constraints are often phrased differently when referenced later ("use a feature branch" vs. "don't push to main"), so exact-term matching misses them. This is exactly the gap that embeddings + reranking would close, and it's why the keyword baseline matters: it shows where lexical matching fails and semantic matching is needed.

### What 96.5% omission does and does not mean

The most cited number from this project is that **96.5% of extracted user constraints from compaction-dropped segments are absent from the compaction summary**. This is precise but easy to misread.

**What it means:**
- Of the 1,531 user constraints the extractor found in segments that compaction dropped, 1,477 do not appear in the compaction summary by keyword match.
- The labeling validation confirms this is exact: lost recall is 100% (every truly-lost fact is caught), lost precision is 92.9% (7 false alarms, all from `file_reference` keyword matching).

**What it does NOT mean:**
- It does **not** mean 96.5% of all user messages are lost. Only messages in compaction-dropped segments are counted. Messages in the kept portion of the session are still visible to the agent.
- It does **not** mean every lost fact causes a task failure. Many constraints are one-time instructions ("use 4-space indentation") that are never needed again. The labeling study found only **7 high-impact truly-lost facts** out of 102 meaningful facts — the rest are low or medium impact.
- It does **not** mean compaction summaries are useless. Summaries preserve some facts (14% overall, and more for `assistant_finding`). The loss is **systematic**, not random — user constraints are the most underrepresented category.

**What it does suggest:**
- Compaction summaries systematically underrepresent user intent. The agent's post-compaction context is biased toward what the model generated (assistant findings) and away from what the user specified (constraints).
- The tail risk is real. The 7 high-impact lost facts include: don't-push constraints, region requirements, credential configurations, feature requirements, decision rules, and a keystore failure that blocks release. Each is a concrete scenario where compaction removed information that would have changed a downstream decision.
- The gap is actionable. Keyword retrieval recovers 28% of lost user constraints; a better retrieval system (embeddings, reranking, or simply passing the full dropped segment) would recover more. The 96.5% omission rate is the problem size, not the solution claim.

**Omission is not harm.** The 96.5% figure sizes the problem; it does not establish a failure rate. Boundary-local replay work (TRACE, below) finds only a weak link between compaction and measurable downstream harm, and a component study of coding harnesses ([2609.20804](https://arxiv.org/abs/2609.20804)) found context management mostly helps by preventing overflow failures.

### Data freshness note

Only aggregate results are published (`results/cross-harness-comparison.json`, `results/label-stats.json`, `results/claude-label-stats.json`), regenerated 2026-10-08 with the v2 extractor. Per-session and per-boundary reports, labeled fact samples, and the underlying session logs are not published because they contain verbatim text from private projects. The synthetic attack fixtures are regenerated locally by `afe attack`. Raw data for reproduction is available on request.

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
- **Lost recall is 100%** — the evaluator catches every truly-lost fact. The reported omission rates (86% overall, 96.5% user constraints, 93.0% error states) are now exact, not just lower bounds.
- **Lost precision is 92.9%** — 7 false alarms, all from `file_reference` category (keyword-matching finds the filename in the summary even though the fact was classified as "lost" by the bucket).
- **Extractor precision is 68%** — 32% of extracted "facts" are noise (conversational acks, bare code fragments, build output). The `assistant_finding` category is worst: 20/30 are bare code lines with no standalone meaning.
- **Category accuracy is 67%** — requests/questions misclassified as user constraints, build output misclassified as error states, math data misclassified as constraints.
- **7 high-impact truly-lost facts** (6 user constraints + 1 error state) — these are the product-critical losses: don't-push constraints, region requirements, credential configs, feature requirements, decision rules, and a keystore failure that blocks release.

**Extractor precision improvements (v2):** noise filters reduced total facts from 284,947 to 170,209 (40% noise removed). Conversational acknowledgments filtered from user constraints, bare code fragments filtered from assistant findings, build progress and informational notes filtered from error states. The `_fact_in_summary` threshold was raised from 30% to 50% term overlap, and common stopwords are now filtered from summary matching.

### Claude Code Labeling (85-fact cross-harness validation)

A second labeling pass on Claude Code sessions (2 categories: `user_constraint` + `assistant_finding`) validates the cross-harness numbers. Stats at `results/claude-label-stats.json`.

| Metric | pi-agent (150 facts) | Claude Code (85 facts) |
|--------|----------------------|----------------------|
| Meaningful | 102/150 (68%) | 60/85 (70.6%) |
| Category correct | 100/150 (67%) | 23/85 (27.1%) |
| Lost precision | 92.9% | **100%** |
| Lost recall | **100%** | 95.7% |
| High-impact lost | 7 | 0 |

**Key differences:**
- **Category accuracy is much lower** (27% vs 67%) — the Claude Code extractor misclassifies most extracted text as `user_constraint` when it's actually work logs, status descriptions, or headings. The pi-agent extractor benefits from record-type fields that distinguish user messages from assistant output; Claude Code's format blurs this distinction.
- **Lost precision is perfect** (100% vs 92.9%) — no false alarms. The 7 pi-agent false alarms were all `file_reference` keyword matches, a category absent in Claude Code.
- **Lost recall drops slightly** (95.7% vs 100%) — 2 false negatives in `assistant_finding`: the evaluator said the fact was in the summary, but human judgment said it wasn't. Both are process narration that the summary paraphrased rather than preserved verbatim.
- **No high-impact facts** (0 vs 7) — the 15 medium-impact facts are all implementation requirements or technical observations. The absence of high-impact lost facts may reflect Claude Code's more aggressive summarization (73.6% vs 86% loss), which preserves more content but at lower fidelity.
- **`assistant_finding` is heavily noisy** — 13 of 30 are meaningful (43%); the rest are process narration ("Let me...", "Now let me..."). This is a Claude Code-specific pattern: the assistant narrates its actions in text blocks, which the extractor treats as findings.


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
- **No downstream task validation** — the methodology measures fact survival after compaction, not whether lost facts cause task failures. Boundary-local replay studies exist (TRACE, Min et al. 2026, [2608.06503](https://arxiv.org/abs/2608.06503); public corpus of 590 AppWorld boundaries analyzed in [2610.08722](https://arxiv.org/abs/2610.08722)) and report that pre-boundary history predicts post-compaction harm only weakly (AUROC 0.66). They use AppWorld, not real coding sessions. The `would_loss_matter` label here is a human-judged proxy only. See [Next Steps](#next-steps).
- **Claude Code partial category coverage** — the FactExtractor's `file_reference` and `error_state` categories depend on pi-agent-specific record fields. Claude Code sessions produce only `user_constraint` and `assistant_finding` facts. Claude Code omission rates are therefore not directly comparable to pi-agent rates at the category level.
- **REPL strategy is circular** — the full-history forensic query "recovers" 100% of facts by construction (search terms are derived from the facts themselves). This is an internal sanity-check upper bound, not evidence of a working mitigation. Do not cite next to real retrieval numbers. Real lossless-pointer or programmatic-context designs exist to compare against: LCM ([2605.04050](https://arxiv.org/abs/2605.04050)), Scroll ([2608.21690](https://arxiv.org/abs/2608.21690)), RLM ([2512.24601](https://arxiv.org/abs/2512.24601)).

## Related Work

Last surveyed 2026-10-08 (arXiv, HF Papers, Semantic Scholar; abstracts only, not a systematic review, Semantic Scholar citation pass was partial due to rate limiting).

**Closest prior work — compaction loses constraints**
- *The Compaction Cliff in Long-Running AI Agent Memory* ([2608.22752](https://arxiv.org/abs/2608.22752)): reports that Claude Code `/compact` preserves 53% of safety rules after one round and 10% after five (20 production agent configurations), and proposes type-aware retention. Differs from this project: curated rule sets and repeated rounds, versus real session logs, extracted facts, and single boundaries here. Directionally consistent with the user-constraint omission measured here.
- *Nautilus Compass* ([2605.09863](https://arxiv.org/abs/2605.09863)): states that production coding agents forget user-specified constraints over long sessions; drift detector built from real Claude Code traces.

**Boundary-local evaluation**
- TRACE ([2608.06503](https://arxiv.org/abs/2608.06503)): evaluates individual compaction events via paired closed-loop continuations; reports compression weakening the influence of recent interactions (more blocked and repeated actions). Corpus analysis: [2610.08722](https://arxiv.org/abs/2610.08722). Domain is AppWorld.

**Security at the compression boundary**
- *Relinking* ([2606.21732](https://arxiv.org/abs/2606.21732)): the summarizer assembles benign fragments into a malicious instruction (86.9% reported success). The vulnerability windows here are the complementary failure: the summarizer drops constraints, which opens an attack surface.

**Alternatives to lossy summarization**
- LCM ([2605.04050](https://arxiv.org/abs/2605.04050)), Scroll ([2608.21690](https://arxiv.org/abs/2608.21690)), RLM ([2512.24601](https://arxiv.org/abs/2512.24601)): lossless pointers or programmatic context.
- *The Complexity Trap* ([2508.21433](https://arxiv.org/abs/2508.21433)): observation masking matches LLM summarization on SWE-bench Verified at lower cost.
- *An Empirical Study of Harness Design for Coding Agents* ([2609.20804](https://arxiv.org/abs/2609.20804)): staged elision then summarization is most efficient; making elided content recoverable yielded no accuracy gain because models rarely use it. This challenges the assumption here that better retrieval would close the gap.

Not found in this survey (title/abstract search only, so not a novelty claim): per-fact survival rates across real compaction summaries from two harnesses with a human-labeled precision/recall check on the extractor.

## Next Steps

Prioritized in light of the TRACE papers and The Compaction Cliff. All are proposals, none implemented.

1. **Observational downstream check (no re-execution).** For each vulnerability window, test whether post-boundary actions in the same log contradict the dropped constraint (e.g. a push after a dropped don't-push rule). Uses existing logs and turns the 7 high-impact labels into a measured violation rate. TRACE replays from re-executed state; session logs here cannot be re-executed, so this is the feasible approximation.
2. **Replay the public TRACE corpus through `afe`.** Run the fact extractor on its 590 boundaries and test whether fact-loss rate predicts the next-action burden TRACE records. This tests whether omission predicts harm in any setting, since [2610.08722](https://arxiv.org/abs/2610.08722) found the history-based predictors weak.
3. **Multi-round survival.** The Compaction Cliff reports decay across repeated rounds. Track each constraint across successive boundaries in the same session instead of one boundary at a time.
4. **Replace the circular REPL baseline** with a real comparator: observation masking (Complexity Trap), type-aware retention (Compaction Cliff), or lossless pointers (LCM).
5. **Replace keyword `_fact_in_summary` matching** with an LLM or embedding judge validated against the human labels, and add a second labeler for Cohen's kappa.
6. **Fix Claude Code extractor category accuracy** (27% in the 85-fact labeling) before using per-category Claude numbers externally.
7. **Widen operators.** All 180 sessions come from one developer.

## Research Context

This project is the Phase 0 Evaluator Prototype from a broader research program on context management for coding agents. It provides prototype evidence that:

- Real compaction summaries omit many extracted facts, including user constraints and error states
- Retrieval-over-history has a measurable recovery gap, especially for user constraints
- Full-history forensic query is a circular upper bound, not a proven deployed mitigation
- Synthetic harness-adversarial omission fixtures are product-visible and reportable

Related research threads (notes in a private Obsidian vault):

- **rlm-switch** — routing between passthrough and REPL context strategies
- **rlm-agents** — context externalization for agent histories
- **harness-adversarial** — H5a compaction information hiding as an attack class
- **business-plan** — product strategy, validation approach, and competitor teardown
