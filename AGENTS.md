# agent-fail-eval

Coding-agent failure evaluation toolkit. Parses pi-agent and Claude Code session logs to detect, evaluate, and report on long-session degradation: compaction loss, context rot, strategy-comparison failures, and harness-adversarial attacks.

## Obsidian Vault

Research notes and business plan: stored in a local Obsidian vault.

## Session Log Location

Pi agent sessions are stored at `~/.pi/agent/sessions/`, organized by project working directory. Each session is a JSONL file with chronologically ordered records.

Claude Code sessions are stored at `~/Library/Application Support/21st-desktop/claude-sessions/<hash>/projects/<project>/<session-id>.jsonl`. The compaction mechanism is a `system` record with `subtype: "compact_boundary"` followed by a synthetic `user` message containing the summary. Use `--harness claude` to select the `ClaudeCodeLoader`.

## Record Types

- `session` — session metadata (id, cwd, version)
- `message` — user/assistant/toolResult messages with content blocks (text, toolCall, thinking)
- `model_change` — model switches during the session
- `thinking_level_change` — reasoning budget changes
- `compaction` — context compaction event with summary, boundary (`firstKeptEntryId`), `tokensBefore`, `details` (readFiles, modifiedFiles), and `fromHook`
- `custom_message` — extension/hook status updates

## Key Abstractions

- **Compaction boundary**: `firstKeptEntryId` marks the first record kept after compaction. Everything before it is dropped and replaced by the `summary`.
- **Dropped segment**: records from the previous boundary (or session start) to this compaction's `firstKeptEntryId`. This is the content at risk of being lost.
- **Visible history at step N**: the records the agent could see at that point — either full history (pre-compaction) or [summary + kept records] (post-compaction). Implemented in `ParsedSession.visible_history_at()`.

## Strategies Evaluated

The evaluator compares four context-management strategies per fact:

1. **Full history** (oracle) — all records available; establishes the ceiling
2. **Compaction only** — only the compaction summary; the current production behavior
3. **Retrieval** — summary + keyword search over the dropped segment; simulates RAG
4. **REPL** — summary + code-based search: full-text matching, verbose-output filtering, structural queries, constraint pattern extraction; simulates the rlm-agents context-externalization pattern

## Fact Categories

The fact extractor pulls testable facts from dropped segments:

- **User constraints** — user decisions, instructions, preferences (from user messages)
- **File references** — file paths from tool call arguments
- **Error states** — failed tool results (from toolResult with isError=True)
- **Assistant findings** — significant text from assistant messages and tool result content

## Attack Fixtures

The `attack` command generates synthetic session fixtures that simulate the compaction-hiding attack (H5a from harness-adversarial):

1. `buried_in_verbose_output` — constraint buried in 200 lines of padding (retrieval fails, REPL recovers)
2. `early_placement` — constraint in step 1-5, needed after compaction (both recover)
3. `single_mention_in_tool_result` — constraint appears once in a tool result (both recover)
4. `control_user_message` — constraint in a user message (survives compaction)

Fixtures are in pi agent JSONL format — parseable by the same `SessionLoader` as real sessions.

## Vulnerability Window Detection

The `vuln` command detects compaction-induced vulnerability windows. After a compaction boundary drops established behavioral/approval constraints, the agent enters a window where adversarial inputs contradicting those constraints may succeed at higher rates.

Pipeline:

1. **ConstraintClassifier** — keyword taxonomy classifies user-message sentences as behavioral_rule, approval_gate, scope_limit, format_requirement, or other. Only behavioral_rule and approval_gate create vulnerability windows.
2. **VulnerabilityWindowDetector** — for each compaction boundary, extracts constraint sentences from `dropped_records` (user messages only), checks survival via keyword overlap with the compaction summary, and flags dropped behavioral/approval constraints.
3. **AdversarialAssertionGenerator** — optional (`--generate`); generates a contradicting statement per dropped constraint via OpenRouter. Generation failures are swallowed (do not abort the scan).

**Commands:**

```
afe vuln ~/.pi/agent/sessions/ -o results/vuln-window-report.json           # scan only
afe vuln ~/.pi/agent/sessions/ --generate -o results/vuln-window-report.json  # with LLM assertions (needs OPENROUTER_API_KEY)
```

## Redaction

The `redactor.py` module provides `SessionRedactor` for sanitizing session content before sharing: file paths → deterministic labels, project names → anonymized labels, code blocks → `[CODE_BLOCK]`, API keys/secrets → `[REDACTED]`, session IDs → hashed. Applied via `afe report --redact`.

## Results Summary

- 115 sessions, 213 compaction boundaries, 170,209 facts (after noise filtering)
- Compaction loses 86% of facts; user constraints 96.5%; error states 93.5%
- Retrieval recovers 64% of lost facts; REPL "recovers 100%" — but this is circular (search terms derived from extracted facts themselves), not a real mitigation result
- Vulnerability windows: 17/115 sessions (14.8%) have a compaction boundary dropping a behavioral or approval constraint; 21 windows, 26 dropped constraints. Report at `results/vuln-window-report.json`.

## Fact Labeling (Priority 1: Precision/Recall Validation)

The headline numbers (94.6% user constraint omission, 84.3% error state omission) are heuristic-extractor output with no relevance filter. The `afe label` tool generates a stratified sample of 150 facts for human labeling to measure extractor precision/recall. **Labeling is complete** (150/150, human-validated); stats at `results/label-stats.json`.

**Stratified sample** (deterministic, seed=42):

| Category | lost | recovered | in_summary | total |
|----------|------|-----------|------------|-------|
| user_constraint | 30 | 15 | 10 | 55 |
| error_state | 25 | 10 | 10 | 45 |
| file_reference | 0 | 10 | 10 | 20 |
| assistant_finding | 15 | 10 | 5 | 30 |
| **Total** | **70** | **45** | **35** | **150** |

File references have zero lost facts (literal paths match trivially).

**Labeling schema** (per fact):
1. `is_meaningful` — genuine discrete information vs noise/fragment
2. `category_correct` — does the assigned category fit the content
3. `in_summary_gt` — human judgment: is this fact in the compaction summary? (ground truth, independent of evaluator keyword matching)
4. `would_loss_matter` — low/medium/high impact if lost at a later decision point
5. `notes` — optional free text

**Commands:**

```
afe label sample ~/.pi/agent/sessions/ -o results/label-sample.json   # generate sample
afe label annotate results/label-sample.json                          # interactive labeling (resumable)
afe label stats results/label-sample.json -o results/label-stats.json # compute precision/recall
```

Sample labeled (150/150). Results: lost-fact precision 92.9%, lost-fact recall 100% (all truly-lost facts caught), extractor precision 68% (32% noise), category accuracy 67%, 7 high-impact truly-lost facts. Noise filters reduced total facts from 284,947 to 170,209 (40% noise removed). See `results/label-stats.json` and README for full breakdown.

## Cross-Harness Scope

Business-plan validation strategy (see `the private research vault`, "Immediate Research Priorities") requires porting to at least one more harness before external claims. **Claude Code is now ported.**

- **Claude Code** — **DONE**. Session data found at `~/Library/Application Support/21st-desktop/claude-sessions/` (not `~/.claude/projects/` — that only has session-index.json metadata). Compaction mechanism: `system` record with `subtype: "compact_boundary"`, `compactMetadata.preTokens`, and a synthetic `user` message immediately after containing the summary (starts with "This session is being continued from a previous conversation..."). Parser: `claude_parser.py` (`ClaudeCodeLoader`). Results: 65 sessions with compaction, 223 boundaries, 28,847 facts. Compaction loses 68.5% of facts; user constraints 65.2%; retrieval recovers 60.8%. Vulnerability windows in 44.6% of sessions (29/65), 51 windows, 78 dropped constraints. Full comparison at `results/cross-harness-comparison.json`.
- **OpenCode** — `~/.local/share/opencode/storage/{session,message}/`, 13 sessions. Different storage shape: one JSON file per message keyed by session ID under a session directory, not a single chronological JSONL. No compaction-equivalent field found in a first pass.
- **Codex (CLI)** — no session/rollout logs found on this machine at all.

### Cross-harness comparison

| Metric | Pi-agent | Claude Code |
|--------|----------|-------------|
| Sessions with compaction | 115 | 65 |
| Compaction boundaries | 213 | 223 |
| Total facts extracted | 284,947 | 28,847 |
| Compaction loss rate | 87.0% | 68.5% |
| User constraint omission | 94.6% | 65.2% |
| Retrieval recovery rate | 69.3% | 60.8% |
| Sessions with vuln windows | 14.8% (17/115) | 44.6% (29/65) |
| Dropped constraints | 27 | 78 |

Note: Claude Code has fewer fact categories (2 vs 4) because `FactExtractor` depends on pi-agent-specific record fields for `file_reference` and `error_state`. The `assistant_finding` category dominates in both harnesses. The `--harness claude` flag on `afe scan`, `afe report`, and `afe vuln` selects the `ClaudeCodeLoader`.
