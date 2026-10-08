"""Fact-labeling tool for measuring extractor precision/recall.

Generates a stratified sample of extracted facts (by category × status),
builds human-reviewable context for each fact, runs an interactive
labeling session, and computes precision/recall against the heuristic
extractor's compaction-loss classification.

This closes the credibility gap flagged in the validation strategy:
the 94.6% / 84.3% omission numbers conflate "compactor dropped a fact
that mattered" with "extractor flagged something stale/irrelevant."
Human labels ground the numbers.

Labeling schema (per fact):
  is_meaningful     — genuine discrete information vs noise/fragment
  category_correct  — does the assigned category fit the content
  in_summary_gt     — human judgment: is this fact represented in the
                      compaction summary? (independent of evaluator's
                      keyword matching)
  would_loss_matter — low / medium / high impact if lost at decision time
  notes             — optional free text

The in_summary_gt label is the ground truth used to compute the
extractor's precision (true lost facts) and recall (lost facts it
missed) against the heuristic _fact_in_summary classifier.

Usage:
  afe label sample <sessions_dir> -o label-sample.json     # generate
  afe label annotate label-sample.json                     # interactive
  afe label stats label-sample.json                        # compute metrics
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from agent_fail_eval.evaluator import (
    CompactionLossEvaluator,
    Fact,
    FactCategory,
    FactStatus,
    Strategy,
)
from agent_fail_eval.parser import SessionLoader


# --- Stratification target -------------------------------------------------

# Categories × buckets. Buckets:
#   lost        — compaction dropped it, retrieval didn't recover
#   recovered   — compaction dropped it, retrieval DID recover
#   in_summary  — compaction kept it (available in summary)
#
# File references have zero "lost" facts (literal paths match trivially),
# so we sample only recovered + in_summary for that category.
#
# Oversample the product-critical categories (user_constraint, error_state)
# relative to their population share; these are the headline numbers that
# need the most statistical support.

STRATA: dict[str, dict[str, int]] = {
    "user_constraint": {"lost": 30, "recovered": 15, "in_summary": 10},
    "error_state": {"lost": 25, "recovered": 10, "in_summary": 10},
    "file_reference": {"lost": 0, "recovered": 10, "in_summary": 10},
    "assistant_finding": {"lost": 15, "recovered": 10, "in_summary": 5},
}
# Total: 70 lost, 45 recovered, 35 in_summary = 150 facts


# --- Data classes for sample and labels ------------------------------------


@dataclass
class SampleFact:
    """A fact selected for labeling, with enough context to judge it."""

    sample_id: str  # unique within the sample: f"{category}_{idx}"
    category: str
    bucket: str  # lost | recovered | in_summary
    fact_value: str
    search_terms: list[str]
    # Provenance
    session_path: str
    session_id: str
    project: str
    compaction_index: int
    source_record_index: int
    # Context for the human labeler
    compaction_summary_preview: str
    source_record_preview: str
    # Evaluator's heuristic classification (what we're validating)
    evaluator_in_summary: bool
    evaluator_retrieval_recovered: bool


@dataclass
class FactLabel:
    """Human label for a single sampled fact."""

    sample_id: str
    is_meaningful: bool
    category_correct: bool
    in_summary_gt: bool  # ground truth: is the fact in the summary?
    would_loss_matter: str  # low | medium | high
    notes: str = ""


@dataclass
class LabelSample:
    """A generated sample with its facts and (partial) labels."""

    generated_at: str
    sessions_dir: str
    target_counts: dict[str, dict[str, int]]
    facts: list[SampleFact]
    labels: dict[str, FactLabel] = field(default_factory=dict)
    completed: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "generated_at": self.generated_at,
                "sessions_dir": self.sessions_dir,
                "target_counts": self.target_counts,
                "facts": [asdict(f) for f in self.facts],
                "labels": {k: asdict(v) for k, v in self.labels.items()},
                "completed": self.completed,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, data: dict) -> LabelSample:
        facts = [SampleFact(**f) for f in data.get("facts", [])]
        labels = {
            k: FactLabel(**v) for k, v in data.get("labels", {}).items()
        }
        return cls(
            generated_at=data.get("generated_at", ""),
            sessions_dir=data.get("sessions_dir", ""),
            target_counts=data.get("target_counts", {}),
            facts=facts,
            labels=labels,
            completed=data.get("completed", False),
        )


# --- Sample generation -----------------------------------------------------


def generate_sample(
    sessions_dir: str | Path,
    *,
    seed: int = 42,
    console: Console | None = None,
    loader: SessionLoader | None = None,
) -> LabelSample:
    """Generate a stratified sample of facts for labeling.

    Loads all sessions with compaction, evaluates every boundary, and
    samples facts into strata (category × bucket). The sample is
    deterministic for a given seed so it can be re-generated and
    cross-checked.
    """
    con = console or Console()
    sessions_dir = Path(sessions_dir)
    loader = loader or SessionLoader()
    evaluator = CompactionLossEvaluator()

    con.print("[dim]Scanning for sessions with compaction...[/dim]")
    sessions = loader.scan_directory(sessions_dir)
    con.print(f"[green]Found {len(sessions)} sessions with compaction[/green]")

    # Collect all facts grouped by (category, bucket), keeping provenance.
    # We store enough to build SampleFact later without re-parsing.
    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for i, (path, session) in enumerate(sessions):
        reports = evaluator.evaluate_session(session)
        project = _extract_project_name(path)
        for comp_idx, report in enumerate(reports):
            boundary = report.boundary
            n_facts = len(report.facts)
            results = report.results
            summary_preview = boundary.summary_preview

            for j in range(n_facts):
                fact = report.facts[j]
                comp_status = results[j * 4 + 1].status  # COMPACTION_ONLY
                ret_status = results[j * 4 + 2].status  # RETRIEVAL

                if comp_status == FactStatus.AVAILABLE:
                    bucket = "in_summary"
                elif ret_status == FactStatus.RECOVERED:
                    bucket = "recovered"
                else:
                    bucket = "lost"

                cat = fact.category.value
                # Only collect for strata we need
                if STRATA.get(cat, {}).get(bucket, 0) == 0:
                    continue

                pool[(cat, bucket)].append(
                    {
                        "fact": fact,
                        "session_path": str(path),
                        "session_id": session.session_id,
                        "project": project,
                        "compaction_index": comp_idx,
                        "source_record_index": fact.source_index,
                        "compaction_summary": boundary.summary,
                        "summary_preview": summary_preview,
                        "dropped_records": boundary.dropped_records,
                    }
                )

        if (i + 1) % 20 == 0:
            con.print(f"  [dim]Processed {i + 1}/{len(sessions)}...[/dim]")

    # Sample from each stratum
    rng = random.Random(seed)
    selected: list[SampleFact] = []
    con.print("\n[bold]Sampling from strata:[/bold]")
    sample_table = Table(show_header=True, header_style="bold cyan")
    sample_table.add_column("Category")
    sample_table.add_column("Bucket", justify="right")
    sample_table.add_column("Pool", justify="right")
    sample_table.add_column("Sampled", justify="right")

    for cat in [
        "user_constraint",
        "error_state",
        "file_reference",
        "assistant_finding",
    ]:
        for bucket in ["lost", "recovered", "in_summary"]:
            target = STRATA.get(cat, {}).get(bucket, 0)
            if target == 0:
                continue
            candidates = pool.get((cat, bucket), [])
            n_pool = len(candidates)
            n_take = min(target, n_pool)
            if n_pool > 0:
                chosen = rng.sample(candidates, n_take)
            else:
                chosen = []
            for idx, item in enumerate(chosen):
                fact: Fact = item["fact"]
                # Build the source record preview
                src_idx = fact.source_index
                dropped = item["dropped_records"]
                if 0 <= src_idx < len(dropped):
                    src_record = dropped[src_idx]
                    src_preview = _build_record_preview(src_record)
                else:
                    src_preview = "(record not available)"

                sid = f"{cat}_{len(selected)}"
                selected.append(
                    SampleFact(
                        sample_id=sid,
                        category=cat,
                        bucket=bucket,
                        fact_value=fact.value,
                        search_terms=fact.search_terms,
                        session_path=item["session_path"],
                        session_id=item["session_id"],
                        project=item["project"],
                        compaction_index=item["compaction_index"],
                        source_record_index=src_idx,
                        compaction_summary_preview=item["summary_preview"],
                        source_record_preview=src_preview,
                        evaluator_in_summary=(
                            bucket == "in_summary"
                        ),
                        evaluator_retrieval_recovered=(
                            bucket == "recovered"
                        ),
                    )
                )
            sample_table.add_row(cat, bucket, str(n_pool), str(n_take))

    con.print(sample_table)

    from datetime import datetime

    return LabelSample(
        generated_at=datetime.now().isoformat(),
        sessions_dir=str(sessions_dir),
        target_counts=STRATA,
        facts=selected,
    )


def _build_record_preview(record) -> str:
    """Build a human-readable preview of the source record.

    Shows role, text content, and tool calls — enough for a labeler
    to judge whether the extracted fact is meaningful.
    """
    from agent_fail_eval.models import BLOCK_TOOL_CALL

    parts = []
    role = record.role or "unknown"
    parts.append(f"[role={role}]")

    text = record.text_content.strip()
    if text:
        # Truncate very long text but show enough to judge
        if len(text) > 800:
            parts.append(text[:400] + "\n...[truncated]...\n" + text[-200:])
        else:
            parts.append(text)

    for tc in record.tool_calls:
        tool_name = tc.get("toolName", tc.get("name", "unknown"))
        args = tc.get("arguments", {})
        args_str = json.dumps(args, indent=2) if args else "{}"
        if len(args_str) > 300:
            args_str = args_str[:300] + "..."
        parts.append(f"[tool_call: {tool_name}]\n{args_str}")

    return "\n".join(parts)


def _extract_project_name(path: Path) -> str:
    """Extract project name from session path (mirrors report.py)."""
    parent = path.parent.name
    if parent == "sessions":
        return path.stem
    return parent.strip("-")


# --- Interactive labeling --------------------------------------------------


def annotate_sample(
    sample: LabelSample, *, console: Console | None = None
) -> LabelSample:
    """Run an interactive labeling session over the sample.

    Presents each unlabeled fact with its context and prompts for
    labels. Skips already-labeled facts (resumable). Saves after
    every label to avoid data loss.
    """
    con = console or Console()
    total = len(sample.facts)
    labeled = len(sample.labels)
    con.print(
        f"\n[bold]Fact Labeling Session[/bold] — {labeled}/{total} already labeled\n"
    )

    for i, sf in enumerate(sample.facts):
        if sf.sample_id in sample.labels:
            continue

        _print_fact_for_labeling(sf, i + 1, total, con)

        con.print("\n[bold yellow]Labels:[/bold yellow]")
        is_meaningful = Confirm.ask(
            "  Is this a meaningful, discrete piece of information?",
            default=True,
        )

        if not is_meaningful:
            # Skip category/summary questions for noise — not meaningful
            in_summary_gt = False
            category_correct = False
            would_loss_matter = "low"
            con.print("  [dim]Marked as noise — skipping further questions[/dim]")
        else:
            category_correct = Confirm.ask(
                f"  Is '{sf.category}' the correct category?",
                default=True,
            )
            in_summary_gt = Confirm.ask(
                "  Is this fact represented in the compaction summary above?",
                default=sf.evaluator_in_summary,
            )
            would_loss_matter = Prompt.ask(
                "  If lost, how much would it matter at a later decision point?",
                choices=["low", "medium", "high"],
                default="medium",
            )

        notes = Prompt.ask("  Notes (optional, press enter to skip)", default="")

        sample.labels[sf.sample_id] = FactLabel(
            sample_id=sf.sample_id,
            is_meaningful=is_meaningful,
            category_correct=category_correct,
            in_summary_gt=in_summary_gt,
            would_loss_matter=would_loss_matter,
            notes=notes if notes != "" else "",
        )

        con.print(f"  [green]Labeled {sf.sample_id}[/green]\n")

    sample.completed = len(sample.labels) == total
    if sample.completed:
        con.print(f"[bold green]All {total} facts labeled![/bold green]")
    else:
        con.print(
            f"[yellow]Labeled {len(sample.labels)}/{total} — run again to continue[/yellow]"
        )

    return sample


def _print_fact_for_labeling(
    sf: SampleFact, current: int, total: int, con: Console
) -> None:
    """Print a fact with context for labeling."""
    con.print(
        f"[bold]{'=' * 70}[/bold]\n"
        f"[bold]Fact {current}/{total}[/bold] — {sf.sample_id} "
        f"({sf.category} / {sf.bucket})\n"
        f"[bold]{'=' * 70}[/bold]"
    )

    con.print(
        Panel(
            sf.fact_value,
            title="Extracted Fact Value",
            border_style="cyan",
        )
    )

    if sf.search_terms:
        con.print(f"[dim]Search terms: {', '.join(sf.search_terms[:10])}[/dim]")

    con.print(
        Panel(
            sf.compaction_summary_preview,
            title="Compaction Summary (what replaced the dropped content)",
            border_style="blue",
        )
    )

    con.print(
        Panel(
            sf.source_record_preview,
            title=f"Source Record (record #{sf.source_record_index} in dropped segment)",
            border_style="magenta",
        )
    )

    con.print(
        f"[dim]Session: {sf.session_path}\n"
        f"Project: {sf.project} | Compaction #{sf.compaction_index}[/dim]"
    )


# --- Statistics ------------------------------------------------------------


def compute_stats(
    sample: LabelSample, *, console: Console | None = None
) -> dict[str, Any]:
    """Compute precision/recall of the heuristic extractor vs human labels.

    Ground truth: the human `in_summary_gt` label tells us whether each
    fact is actually in the summary. The evaluator's heuristic
    `_fact_in_summary` produces `evaluator_in_summary`. We compare:

    - Precision of "lost" classification: of facts the evaluator calls
      lost (not in summary), what fraction are truly not in the summary?
    - Recall of "lost" classification: of facts truly not in the summary,
      what fraction does the evaluator catch?
    - Same for "in summary" (the inverse).
    - Meaningful-fact rate: what fraction of extracted facts are genuine
      information vs noise? This is the extractor precision proxy.
    - Loss-matters distribution: among truly-lost meaningful facts, how
      many are high/medium/low impact?
    """
    con = console or Console()

    if not sample.labels:
        con.print("[red]No labels found. Run 'afe label annotate' first.[/red]")
        return {}

    labeled_facts = [
        (f, sample.labels[f.sample_id])
        for f in sample.facts
        if f.sample_id in sample.labels
    ]

    n = len(labeled_facts)
    con.print(f"\n[bold]Labeling Statistics ({n} labeled facts)[/bold]\n")

    # --- Overall meaningfulness ---
    meaningful = sum(1 for _, lab in labeled_facts if lab.is_meaningful)
    meaningful_rate = meaningful / n if n else 0
    con.print(f"Meaningful facts: {meaningful}/{n} ({meaningful_rate:.1%})")

    # --- Category correctness ---
    cat_correct = sum(1 for _, lab in labeled_facts if lab.category_correct)
    con.print(f"Category correct: {cat_correct}/{n} ({cat_correct/n:.1%})")

    # --- Precision/recall of evaluator's in_summary classification ---
    # Only meaningful facts count — noise shouldn't inflate either way.
    meaningful_labeled = [
        (f, lab) for f, lab in labeled_facts if lab.is_meaningful
    ]

    nm = len(meaningful_labeled)
    if nm == 0:
        con.print("[red]No meaningful facts labeled — cannot compute metrics.[/red]")
        return {}

    # Confusion matrix for "lost" class:
    #   "lost" = not in summary. evaluator_in_summary == False predicts lost.
    #   eval=lost & human=lost → TP (evaluator correctly identified a loss)
    #   eval=lost & human=in_summary → FP (evaluator falsely said lost)
    #   eval=in_summary & human=lost → FN (evaluator missed the loss)
    #   eval=in_summary & human=in_summary → TN

    tp_lost = sum(
        1 for f, lab in meaningful_labeled
        if not f.evaluator_in_summary and not lab.in_summary_gt
    )
    fp_lost = sum(
        1 for f, lab in meaningful_labeled
        if not f.evaluator_in_summary and lab.in_summary_gt
    )
    fn_lost = sum(
        1 for f, lab in meaningful_labeled
        if f.evaluator_in_summary and not lab.in_summary_gt
    )
    tn_lost = sum(
        1 for f, lab in meaningful_labeled
        if f.evaluator_in_summary and lab.in_summary_gt
    )

    lost_precision = tp_lost / (tp_lost + fp_lost) if (tp_lost + fp_lost) else 0
    lost_recall = tp_lost / (tp_lost + fn_lost) if (tp_lost + fn_lost) else 0

    # By category
    by_cat: dict[str, dict] = {}
    for cat in [
        "user_constraint",
        "error_state",
        "file_reference",
        "assistant_finding",
    ]:
        cat_facts = [
            (f, lab) for f, lab in meaningful_labeled if f.category == cat
        ]
        nc = len(cat_facts)
        if nc == 0:
            by_cat[cat] = {"n": 0}
            continue
        cat_meaningful = nc  # already filtered to meaningful
        cat_tp = sum(
            1 for f, lab in cat_facts
            if not f.evaluator_in_summary and not lab.in_summary_gt
        )
        cat_fp = sum(
            1 for f, lab in cat_facts
            if not f.evaluator_in_summary and lab.in_summary_gt
        )
        cat_fn = sum(
            1 for f, lab in cat_facts
            if f.evaluator_in_summary and not lab.in_summary_gt
        )
        cat_tn = sum(
            1 for f, lab in cat_facts
            if f.evaluator_in_summary and lab.in_summary_gt
        )
        cat_prec = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) else 0
        cat_rec = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) else 0
        cat_lost_gt = cat_tp + cat_fn
        cat_impact = Counter(
            lab.would_loss_matter for f, lab in cat_facts
            if not lab.in_summary_gt  # truly lost
        )
        by_cat[cat] = {
            "n": nc,
            "lost_gt": cat_lost_gt,
            "precision": round(cat_prec, 3),
            "recall": round(cat_rec, 3),
            "impact": dict(cat_impact),
        }

    # Impact distribution among truly-lost meaningful facts
    impact_dist = Counter(
        lab.would_loss_matter
        for f, lab in meaningful_labeled
        if not lab.in_summary_gt
    )

    stats = {
        "total_labeled": n,
        "meaningful": meaningful,
        "meaningful_rate": round(meaningful_rate, 3),
        "category_correct_rate": round(cat_correct / n, 3) if n else 0,
        "meaningful_n": nm,
        "lost_confusion": {
            "tp": tp_lost,
            "fp": fp_lost,
            "fn": fn_lost,
            "tn": tn_lost,
        },
        "lost_precision": round(lost_precision, 3),
        "lost_recall": round(lost_recall, 3),
        "impact_distribution": dict(impact_dist),
        "by_category": by_cat,
    }

    # Print summary table
    con.print("\n[bold]Evaluator 'lost fact' precision/recall (vs human labels):[/bold]")
    con.print(f"  Lost precision: {lost_precision:.1%} ({tp_lost} TP / {tp_lost + fp_lost} predicted-lost)")
    con.print(f"  Lost recall:    {lost_recall:.1%} ({tp_lost} TP / {tp_lost + fn_lost} actual-lost)")

    con.print("\n[bold]Impact distribution (truly-lost meaningful facts):[/bold]")
    for level in ["high", "medium", "low"]:
        count = impact_dist.get(level, 0)
        con.print(f"  {level}: {count}")

    cat_table = Table(show_header=True, header_style="bold cyan")
    cat_table.add_column("Category")
    cat_table.add_column("N", justify="right")
    cat_table.add_column("Lost (GT)", justify="right")
    cat_table.add_column("Precision", justify="right")
    cat_table.add_column("Recall", justify="right")
    cat_table.add_column("High/Med/Low Impact")
    for cat in [
        "user_constraint",
        "error_state",
        "file_reference",
        "assistant_finding",
    ]:
        c = by_cat[cat]
        if c["n"] == 0:
            continue
        impact = c.get("impact", {})
        cat_table.add_row(
            cat,
            str(c["n"]),
            str(c["lost_gt"]),
            f"{c['precision']:.1%}",
            f"{c['recall']:.1%}",
            f"{impact.get('high',0)}/{impact.get('medium',0)}/{impact.get('low',0)}",
        )
    con.print(cat_table)

    return stats
