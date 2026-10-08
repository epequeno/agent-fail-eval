"""Aggregate report generator.

Runs the compaction-loss evaluator across many sessions and produces
aggregate statistics: per-strategy recovery rates, per-category breakdowns,
loss-rate distribution, and worst-case boundaries.

This is the "reproducible failure report" artifact from the business plan's
First Step #2 — a single artifact that quantifies compaction loss across
real coding-agent sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_fail_eval.evaluator import (
    CompactionLossEvaluator,
    CompactionLossReport,
    FactCategory,
    FactStatus,
    Strategy,
)
from agent_fail_eval.models import CompactionBoundary
from agent_fail_eval.parser import SessionLoader

if TYPE_CHECKING:
    from agent_fail_eval.redactor import SessionRedactor



@dataclass
class BoundaryResult:
    """Results for a single compaction boundary within a session."""

    session_path: str
    session_id: str
    project: str
    compaction_index: int
    tokens_before: int
    records_dropped: int
    total_facts: int
    # Per-strategy counts
    compaction_available: int
    compaction_lost: int
    retrieval_available: int
    retrieval_lost: int
    retrieval_recovered: int
    repl_available: int = 0
    repl_lost: int = 0
    repl_recovered: int = 0
    # Per-category loss counts (compaction_only strategy)
    category_totals: dict[str, int] = field(default_factory=dict)
    category_lost: dict[str, int] = field(default_factory=dict)
    category_recovered: dict[str, int] = field(default_factory=dict)

    @property
    def compaction_loss_rate(self) -> float:
        """Fraction of facts lost by compaction-only."""
        if self.total_facts == 0:
            return 0.0
        return self.compaction_lost / self.total_facts

    @property
    def retrieval_recovery_rate(self) -> float:
        """Fraction of lost facts that retrieval recovers."""
        if self.compaction_lost == 0:
            return 0.0
        return self.retrieval_recovered / self.compaction_lost

    @property
    def retrieval_loss_rate(self) -> float:
        """Fraction of facts still lost after retrieval."""
        if self.total_facts == 0:
            return 0.0
        return self.retrieval_lost / self.total_facts

    @property
    def repl_recovery_rate(self) -> float:
        """Fraction of lost facts that REPL recovers."""
        if self.compaction_lost == 0:
            return 0.0
        return self.repl_recovered / self.compaction_lost

    @property
    def repl_loss_rate(self) -> float:
        """Fraction of facts still lost after REPL."""
        if self.total_facts == 0:
            return 0.0
        return self.repl_lost / self.total_facts


@dataclass
class AggregateReport:
    """Aggregate statistics across all evaluated sessions.

    This is the serializable report artifact. It can be exported to JSON
    for reproducibility and sharing.
    """

    generated_at: str
    sessions_dir: str
    total_sessions: int
    total_boundaries: int
    total_facts: int
    boundaries: list[BoundaryResult]

    @property
    def total_compaction_lost(self) -> int:
        return sum(b.compaction_lost for b in self.boundaries)

    @property
    def total_retrieval_recovered(self) -> int:
        return sum(b.retrieval_recovered for b in self.boundaries)

    @property
    def total_retrieval_lost(self) -> int:
        return sum(b.retrieval_lost for b in self.boundaries)

    @property
    def total_repl_recovered(self) -> int:
        return sum(b.repl_recovered for b in self.boundaries)

    @property
    def total_repl_lost(self) -> int:
        return sum(b.repl_lost for b in self.boundaries)

    @property
    def overall_compaction_loss_rate(self) -> float:
        if self.total_facts == 0:
            return 0.0
        return self.total_compaction_lost / self.total_facts

    @property
    def overall_retrieval_recovery_rate(self) -> float:
        if self.total_compaction_lost == 0:
            return 0.0
        return self.total_retrieval_recovered / self.total_compaction_lost

    @property
    def overall_repl_recovery_rate(self) -> float:
        if self.total_compaction_lost == 0:
            return 0.0
        return self.total_repl_recovered / self.total_compaction_lost

    @property
    def overall_repl_loss_rate(self) -> float:
        if self.total_facts == 0:
            return 0.0
        return self.total_repl_lost / self.total_facts

    @property
    def overall_retrieval_loss_rate(self) -> float:
        if self.total_facts == 0:
            return 0.0
        return self.total_retrieval_lost / self.total_facts

    def category_stats(self) -> dict[str, dict[str, int | float]]:
        """Per-category aggregate stats across all boundaries."""
        cats: dict[str, dict[str, int]] = {}
        for cat in FactCategory:
            cats[cat.value] = {
                "total": 0,
                "compaction_lost": 0,
                "retrieval_recovered": 0,
                "retrieval_lost": 0,
            }

        for b in self.boundaries:
            for cat_str, stats in cats.items():
                stats["total"] += b.category_totals.get(cat_str, 0)
                stats["compaction_lost"] += b.category_lost.get(cat_str, 0)
                stats["retrieval_recovered"] += b.category_recovered.get(cat_str, 0)
                stats["retrieval_lost"] += (
                    b.category_lost.get(cat_str, 0)
                    - b.category_recovered.get(cat_str, 0)
                )

        result: dict[str, dict[str, int | float]] = {}
        for cat_str, stats in cats.items():
            total = stats["total"]
            lost = stats["compaction_lost"]
            recovered = stats["retrieval_recovered"]
            result[cat_str] = {
                **stats,
                "compaction_loss_rate": lost / total if total > 0 else 0.0,
                "retrieval_recovery_rate": recovered / lost if lost > 0 else 0.0,
                "retrieval_loss_rate": (lost - recovered) / total if total > 0 else 0.0,
            }
        return result

    def loss_rate_distribution(self) -> dict[str, float]:
        """Distribution stats for compaction loss rate across boundaries."""
        rates = [b.compaction_loss_rate for b in self.boundaries]
        if not rates:
            return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0, "stdev": 0}

        rates_sorted = sorted(rates)
        n = len(rates_sorted)

        def percentile(p: float) -> float:
            idx = int(n * p)
            if idx >= n:
                idx = n - 1
            return rates_sorted[idx]

        return {
            "min": rates_sorted[0],
            "p25": percentile(0.25),
            "median": median(rates),
            "p75": percentile(0.75),
            "max": rates_sorted[-1],
            "mean": mean(rates),
            "stdev": stdev(rates) if len(rates) > 1 else 0.0,
        }

    def worst_boundaries(self, n: int = 10) -> list[BoundaryResult]:
        """Boundaries with the highest compaction loss rate."""
        return sorted(
            self.boundaries, key=lambda b: b.compaction_loss_rate, reverse=True
        )[:n]

    def to_json(self) -> str:
        """Serialize to JSON for reproducibility."""
        return json.dumps(
            {
                "generated_at": self.generated_at,
                "sessions_dir": self.sessions_dir,
                "total_sessions": self.total_sessions,
                "total_boundaries": self.total_boundaries,
                "total_facts": self.total_facts,
                "overall": {
                    "compaction_lost": self.total_compaction_lost,
                    "retrieval_recovered": self.total_retrieval_recovered,
                    "retrieval_still_lost": self.total_retrieval_lost,
                    "repl_recovered": self.total_repl_recovered,
                    "repl_still_lost": self.total_repl_lost,
                    "compaction_loss_rate": self.overall_compaction_loss_rate,
                    "retrieval_recovery_rate": self.overall_retrieval_recovery_rate,
                    "retrieval_loss_rate": self.overall_retrieval_loss_rate,
                    "repl_recovery_rate": self.overall_repl_recovery_rate,
                    "repl_loss_rate": self.overall_repl_loss_rate,
                },
                "by_category": self.category_stats(),
                "loss_rate_distribution": self.loss_rate_distribution(),
                "boundaries": [
                    {
                        "session_id": b.session_id,
                        "project": b.project,
                        "compaction_index": b.compaction_index,
                        "tokens_before": b.tokens_before,
                        "records_dropped": b.records_dropped,
                        "total_facts": b.total_facts,
                        "compaction_lost": b.compaction_lost,
                        "retrieval_recovered": b.retrieval_recovered,
                        "retrieval_lost": b.retrieval_lost,
                        "repl_recovered": b.repl_recovered,
                        "repl_lost": b.repl_lost,
                        "compaction_loss_rate": b.compaction_loss_rate,
                        "retrieval_recovery_rate": b.retrieval_recovery_rate,
                        "repl_recovery_rate": b.repl_recovery_rate,
                    }
                    for b in self.boundaries
                ],
            },
            indent=2,
        )


class ReportGenerator:
    """Runs the evaluator across many sessions and produces an AggregateReport."""

    def __init__(self, loader: SessionLoader | None = None) -> None:
        self.loader = loader or SessionLoader()
        self.evaluator = CompactionLossEvaluator()

    def generate(
        self,
        sessions_dir: str | Path,
        *,
        redactor: SessionRedactor | None = None,
        console: Console | None = None,
    ) -> AggregateReport:
        """Generate an aggregate report from all sessions with compaction.

        Args:
            sessions_dir: Path to the pi agent sessions directory.
            redactor: Optional SessionRedactor to sanitize all session data
                      before evaluation. Facts, reports, and output JSON
                      will contain only redacted content.
            console: Optional rich console for progress output.

        Returns:
            AggregateReport with per-boundary results and aggregate stats.
        """
        sessions_dir = Path(sessions_dir)
        con = console or Console()

        # Scan for sessions with compaction
        con.print("[dim]Scanning for sessions with compaction...[/dim]")
        sessions = self.loader.scan_directory(sessions_dir, redactor=redactor)
        con.print(f"[green]Found {len(sessions)} sessions with compaction events[/green]")

        boundary_results: list[BoundaryResult] = []

        for i, (path, session) in enumerate(sessions):
            raw_project = _extract_project_name(path)
            project = redactor.redact_project_name(str(path)) if redactor else raw_project
            con.print(
                f"  [{i + 1}/{len(sessions)}] {project} "
                f"({len(session.compaction_boundaries)} compactions)...",
                end="",
            )

            reports = self.evaluator.evaluate_session(session)

            for comp_idx, report in enumerate(reports):
                raw_session_path = str(path)
                clean_session_path = (
                    redactor.redact_session_path(raw_session_path) if redactor
                    else raw_session_path
                )
                br = self._report_to_boundary_result(
                    report, clean_session_path, session.session_id, project, comp_idx
                )
                boundary_results.append(br)

            con.print(" [green]done[/green]")

        return AggregateReport(
            generated_at=datetime.now().isoformat(),
            sessions_dir=str(sessions_dir),
            total_sessions=len(sessions),
            total_boundaries=len(boundary_results),
            total_facts=sum(b.total_facts for b in boundary_results),
            boundaries=boundary_results,
        )

    def _report_to_boundary_result(
        self,
        report: CompactionLossReport,
        session_path: str,
        session_id: str,
        project: str,
        compaction_index: int,
    ) -> BoundaryResult:
        """Convert a CompactionLossReport into a serializable BoundaryResult."""
        boundary = report.boundary
        stats = report.summary_stats()

        comp_stats = stats[Strategy.COMPACTION_ONLY.value]
        ret_stats = stats[Strategy.RETRIEVAL.value]
        repl_stats = stats[Strategy.REPL.value]

        # Per-category breakdown
        cat_totals: dict[str, int] = {}
        cat_lost: dict[str, int] = {}
        cat_recovered: dict[str, int] = {}

        for cat in FactCategory:
            facts_in_cat = [f for f in report.facts if f.category == cat]
            cat_totals[cat.value] = len(facts_in_cat)

            comp_results = [
                r for r in report.results
                if r.fact.category == cat and r.strategy == Strategy.COMPACTION_ONLY
            ]
            ret_results = [
                r for r in report.results
                if r.fact.category == cat and r.strategy == Strategy.RETRIEVAL
            ]

            cat_lost[cat.value] = sum(
                1 for r in comp_results if r.status == FactStatus.LOST
            )
            cat_recovered[cat.value] = sum(
                1 for r in ret_results if r.status == FactStatus.RECOVERED
            )

        return BoundaryResult(
            session_path=session_path,
            session_id=session_id,
            project=project,
            compaction_index=compaction_index,
            tokens_before=boundary.tokens_before,
            records_dropped=boundary.newly_dropped_count,
            total_facts=len(report.facts),
            compaction_available=comp_stats["available"],
            compaction_lost=comp_stats["lost"],
            retrieval_available=ret_stats["available"],
            retrieval_recovered=ret_stats["recovered"],
            retrieval_lost=ret_stats["lost"],
            repl_available=repl_stats["available"],
            repl_lost=repl_stats["lost"],
            repl_recovered=repl_stats["recovered"],
            category_totals=cat_totals,
            category_lost=cat_lost,
            category_recovered=cat_recovered,
        )


def print_report(report: AggregateReport, console: Console | None = None) -> None:
    """Print a human-readable aggregate report."""
    con = console or Console()

    # Header
    con.print(
        Panel(
            f"[bold]Generated:[/bold] {report.generated_at}\n"
            f"[bold]Sessions dir:[/bold] {report.sessions_dir}\n"
            f"[bold]Sessions evaluated:[/bold] {report.total_sessions}\n"
            f"[bold]Compaction boundaries:[/bold] {report.total_boundaries}\n"
            f"[bold]Total facts extracted:[/bold] {report.total_facts:,}",
            title="Compaction-Loss Aggregate Report",
            border_style="red",
        )
    )

    # Overall stats
    con.print("\n[bold]Overall Strategy Comparison[/bold]\n")

    overall_table = Table(show_header=True, header_style="bold cyan")
    overall_table.add_column("Strategy", style="bold", width=22)
    overall_table.add_column("Facts Available", justify="right", width=16)
    overall_table.add_column("Facts Lost", justify="right", width=12)
    overall_table.add_column("Recovery Rate", justify="right", width=14)

    total = report.total_facts
    comp_avail = total - report.total_compaction_lost
    ret_avail = total - report.total_retrieval_lost
    repl_avail = total - report.total_repl_lost

    overall_table.add_row(
        "full_history (oracle)",
        f"{total:,}",
        "0",
        "100%",
        style="dim",
    )
    overall_table.add_row(
        "compaction_only",
        f"{comp_avail:,}",
        f"{report.total_compaction_lost:,}",
        f"{comp_avail / total:.1%}" if total > 0 else "—",
        style="red",
    )
    overall_table.add_row(
        "retrieval",
        f"{ret_avail:,}",
        f"{report.total_retrieval_lost:,}",
        f"{ret_avail / total:.1%}" if total > 0 else "—",
        style="yellow",
    )
    overall_table.add_row(
        "repl",
        f"{repl_avail:,}",
        f"{report.total_repl_lost:,}",
        f"{repl_avail / total:.1%}" if total > 0 else "—",
        style="green",
    )

    con.print(overall_table)

    # Recovery delta
    ret_delta = report.total_retrieval_recovered
    ret_delta_pct = report.overall_retrieval_recovery_rate
    repl_delta = report.total_repl_recovered
    repl_delta_pct = report.overall_repl_recovery_rate
    con.print(
        f"\n[bold yellow]Retrieval recovers {ret_delta:,} of {report.total_compaction_lost:,} "
        f"lost facts ({ret_delta_pct:.1%} recovery rate)[/bold yellow]"
    )
    con.print(
        f"[bold green]REPL recovers {repl_delta:,} of {report.total_compaction_lost:,} "
        f"lost facts ({repl_delta_pct:.1%} recovery rate)[/bold green]"
    )
    con.print(
        f"[bold red]{report.total_repl_lost:,} facts remain lost even with "
        f"REPL ({report.overall_repl_loss_rate:.1%} of total)[/bold red]"
    )

    # Per-category breakdown
    con.print("\n[bold]Per-Category Breakdown[/bold]\n")

    cat_table = Table(show_header=True, header_style="bold magenta")
    cat_table.add_column("Category", width=22)
    cat_table.add_column("Total Facts", justify="right", width=12)
    cat_table.add_column("Compaction Lost", justify="right", width=16)
    cat_table.add_column("Loss Rate", justify="right", width=10)
    cat_table.add_column("Retrieval Recovers", justify="right", width=18)
    cat_table.add_column("Recovery Rate", justify="right", width=14)
    cat_table.add_column("Still Lost", justify="right", width=10)

    cat_stats = report.category_stats()
    for cat in FactCategory:
        s = cat_stats[cat.value]
        total_cat = s["total"]
        if total_cat == 0:
            continue
        lost_cat = s["compaction_lost"]
        recovered_cat = s["retrieval_recovered"]
        still_lost = lost_cat - recovered_cat

        cat_table.add_row(
            cat.value,
            f"{total_cat:,}",
            f"{lost_cat:,}",
            f"{s['compaction_loss_rate']:.1%}",
            f"{recovered_cat:,}",
            f"{s['retrieval_recovery_rate']:.1%}",
            f"{still_lost:,}",
        )

    con.print(cat_table)

    # Loss rate distribution
    con.print("\n[bold]Compaction Loss Rate Distribution[/bold]\n")

    dist = report.loss_rate_distribution()
    dist_table = Table(show_header=True, header_style="bold cyan")
    dist_table.add_column("Statistic", width=12)
    dist_table.add_column("Loss Rate", justify="right", width=12)

    for label in ["min", "p25", "median", "mean", "p75", "max", "stdev"]:
        dist_table.add_row(label, f"{dist[label]:.1%}")

    con.print(dist_table)

    # Worst-case boundaries
    con.print("\n[bold]Worst-Case Compaction Boundaries (highest loss rate)[/bold]\n")

    worst = report.worst_boundaries(15)
    worst_table = Table(show_header=True, header_style="bold red")
    worst_table.add_column("#", style="dim", width=4)
    worst_table.add_column("Project", style="green", width=20)
    worst_table.add_column("Comp", justify="right", width=5)
    worst_table.add_column("Tokens", justify="right", width=12)
    worst_table.add_column("Facts", justify="right", width=8)
    worst_table.add_column("Lost", justify="right", width=8)
    worst_table.add_column("Loss Rate", justify="right", width=10)
    worst_table.add_column("Retrieval", justify="right", width=10)

    for i, b in enumerate(worst):
        worst_table.add_row(
            str(i),
            b.project[:20],
            str(b.compaction_index),
            f"{b.tokens_before:,}",
            str(b.total_facts),
            str(b.compaction_lost),
            f"{b.compaction_loss_rate:.1%}",
            f"{b.retrieval_recovery_rate:.1%}",
        )

    con.print(worst_table)


def _extract_project_name(path: Path) -> str:
    """Extract project name from session path."""
    parent = path.parent.name
    parts = parent.strip("-").split("-")
    if "code" in parts:
        idx = parts.index("code")
        return "-".join(parts[idx + 1 :]) or parent
    return parent.strip("-")
