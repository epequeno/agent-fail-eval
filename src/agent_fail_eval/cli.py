"""CLI for agent-fail-eval.

Commands:
    parse <session.jsonl>   Parse a single session and print a summary report.
    scan <sessions-dir>     Scan all sessions for compaction events.
    eval <session.jsonl>    Evaluate compaction loss and print a failure report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_fail_eval.models import CompactionBoundary, ParsedSession, SessionRecord
from agent_fail_eval.parser import SessionLoader
from agent_fail_eval.evaluator import CompactionLossEvaluator, FactCategory, FactStatus, Strategy
from agent_fail_eval.report import ReportGenerator, print_report
from agent_fail_eval.label import (
    LabelSample,
    annotate_sample,
    compute_stats,
    generate_sample,
)
from agent_fail_eval.redactor import SessionRedactor

console = Console()


def _get_loader(harness: str):
    """Return the session loader for the given harness ('pi' or 'claude')."""
    if harness == "claude":
        from agent_fail_eval.claude_parser import ClaudeCodeLoader
        return ClaudeCodeLoader()
    return SessionLoader()


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse a single session file and print a summary."""
    loader = _get_loader(getattr(args, 'harness', 'pi'))


    try:
        session = loader.load(args.session_file)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1

    _print_session_summary(session)

    if args.compaction is not None:
        if args.compaction >= len(session.compaction_boundaries):
            console.print(
                f"[red]Error:[/red] Compaction index {args.compaction} out of range "
                f"(0–{len(session.compaction_boundaries) - 1})"
            )
            return 1
        _print_compaction_detail(session.compaction_boundaries[args.compaction])

    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a directory for sessions with compaction events."""
    loader = _get_loader(getattr(args, 'harness', 'pi'))
    sessions = loader.scan_directory(args.sessions_dir)

    if not sessions:
        console.print("[yellow]No sessions with compaction events found.[/yellow]")
        return 0

    console.print(
        f"[bold green]Found {len(sessions)} sessions with compaction events[/bold green]\n"
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Session", style="cyan", max_width=50)
    table.add_column("Project", style="green", max_width=25)
    table.add_column("Msgs", justify="right", width=6)
    table.add_column("Compactions", justify="right", width=10)
    table.add_column("Total Dropped", justify="right", width=13)
    table.add_column("Max Tokens", justify="right", width=12)

    for i, (path, session) in enumerate(sessions):
        project = _extract_project_name(path)
        max_tokens = max(
            (b.tokens_before for b in session.compaction_boundaries), default=0
        )
        total_dropped = sum(b.newly_dropped_count for b in session.compaction_boundaries)

        table.add_row(
            str(i),
            path.stem[:50],
            project[:25],
            str(session.total_messages),
            str(len(session.compaction_boundaries)),
            str(total_dropped),
            f"{max_tokens:,}",
        )

    console.print(table)
    console.print(
        f"\n[dim]Total: {len(sessions)} sessions, "
        f"{sum(len(s.compaction_boundaries) for _, s in sessions)} compaction events[/dim]"
    )

    return 0


def _print_session_summary(session: ParsedSession) -> None:
    """Print a summary of the parsed session."""
    # Header
    console.print(
        Panel(
            f"[bold]Session:[/bold] {session.session_id}\n"
            f"[bold]CWD:[/bold] {session.cwd}\n"
            f"[bold]Started:[/bold] {session.started_at.isoformat()}\n"
            f"[bold]Records:[/bold] {session.total_records}  "
            f"[bold]Messages:[/bold] {session.total_messages}  "
            f"[bold]Compactions:[/bold] {len(session.compaction_boundaries)}",
            title="Session Summary",
            border_style="cyan",
        )
    )

    if session.models_used:
        console.print(f"[dim]Models:[/dim] {' → '.join(session.models_used)}")
    if session.thinking_levels:
        console.print(f"[dim]Thinking levels:[/dim] {' → '.join(session.thinking_levels)}")

    # Compaction boundaries
    if not session.compaction_boundaries:
        console.print("\n[yellow]No compaction events in this session.[/yellow]")
        return

    console.print(f"\n[bold]Compaction Boundaries ({len(session.compaction_boundaries)})[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Timestamp", width=26)
    table.add_column("Tokens Before", justify="right", width=14)
    table.add_column("Dropped Recs", justify="right", width=12)
    table.add_column("Dropped Msgs", justify="right", width=12)
    table.add_column("Tool Calls", justify="right", width=10)
    table.add_column("From Hook", width=9)

    for i, b in enumerate(session.compaction_boundaries):
        table.add_row(
            str(i),
            b.timestamp[:26],
            f"{b.tokens_before:,}",
            str(b.newly_dropped_count),
            str(len(b.dropped_messages)),
            str(len(b.dropped_tool_calls)),
            str(b.from_hook) if b.from_hook is not None else "-",
        )

    console.print(table)

    # Dropped content summary for each boundary
    for i, b in enumerate(session.compaction_boundaries):
        _print_dropped_summary(i, b)


def _print_dropped_summary(index: int, boundary: CompactionBoundary) -> None:
    """Print a summary of what was dropped at a compaction boundary."""
    console.print(
        f"\n[bold magenta]Compaction {index} — Dropped Content[/bold magenta]"
    )
    console.print(f"[dim]Tokens before:[/dim] {boundary.tokens_before:,}")
    console.print(f"[dim]Records dropped:[/dim] {boundary.newly_dropped_count}")
    console.print(
        f"[dim]Messages dropped:[/dim] {len(boundary.dropped_messages)} "
        f"({len(boundary.dropped_tool_calls)} tool calls)"
    )

    if boundary.read_files:
        console.print(f"[dim]Read files:[/dim] {', '.join(boundary.read_files[:5])}")
    if boundary.modified_files:
        console.print(
            f"[dim]Modified files:[/dim] {', '.join(boundary.modified_files[:5])}"
        )

    # Summary preview
    console.print("\n[bold]Summary that replaced dropped content:[/bold]")
    console.print(
        Panel(
            boundary.summary_preview,
            border_style="dim",
            expand=False,
        )
    )

    # Tool call types in dropped segment
    tool_names = [tc.get("name", "?") for tc in boundary.dropped_tool_calls]
    if tool_names:
        from collections import Counter

        counts = Counter(tool_names)
        tool_str = ", ".join(f"{name} ({n})" for name, n in counts.most_common(10))
        console.print(f"\n[dim]Dropped tool calls by type:[/dim] {tool_str}")

    # First and last dropped messages for context
    msgs = boundary.dropped_messages
    if msgs:
        console.print("\n[dim]First dropped message:[/dim]")
        first = msgs[0]
        _print_message_brief(first)

        console.print("\n[dim]Last dropped message:[/dim]")
        last = msgs[-1]
        _print_message_brief(last)


def _print_compaction_detail(boundary: CompactionBoundary) -> None:
    """Print full details of a specific compaction boundary."""
    console.print(
        Panel(
            f"[bold]Compaction ID:[/bold] {boundary.compaction_id}\n"
            f"[bold]Timestamp:[/bold] {boundary.timestamp}\n"
            f"[bold]Tokens before:[/bold] {boundary.tokens_before:,}\n"
            f"[bold]First kept entry:[/bold] {boundary.first_kept_entry_id}\n"
            f"[bold]File position:[/bold] {boundary.file_position}\n"
            f"[bold]First kept position:[/bold] {boundary.first_kept_position}\n"
            f"[bold]From hook:[/bold] {boundary.from_hook}",
            title=f"Compaction Detail",
            border_style="magenta",
        )
    )

    console.print("\n[bold]Full Summary:[/bold]\n")
    console.print(boundary.summary)

    console.print(f"\n\n[bold]Dropped Records ({boundary.newly_dropped_count}):[/bold]\n")

    for i, rec in enumerate(boundary.dropped_records):
        if rec.is_message:
            _print_message_brief(rec, index=i)
        elif rec.is_compaction:
            console.print(
                f"  [{i}] [dim](previous compaction summary)[/dim]"
            )


def _print_message_brief(record: SessionRecord, index: int | None = None) -> None:
    """Print a brief summary of a message record."""
    prefix = f"  [{index}] " if index is not None else "  "

    role = record.role or "?"
    role_style = {
        "user": "bold blue",
        "assistant": "bold green",
        "toolResult": "bold yellow",
    }.get(role, "dim")

    # Build content summary
    parts: list[str] = []

    for block in record.content_blocks:
        bt = block.get("type", "?")
        if bt == "text":
            text = block.get("text", "")[:120]
            if text:
                parts.append(f"text: {text}")
        elif bt == "toolCall":
            name = block.get("name", "?")
            args = block.get("arguments", {})
            # Show first arg or command
            if "command" in args:
                parts.append(f"tool({name}): {args['command'][:100]}")
            elif args:
                first_key = next(iter(args), "")
                parts.append(f"tool({name}): {first_key}={str(args[first_key])[:80]}")
            else:
                parts.append(f"tool({name})")
        elif bt == "thinking":
            thinking = block.get("thinking", "")[:100]
            if thinking:
                parts.append(f"thinking: {thinking}")
        elif bt == "toolResult":
            content = block.get("content", [])
            if isinstance(content, list):
                text = " ".join(
                    b.get("text", "")[:80]
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)[:80]
            parts.append(f"result: {text[:100]}")

    content_str = " | ".join(parts) if parts else "(empty)"
    console.print(
        Text.assemble(
            (prefix, "dim"),
            (f"{role:>11s} ", role_style),
            (content_str, "white"),
        )
    )

def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate compaction loss and print a failure report."""
    loader = SessionLoader()

    try:
        session = loader.load(args.session_file)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1

    if not session.compaction_boundaries:
        console.print("[yellow]No compaction events in this session.[/yellow]")
        return 0

    evaluator = CompactionLossEvaluator()
    reports = evaluator.evaluate_session(session)

    if args.compaction is not None:
        if args.compaction >= len(reports):
            console.print(
                f"[red]Error:[/red] Compaction index {args.compaction} out of range "
                f"(0–{len(reports) - 1})"
            )
            return 1
        reports = [reports[args.compaction]]

    for i, report in enumerate(reports):
        _print_eval_report(i, report, args.verbose)

    return 0


def _print_eval_report(
    index: int, report: "CompactionLossReport", verbose: bool
) -> None:
    """Print a compaction-loss evaluation report."""
    boundary = report.boundary
    stats = report.summary_stats()

    # Header
    console.print(
        Panel(
            f"[bold]Compaction {index}[/bold]\n"
            f"[bold]Tokens before:[/bold] {boundary.tokens_before:,}\n"
            f"[bold]Records dropped:[/bold] {boundary.newly_dropped_count}\n"
            f"[bold]Facts extracted:[/bold] {len(report.facts)}\n"
            f"[bold]Timestamp:[/bold] {boundary.timestamp}",
            title="Compaction-Loss Evaluation",
            border_style="red",
        )
    )

    # Strategy comparison table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Strategy", style="bold", width=20)
    table.add_column("Available", justify="right", width=10)
    table.add_column("Lost", justify="right", width=8)
    table.add_column("Recovered", justify="right", width=10)
    table.add_column("Recovery Rate", justify="right", width=14)

    for strategy in Strategy:
        s = stats[strategy.value]
        total = s["total"]
        if total == 0:
            continue
        recovery = s["available"] + s["recovered"]
        rate = f"{recovery / total:.0%}" if total > 0 else "—"

        if strategy == Strategy.FULL_HISTORY:
            style = "dim"
        elif strategy == Strategy.COMPACTION_ONLY:
            style = "red" if s["lost"] > 0 else "green"
        else:
            style = "yellow"

        table.add_row(
            strategy.value,
            str(s["available"]),
            str(s["lost"]),
            str(s["recovered"]),
            rate,
            style=style,
        )

    console.print(table)

    # Facts by category
    console.print("\n[bold]Facts by Category[/bold]\n")

    cat_table = Table(show_header=True, header_style="bold magenta")
    cat_table.add_column("Category", width=22)
    cat_table.add_column("Total", justify="right", width=8)
    cat_table.add_column("In Summary", justify="right", width=10)
    cat_table.add_column("Lost", justify="right", width=8)
    cat_table.add_column("Retrieval Recovers", justify="right", width=18)

    for cat in FactCategory:
        facts_in_cat = [f for f in report.facts if f.category == cat]
        if not facts_in_cat:
            continue

        comp_results = [
            r for r in report.results
            if r.fact.category == cat and r.strategy == Strategy.COMPACTION_ONLY
        ]
        ret_results = [
            r for r in report.results
            if r.fact.category == cat and r.strategy == Strategy.RETRIEVAL
        ]

        in_summary = sum(1 for r in comp_results if r.status == FactStatus.AVAILABLE)
        lost = sum(1 for r in comp_results if r.status == FactStatus.LOST)
        recovered = sum(1 for r in ret_results if r.status == FactStatus.RECOVERED)

        cat_table.add_row(
            cat.value,
            str(len(facts_in_cat)),
            str(in_summary),
            str(lost),
            str(recovered),
        )

    console.print(cat_table)

    # Lost facts detail
    lost_results = [
        r for r in report.results
        if r.strategy == Strategy.COMPACTION_ONLY and r.status == FactStatus.LOST
    ]

    if lost_results:
        console.print(
            f"\n[bold red]Lost Facts ({len(lost_results)})[/bold red] — "
            f"present in history, absent from compaction summary:\n"
        )

        # Group by category
        for cat in FactCategory:
            cat_lost = [r for r in lost_results if r.fact.category == cat]
            if not cat_lost:
                continue
            console.print(f"  [bold magenta]{cat.value}[/bold magenta]:")
            for r in cat_lost[:10]:  # Limit to 10 per category
                value = r.fact.value[:120]
                # Check if retrieval recovers it
                ret_result = next(
                    (rr for rr in report.results
                     if rr.fact is r.fact and rr.strategy == Strategy.RETRIEVAL),
                    None,
                )
                recovered_marker = ""
                if ret_result and ret_result.status == FactStatus.RECOVERED:
                    recovered_marker = " [green](retrieval recovers)[/green]"

                console.print(f"    [dim]-[/dim] {value}{recovered_marker}")

            if len(cat_lost) > 10:
                console.print(f"    [dim]... and {len(cat_lost) - 10} more[/dim]")
        console.print()

    if verbose:
        _print_lost_facts_verbose(report)


def _print_lost_facts_verbose(report: "CompactionLossReport") -> None:
    """Print full detail of lost facts with retrieval evidence."""
    lost_facts = [
        r.fact for r in report.results
        if r.strategy == Strategy.COMPACTION_ONLY and r.status == FactStatus.LOST
    ]

    if not lost_facts:
        return

    console.print("\n[bold]Verbose Lost Facts Detail[/bold]\n")

    for fact in lost_facts:
        console.print(f"  [bold]{fact.category.value}[/bold] (record idx={fact.source_index})")
        console.print(f"  [dim]Value:[/dim] {fact.value[:200]}")
        console.print(f"  [dim]Search terms:[/dim] {', '.join(fact.search_terms[:10])}")

        ret_result = next(
            (r for r in report.results
             if r.fact is fact and r.strategy == Strategy.RETRIEVAL),
            None,
        )
        if ret_result:
            if ret_result.status == FactStatus.RECOVERED:
                console.print(f"  [green]Retrieval evidence:[/green] {ret_result.evidence[:200]}")
            else:
                console.print(f"  [red]Retrieval: NOT RECOVERED[/red]")
        console.print()


def _extract_project_name(path: Path) -> str:
    """Extract project name from session path.

    Session directories are named like:
    --Users-example-code-project-a--
    --Users-example-code-project-b--
    """
    # Parent directory name
    parent = path.parent.name
    # Strip leading/trailing dashes and convert to readable form
    parts = parent.strip("-").split("-")
    # Try to find the project name (usually after "code")
    if "code" in parts:
        idx = parts.index("code")
        return "-".join(parts[idx + 1 :]) or parent
    return parent.strip("-")

def cmd_report(args: argparse.Namespace) -> int:
    """Generate an aggregate compaction-loss report across all sessions."""
    loader = _get_loader(getattr(args, 'harness', 'pi'))
    generator = ReportGenerator(loader=loader)
    redactor = SessionRedactor() if args.redact else None
    report = generator.generate(args.sessions_dir, redactor=redactor, console=console)

    if args.json:
        # Output JSON to stdout or file
        json_str = report.to_json()
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
            console.print(f"[green]Report written to {args.output}[/green]")
        else:
            print(json_str)
    else:
        print_report(report, console=console)
        if args.output:
            json_str = report.to_json()
            Path(args.output).write_text(json_str, encoding="utf-8")
            console.print(f"\n[green]JSON report written to {args.output}[/green]")

    return 0
def cmd_label(args: argparse.Namespace) -> int:
    """Fact-labeling tool: generate sample, annotate, or compute stats."""
    if args.label_action == "sample":
        loader = _get_loader(getattr(args, 'harness', 'pi'))
        sample = generate_sample(args.sessions_dir, console=console, loader=loader)
        json_str = sample.to_json()
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
            console.print(f"[green]Sample written to {args.output}[/green]")
        else:
            print(json_str)
        console.print(f"[green]Generated {len(sample.facts)} facts for labeling[/green]")
    elif args.label_action == "annotate":
        sample = _load_sample(args.sample_file)
        # Save to the same file after annotation (resumable)
        out_path = args.output or args.sample_file
        sample = annotate_sample(sample, console=console)
        Path(out_path).write_text(sample.to_json(), encoding="utf-8")
        console.print(f"[green]Labels saved to {out_path}[/green]")
    elif args.label_action == "stats":
        sample = _load_sample(args.sample_file)
        stats = compute_stats(sample, console=console)
        if args.output and stats:
            Path(args.output).write_text(
                json.dumps(stats, indent=2), encoding="utf-8"
            )
            console.print(f"[green]Stats written to {args.output}[/green]")
    return 0


def _load_sample(path: str) -> LabelSample:
    """Load a LabelSample from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LabelSample.from_json(data)
def cmd_attack(args: argparse.Namespace) -> int:
    """Run compaction-hiding attack fixtures and report results."""
    console.print("[bold]Running compaction information hiding attack (H5a)...[/bold]\n")
    results = run_all_variants(console=console)
    console.print()
    print_attack_report(results, console=console)
    return 0


def cmd_vuln(args: argparse.Namespace) -> int:
    """Detect compaction-induced vulnerability windows across sessions."""
    from agent_fail_eval.vuln_window import (
        ConstraintClassifier,
        AdversarialAssertionGenerator,
        VulnerabilityWindowDetector,
    )
    import dataclasses, json as _json

    sessions_dir = Path(args.sessions_dir)
    loader = _get_loader(getattr(args, 'harness', 'pi'))
    sessions = loader.scan_directory(sessions_dir)
    if not sessions:
        console.print(f"[yellow]No sessions with compaction found in[/yellow] {sessions_dir}")
        return 0

    classifier = ConstraintClassifier()
    generator = AdversarialAssertionGenerator(model=args.model) if args.generate else None
    detector = VulnerabilityWindowDetector(classifier=classifier, generator=generator)

    console.print(f"Scanning [cyan]{len(sessions)}[/cyan] sessions for vulnerability windows…")
    report = detector.detect_windows(sessions)

    console.print(f"\n[bold]Window Detection Report[/bold]")
    console.print(report.summary())

    if report.windows:
        t = Table(title=f"Vulnerability Windows ({report.total_windows} found)")
        t.add_column("Session ID", style="dim", max_width=32)
        t.add_column("Boundary", justify="right")
        t.add_column("Dropped", justify="right")
        t.add_column("Types")
        t.add_column("Assertions", justify="right")
        for w in report.windows:
            types = ", ".join(sorted({c.constraint_type.value for c in w.dropped_constraints}))
            t.add_row(
                w.session_id[:32],
                str(w.boundary_index),
                str(len(w.dropped_constraints)),
                types,
                str(len(w.adversarial_assertions)),
            )
        console.print(t)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            _json.dump(dataclasses.asdict(report), f, indent=2, default=str)
        console.print(f"Report written to [cyan]{out}[/cyan]")

    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="afe",
        description="Coding-agent failure evaluation toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse
    parse_cmd = subparsers.add_parser(
        "parse", help="Parse a single session and print a summary"
    )
    parse_cmd.add_argument("session_file", help="Path to the .jsonl session file")
    parse_cmd.add_argument(
        "--compaction",
        type=int,
        default=None,
        help="Show full details of compaction at this index",
    )
    parse_cmd.add_argument(
        "--harness",
        choices=["pi", "claude"],
        default="pi",
        help="Session format: 'pi' for pi-agent, 'claude' for Claude Code",
    )

    # scan
    scan_cmd = subparsers.add_parser(
        "scan", help="Scan a directory for sessions with compaction events"
    )
    scan_cmd.add_argument("sessions_dir", help="Path to sessions directory")
    scan_cmd.add_argument(
        "--harness",
        choices=["pi", "claude"],
        default="pi",
        help="Session format: 'pi' for pi-agent, 'claude' for Claude Code",
    )

    # eval
    eval_cmd = subparsers.add_parser(
        "eval", help="Evaluate compaction loss and print a failure report"
    )
    eval_cmd.add_argument("session_file", help="Path to the .jsonl session file")
    eval_cmd.add_argument(
        "--compaction",
        type=int,
        default=None,
        help="Evaluate only the compaction at this index",
    )
    eval_cmd.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show full detail of lost facts with retrieval evidence",
    )
    # report
    report_cmd = subparsers.add_parser(
        "report", help="Generate aggregate compaction-loss report across all sessions"
    )
    report_cmd.add_argument("sessions_dir", help="Path to sessions directory")
    report_cmd.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted table",
    )
    report_cmd.add_argument(
        "-o", "--output",
        default=None,
        help="Write JSON report to this file",
    )
    report_cmd.add_argument(
        "--redact",
        action="store_true",
        help="Redact sensitive content (paths, secrets, code) from session data and reports",
    )
    report_cmd.add_argument(
        "--harness",
        choices=["pi", "claude"],
        default="pi",
        help="Session format: 'pi' for pi-agent, 'claude' for Claude Code",
    )

    # label
    label_cmd = subparsers.add_parser(
        "label", help="Fact-labeling tool for measuring extractor precision/recall"
    )
    label_sub = label_cmd.add_subparsers(dest="label_action", required=True)

    label_sample = label_sub.add_parser(
        "sample", help="Generate a stratified sample of facts for labeling"
    )
    label_sample.add_argument("sessions_dir", help="Path to sessions directory")
    label_sample.add_argument(
        "-o", "--output",
        default=None,
        help="Write sample JSON to this file",
    )
    label_sample.add_argument(
        "--harness",
        choices=["pi", "claude"],
        default="pi",
        help="Session format: 'pi' for pi-agent, 'claude' for Claude Code",
    )

    label_annotate = label_sub.add_parser(
        "annotate", help="Interactive labeling session over a sample file"
    )
    label_annotate.add_argument("sample_file", help="Path to the sample JSON file")
    label_annotate.add_argument(
        "-o", "--output",
        default=None,
        help="Save labels to this file (default: overwrite input)",
    )

    label_stats = label_sub.add_parser(
        "stats", help="Compute precision/recall from labeled sample"
    )
    label_stats.add_argument("sample_file", help="Path to the labeled sample JSON")
    label_stats.add_argument(
        "-o", "--output",
        default=None,
        help="Write stats JSON to this file",
    )

    # attack
    attack_cmd = subparsers.add_parser(
        "attack", help="Run compaction-hiding attack fixtures (harness-adversarial H5a)"
    )

    # vuln
    vuln_cmd = subparsers.add_parser(
        "vuln",
        help="Detect compaction-induced vulnerability windows across sessions",
    )
    vuln_cmd.add_argument("sessions_dir", help="Path to sessions directory")
    vuln_cmd.add_argument(
        "--generate",
        action="store_true",
        help="Generate adversarial assertions via OpenRouter (requires OPENROUTER_API_KEY)",
    )
    vuln_cmd.add_argument(
        "--model",
        default="anthropic/claude-haiku-4",
        help="Model for assertion generation",
    )
    vuln_cmd.add_argument(
        "-o", "--output",
        default=None,
        help="Write JSON report to this file",
    )
    vuln_cmd.add_argument(
        "--harness",
        choices=["pi", "claude"],
        default="pi",
        help="Session format: 'pi' for pi-agent, 'claude' for Claude Code",
    )

    args = parser.parse_args()

    if args.command == "parse":
        sys.exit(cmd_parse(args))
    elif args.command == "scan":
        sys.exit(cmd_scan(args))
    elif args.command == "eval":
        sys.exit(cmd_eval(args))
    elif args.command == "report":
        sys.exit(cmd_report(args))
    elif args.command == "label":
        sys.exit(cmd_label(args))
    elif args.command == "attack":
        sys.exit(cmd_attack(args))
    elif args.command == "vuln":
        sys.exit(cmd_vuln(args))


if __name__ == "__main__":
    main()
