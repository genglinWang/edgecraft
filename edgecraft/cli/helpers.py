"""CLI helper functions for formatting and displaying agent/trial data."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def get_imgsz_from_variant(variant: Any) -> str:
    """Get imgsz from SolutionVariant or variant dict (from model_dump)."""
    if hasattr(variant, "get_imgsz"):
        return str(variant.get_imgsz())
    train_code = (variant or {}).get("train_code") or ""
    match = re.search(r"imgsz[=\s]*(\d+)", train_code)
    return match.group(1) if match else "—"


def get_primary_metric_display(trial: Any) -> str:
    """Get first local metric value for display (TrialResult has all_metrics, not primary_metric_value)."""
    if not trial.local_metrics or not trial.local_metrics.all_metrics:
        return "—"
    first_val = next(iter(trial.local_metrics.all_metrics.values()), None)
    if first_val is None:
        return "—"
    return f"{first_val:.4f}"


def print_trial_summary_table(console: Console, agent_result: dict) -> None:
    """Print a compact table of all trials from the agent result."""
    from edgecraft.agent.search.trial_bank import TrialBank

    run_id = agent_result.get("run_id", "")
    best_id = (agent_result.get("best_trial") or {}).get("trial_id")

    trial_objects = []
    try:
        from edgecraft.agent.workspace.manager import trial_bank_path

        persist_path = trial_bank_path(
            run_id,
            tenant_id=agent_result.get("tenant_id") or "default",
        )
        bank = TrialBank(run_id=run_id, persist_path=persist_path)
        trial_objects = bank.get_all()
    except Exception:
        pass

    if not trial_objects:
        ids = agent_result.get("all_trials", [])
        if ids:
            console.print(f"Trials run: {', '.join(ids)}")
        return

    status_str = agent_result.get("status", "?")
    status_color = "green" if status_str == "completed" else "yellow"
    console.print(
        f"[bold]Status:[/bold] [{status_color}]{status_str}[/{status_color}]  "
        f"[bold]Run:[/bold] {run_id}  "
        f"[bold]Trials:[/bold] {agent_result.get('iterations', len(trial_objects))}/{agent_result.get('iterations', len(trial_objects))}"
    )
    console.print()

    tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    tbl.add_column("#", style="dim", width=3, justify="right")
    tbl.add_column("trial_id", style="cyan", width=14)
    tbl.add_column("model", width=22)
    tbl.add_column("quant/fmt", width=12)
    tbl.add_column("imgsz", width=6, justify="right")
    tbl.add_column("stage", width=14)
    tbl.add_column("metric", width=10, justify="right")
    tbl.add_column("p95 lat.", width=9, justify="right")
    tbl.add_column("mem", width=7, justify="right")
    tbl.add_column("score", width=7, justify="right")
    tbl.add_column("ok?", width=4)

    for i, t in enumerate(trial_objects, 1):
        v = t.variant
        is_best = t.trial_id == best_id

        metric_str = get_primary_metric_display(t)
        decision_latency = None
        if t.edge_metrics:
            decision_latency = (
                t.edge_metrics.latency_p95_ms
                if t.edge_metrics.latency_p95_ms is not None
                else t.edge_metrics.latency_ms
            )
        latency_str = (
            f"{decision_latency:.1f}ms" if decision_latency is not None else "—"
        )
        mem_str = (
            f"{t.edge_metrics.memory_mb:.0f}MB"
            if t.edge_metrics and t.edge_metrics.memory_mb is not None else "—"
        )
        score_str = f"{t.score:.4f}" if t.score else "—"
        imgsz = get_imgsz_from_variant(v)
        quant_fmt = f"{v.quant_mode}/{v.export_format}"
        stage = t.stage_reached.value if t.stage_reached else "—"
        feasible = "[green]✓[/green]" if t.is_feasible else "[red]✗[/red]"

        row_style = "bold green" if is_best else ("red" if t.error else "")

        tbl.add_row(
            str(i),
            t.trial_id,
            v.model_name,
            quant_fmt,
            str(imgsz),
            stage,
            metric_str,
            latency_str,
            mem_str,
            score_str,
            feasible,
            style=row_style,
        )

        if t.error:
            tbl.add_row(
                "", "", f"[dim red]  └ {t.error_stage}: {t.error[:60]}[/dim red]",
                "", "", "", "", "", "", "", "",
            )

    console.print(tbl)
