from __future__ import annotations

import os
from pathlib import Path
from collections import Counter, defaultdict
import builtins
import ipaddress
import json
import re
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from edgecraft.cli.banner import SYNTH_BANNER
from edgecraft.cli.helpers import (
    get_imgsz_from_variant,
    print_trial_summary_table,
)
from edgecraft.cli.settings_overrides import (
    build_synth_settings_overrides,
    describe_settings_overrides,
    temporary_settings_attrs,
)
from edgecraft.config.profiles import (
    profile_default_iterations,
    profile_settings,
    resolved_profile_manifest,
)

if TYPE_CHECKING:
    from edgecraft.utils.synth_checks import DevicePreflight, SynthCheckResult

console = Console()

_ULTRA_SIZE_RANK = {
    "n": 0,
    "t": 1,
    "s": 2,
    "m": 3,
    "b": 4,
    "l": 5,
    "x": 6,
    "c": 7,
    "e": 8,
    "p": 9,
}


def _ultralytics_sort_tuple(model_name: str) -> tuple:
    """Natural display order for Ultralytics model IDs."""
    # yolo/yolov series: e.g. yolov8n, yolov10m, yolo11x, yolo26n, yolo11n-obb
    m = re.match(r"^yolov?(\d+)(.*)$", model_name)
    if m:
        version = int(m.group(1))
        rest = m.group(2).lstrip("-_")
        size = ""
        for ch in rest:
            if ch.isalpha():
                size += ch
                break
        if not size:
            size = rest.split("-")[0].split("_")[0] if rest else ""
        size_rank = _ULTRA_SIZE_RANK.get(size, 99)
        return (0, version, size_rank, rest, model_name)

    # yoloe family: e.g. yoloe-11s, yoloe-11m-seg
    m = re.match(r"^yoloe-(\d+)([a-z]?)(.*)$", model_name)
    if m:
        version = int(m.group(1))
        size = m.group(2) or ""
        size_rank = _ULTRA_SIZE_RANK.get(size, 99)
        rest = m.group(3).lstrip("-_")
        return (1, version, size_rank, rest, model_name)

    # yoloworld family
    m = re.match(r"^yoloworld-([a-z]+)$", model_name)
    if m:
        size = m.group(1)
        return (2, 0, _ULTRA_SIZE_RANK.get(size, 99), "", model_name)

    # yolo_nas family
    m = re.match(r"^yolo_nas_([a-z]+)$", model_name)
    if m:
        size = m.group(1)
        return (3, 0, _ULTRA_SIZE_RANK.get(size, 99), "", model_name)

    # rtdetr family
    m = re.match(r"^rtdetr-([a-z]+)$", model_name)
    if m:
        size = m.group(1)
        return (4, 0, _ULTRA_SIZE_RANK.get(size, 99), "", model_name)

    return (9, 0, 99, "", model_name)


def _model_display_sort_key(model_id: str, runtime: str) -> tuple:
    """Stable display order across families; natural order for Ultralytics."""
    if ":" in model_id:
        family, model_name = model_id.split(":", 1)
    else:
        family, model_name = "unknown", model_id
    if family == "ultralytics":
        return (family, *_ultralytics_sort_tuple(model_name), runtime)
    return (family, model_name, runtime)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """EdgeCraft CLI: MCaaS prototype for IoT/Edge AI

    Automatically synthesize, optimize, and deploy AI models to edge devices.
    """
    pass


from edgecraft.cli.data import data_commands
main.add_command(data_commands)


def _load_synth_preflight_snapshot(
    path: Path,
    *,
    intent: str,
    dataset: str | None,
    device_ip: str | None,
) -> SynthCheckResult:
    """Load one preflight only when it describes this exact request."""
    from edgecraft.utils.synth_checks import SynthCheckResult

    result = SynthCheckResult.model_validate_json(path.read_text(encoding="utf-8"))
    mismatches = []
    if not result.success or not (result.intent and result.dataset and result.device_ctx):
        mismatches.append("snapshot is not a successful complete preflight")
    else:
        if result.intent.raw_user_intent != intent:
            mismatches.append("intent")
        if dataset is None or Path(result.dataset.root_path).resolve() != Path(dataset).resolve():
            mismatches.append("dataset")
        if result.device_ctx.device_ip != (device_ip or ""):
            mismatches.append("device_ip")
    if mismatches:
        raise ValueError(
            "preflight snapshot does not match this request: " + ", ".join(mismatches)
        )
    return result


def _save_synth_preflight_snapshot(path: Path, result: SynthCheckResult) -> None:
    """Atomically preserve the structured facts consumed by the search arms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    pending.replace(path)


def _load_device_preflight_snapshot(
    path: Path, *, device_ip: str | None
) -> DevicePreflight:
    """Load reusable measured device facts without reusing request semantics."""
    from edgecraft.utils.synth_checks import DevicePreflight

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("device_ctx", "device"):
            if key in payload:
                payload = payload[key]
                break
    device = DevicePreflight.model_validate(payload)
    mismatches = []
    if device.device_ip != (device_ip or ""):
        mismatches.append("device_ip")
    if device.device_info.status != "success" or not device.device_id:
        mismatches.append("device probe is not successful")
    if device.device_info.device_id != device.device_id:
        mismatches.append("device identity")
    if not device.runtime_config.available_runtimes:
        mismatches.append("runtime evidence is empty")
    if mismatches:
        raise ValueError(
            "device preflight snapshot is unusable: " + ", ".join(mismatches)
        )
    return device


def _save_device_preflight_snapshot(path: Path, device: DevicePreflight) -> None:
    """Atomically preserve device/runtime facts independently of a request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(device.model_dump_json(indent=2), encoding="utf-8")
    pending.replace(path)


@main.command()
@click.argument('intent')
@click.option('--dataset', '-d', help='Path to dataset')
@click.option('--ip', help='Edge device IP (user@host format)')
@click.option(
    '--ssh-key',
    '-k',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Explicit SSH private key path (used with IdentitiesOnly=yes).',
)
@click.option(
    '--profile',
    'profile_name',
    type=click.Choice(['paper', 'paper-audit', 'offline'], case_sensitive=False),
    default=None,
    help='Mechanism profile; defaults to EDGECRAFT_PROFILE (paper).',
)
@click.option('--iterations', '-n', default=None, type=int,
              help='Maximum tree-search trials (paper/audit: 24; offline: 2).')
@click.option(
    '--branching-factor',
    type=int,
    default=None,
    help='Number of variants proposed per tree expansion (overrides env for this run).',
)
@click.option('--run-id', default=None, help='Resume or label a specific run')
@click.option(
    '--continuation-parent-trial-id',
    default=None,
    help='Expand this existing trial once when resuming a TrialBank.',
)
@click.option(
    '--debugger-enabled',
    type=click.Choice(['on', 'off'], case_sensitive=False),
    default=None,
    help='Temporarily override debugger switch for this synth run only.',
)
@click.option(
    '--debugger-retries',
    type=int,
    default=None,
    help='Temporarily override debugger max retries for this synth run only.',
)
@click.option(
    '--docker-image',
    required=False,
    default=None,
    help='Pre-built Docker image for containerized edge hosts; omit for native-runtime devices.',
)
@click.option(
    '--tenant',
    'tenant_id',
    type=str,
    default='default',
    help='Tenant identifier used by the scheduler for fair-share ordering when multiple synth requests run concurrently.',
)
@click.option('--modality-hint', default=None, help='Optional user/task contract for input modality, e.g. audio or vision.')
@click.option('--task-type-hint', default=None, help='Optional user/task contract for task type, e.g. audio_classification.')
@click.option(
    '--split-manifest',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help='Optional immutable evaluation split manifest. Recorded as dataset evidence; old loaders remain compatible.',
)
@click.option(
    '--preflight-snapshot',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help='Reuse a request-matched structured preflight snapshot.',
)
@click.option(
    '--save-preflight-snapshot',
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help='Save the structured preflight consumed by this run.',
)
@click.option(
    '--device-preflight-snapshot',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help='Reuse measured device/runtime facts while re-running intent and dataset preflight.',
)
@click.option(
    '--save-device-preflight-snapshot',
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help='Save reusable measured device/runtime facts from this preflight.',
)
def synth(
    intent,
    dataset,
    ip,
    ssh_key,
    profile_name,
    iterations,
    branching_factor,
    run_id,
    continuation_parent_trial_id,
    debugger_enabled,
    debugger_retries,
    docker_image,
    tenant_id,
    modality_hint,
    task_type_hint,
    split_manifest,
    preflight_snapshot,
    save_preflight_snapshot,
    device_preflight_snapshot,
    save_device_preflight_snapshot,
):
    """Synthesize a model from natural language description.

    Constraints are extracted from your intent with LLM (UserSpec).
    The agent uses iterative tree search over model configurations, training on
    your dataset and benchmarking on the target edge device.

    Examples:

      edgecraft synth "..." -d ./data --ip user@host -k ~/.ssh/key --docker-image edgecraft-runtime:latest

      edgecraft synth "..." -d /data -n 24 --ip user@host -k ~/.ssh/key --docker-image edgecraft-runtime:latest
    """
    from edgecraft.agent import run_agent
    from edgecraft.config.settings import settings as app_settings
    from edgecraft.utils.synth_checks import run_synth_preflight

    profile_name = (profile_name or app_settings.PROFILE or "paper").strip().lower()
    try:
        profile_updates = profile_settings(profile_name)
        if iterations is None:
            iterations = profile_default_iterations(profile_name)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if iterations < 1:
        raise click.UsageError("--iterations must be >= 1")

    # print banner
    console.print(SYNTH_BANNER)

    if preflight_snapshot and save_preflight_snapshot:
        raise click.UsageError(
            "--preflight-snapshot and --save-preflight-snapshot are mutually exclusive"
        )
    if preflight_snapshot and device_preflight_snapshot:
        raise click.UsageError(
            "--preflight-snapshot and --device-preflight-snapshot are mutually exclusive"
        )

    device_snapshot = None
    if device_preflight_snapshot:
        try:
            device_snapshot = _load_device_preflight_snapshot(
                device_preflight_snapshot,
                device_ip=ip,
            )
        except (OSError, ValueError) as exc:
            raise click.UsageError(str(exc)) from exc

    # Run the ordinary checks once, then reuse the exact structured facts for
    # causal search comparisons. Normal synth calls continue down this path.
    if preflight_snapshot:
        try:
            result = _load_synth_preflight_snapshot(
                preflight_snapshot,
                intent=intent,
                dataset=dataset,
                device_ip=ip,
            )
        except (OSError, ValueError) as exc:
            raise click.UsageError(str(exc)) from exc
        console.print(f"[dim]Reusing preflight snapshot: {preflight_snapshot}[/dim]")
    else:
        # The named profile governs admission as well as the later search loop.
        # This matters when the process-level environment selects a weaker
        # profile but the caller explicitly requests the paper path.
        with temporary_settings_attrs(app_settings, profile_updates):
            result = run_synth_preflight(
                intent,
                dataset,
                ip,
                ssh_key,
                console,
                docker_image=docker_image,
                modality_hint=modality_hint,
                task_type_hint=task_type_hint,
                device_preflight=device_snapshot,
            )
    if not result.success:
        console.print(f"[bold yellow]Warning:[/bold yellow] {result.message}")
        raise click.Abort()

    if split_manifest is not None and result.dataset is not None:
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        content_hash = str(manifest.get("content_hash") or "")
        if not content_hash:
            console.print("[bold red]Error:[/bold red] split manifest has no content_hash")
            raise click.Abort()
        result.dataset.info["split_manifest"] = {
            "path": str(split_manifest.resolve()),
            "content_hash": content_hash,
            "counts": manifest.get("counts") or {},
            "strategy": manifest.get("strategy") or "",
            "schema": {
                "top_level_keys": sorted(manifest),
                "splits": {
                    str(name): {
                        "container": "list[str]" if isinstance(ids, builtins.list) else type(ids).__name__,
                        "count": len(ids) if isinstance(ids, builtins.list) else None,
                        "examples": [str(item) for item in ids[:2]] if isinstance(ids, builtins.list) else [],
                    }
                    for name, ids in (manifest.get("splits") or {}).items()
                },
            },
        }

    if save_preflight_snapshot is not None:
        _save_synth_preflight_snapshot(save_preflight_snapshot, result)
        console.print(f"[dim]Saved preflight snapshot: {save_preflight_snapshot}[/dim]")
    if save_device_preflight_snapshot is not None and result.device_ctx is not None:
        _save_device_preflight_snapshot(save_device_preflight_snapshot, result.device_ctx)
        console.print(
            f"[dim]Saved device preflight snapshot: {save_device_preflight_snapshot}[/dim]"
        )

    # print iterative tree search banner
    console.print(Rule(f"[bold]Iterative Tree Search[/bold]  run_id={run_id or '(new)'}  max={iterations} trials", style="blue"))
    console.print()

    # build synth settings overrides
    try:
        settings_updates = {
            **profile_updates,
            **build_synth_settings_overrides(
                debugger_enabled=debugger_enabled,
                debugger_retries=debugger_retries,
            ),
        }
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise click.Abort()

    # print synth settings overrides
    console.print(
        f"[dim]Profile: {profile_name}; "
        f"verifier={settings_updates['VERIFIER_MODE']}; "
        f"scheduler={settings_updates['SCHEDULER_POLICY']}[/dim]"
    )
    explicit_updates = {
        key: value
        for key, value in settings_updates.items()
        if key not in profile_updates or profile_updates[key] != value
    }
    _override_msg = describe_settings_overrides(explicit_updates)
    if _override_msg:
        console.print(f"[dim]{_override_msg}[/dim]")
    console.print()

    # check branching factor
    if branching_factor is not None and branching_factor < 1:
        console.print("[bold red]Error:[/bold red] --branching-factor must be >= 1")
        raise click.Abort()

    # expand ssh key
    ssh_key = os.path.expanduser(ssh_key) if ssh_key else ssh_key

    # run agent
    try:
        with temporary_settings_attrs(app_settings, settings_updates):
            # run agent with temporary settings overrides
            agent_result = run_agent(
                preflight=result,
                ssh_key=ssh_key,
                max_iterations=iterations,
                branching_factor=branching_factor,
                run_id=run_id,
                tenant_id=tenant_id,
                continuation_parent_trial_id=continuation_parent_trial_id,
            )
    except Exception as exc:
        # print error
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise click.Abort()

    # print trial summary table
    console.print()
    console.print(Rule("[bold]Search Results[/bold]", style="blue"))

    # print best solution detail
    print_trial_summary_table(console, agent_result)

    # print best solution detail
    best = agent_result.get("best_trial")
    if best:
        console.print()
        if agent_result.get("constraints_satisfied") is False:
            console.print(
                "[bold yellow]Best P2-Verified Candidate (constraints unmet)[/bold yellow]"
            )
        else:
            console.print("[bold green]Best Feasible Solution[/bold green]")
        variant = best.get("variant", {})

        # print best solution detail table
        best_table = Table(show_header=False, box=None)
        best_table.add_column("", style="bold cyan", width=22)
        best_table.add_column("")

        # print best solution detail table rows
        best_table.add_row("trial_id", best.get("trial_id", "-"))
        best_table.add_row("model", variant.get("model_name", "-"))
        best_table.add_row("quant_mode", variant.get("quant_mode", "-"))
        best_table.add_row("export_format", variant.get("export_format", "-"))

        # print best solution detail table rows
        imgsz = get_imgsz_from_variant(variant)
        if imgsz != "—":
            best_table.add_row("imgsz", imgsz)
        best_table.add_row("score", f"{best.get('score', 0):.4f}")

        # print best solution detail table rows

        # print best solution detail table rows
        local = best.get("local_metrics")
        if local:
            all_metrics = local.get("all_metrics") or {}
            for name, value in all_metrics.items():
                best_table.add_row(name, f"{value:.4f}")

        # print best solution detail table rows
        edge = best.get("edge_metrics")
        if edge:
            latency_p95 = edge.get("latency_p95_ms")
            if latency_p95 is None:
                latency_p95 = edge.get("latency_ms")
            best_table.add_row(
                "latency_p95_ms",
                f"{float(latency_p95):.1f}" if latency_p95 is not None else "not measured",
            )
            memory_mb = edge.get("memory_mb")
            best_table.add_row(
                "memory_mb",
                f"{float(memory_mb):.0f}" if memory_mb is not None else "not measured",
            )

        # print best solution detail table rows
        artifacts = best.get("artifact_paths", {})
        for stage, path in artifacts.items():
            best_table.add_row(f"artifact ({stage})", path)

        if agent_result.get("remaining_gaps"):
            known = [
                item for item in agent_result["remaining_gaps"]
                if item.get("known") and item.get("gap") is not None
            ]
            for item in known:
                best_table.add_row(
                    f"remaining gap ({item.get('metric', '?')})",
                    f"{float(item['gap']):.4g}",
                )

        # print best solution detail table
        console.print(best_table)

    else:
        # print no feasible solution found
        console.print()
        console.print(
            "[yellow]No feasible solution found. "
            "Try relaxing constraints or increasing --iterations.[/yellow]"
        )

    # print reflector diagnosis if available
    diagnosis = agent_result.get("reflector_diagnosis")
    if diagnosis:
        console.print()
        # print reflector diagnosis
        console.print(f"[dim]Reflector: {diagnosis.get('diagnosis')} — "
                      f"{diagnosis.get('reasoning', '')}[/dim]")

    # print error if any
    if agent_result.get("error"):
        console.print(f"\n[bold red]Error:[/bold red] {agent_result['error']}")


@main.group("profile")
def profile_commands():
    """Inspect named, non-secret EdgeCraft mechanism profiles."""
    pass


@profile_commands.command("show")
@click.argument(
    "profile_name",
    required=False,
    type=click.Choice(["paper", "paper-audit", "offline"], case_sensitive=False),
)
def show_profile(profile_name):
    """Print the exact switches selected by a mechanism profile."""
    from edgecraft.config.settings import settings as app_settings

    profile_name = (profile_name or app_settings.PROFILE or "paper").strip().lower()
    updates = profile_settings(profile_name)
    with temporary_settings_attrs(app_settings, updates):
        _print_json(resolved_profile_manifest(app_settings, profile_name))


@main.group("evidence")
def evidence_commands():
    """Export small, allowlisted review evidence snapshots."""
    pass


@evidence_commands.command("export-rules")
@click.option("--store", "store_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--overwrite", is_flag=True, help="Replace an existing output file.")
def export_rules(store_path, output, overwrite):
    """Export verified rules without raw failures or private provenance."""
    from edgecraft.knowledge.compatibility import CompatibilityRuleStore

    path = CompatibilityRuleStore(path=store_path).export_public_snapshot(
        output,
        overwrite=overwrite,
    )
    console.print(f"[green]Wrote sanitized rule snapshot:[/green] {path}")


@evidence_commands.command("export-calibration")
@click.option("--store", "store_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--overwrite", is_flag=True, help="Replace an existing output file.")
def export_calibration(store_path, output, overwrite):
    """Export exact-context P1/P2 pairs without private provenance."""
    from edgecraft.knowledge.calibration_store import CalibrationStore

    path = CalibrationStore(store_path).export_public_snapshot(
        output,
        overwrite=overwrite,
    )
    console.print(f"[green]Wrote sanitized calibration snapshot:[/green] {path}")


@main.command()
@click.argument('artifact_path')
@click.argument('device_ip')
@click.option(
    '--key',
    '-k',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Explicit SSH private key path (used with IdentitiesOnly=yes).',
)
@click.option('--device-id', default='edge_device', help='Device identifier')
@click.option(
    '--docker-image',
    '-i',
    required=False,
    default=None,
    help='Pre-built Docker image for containerized edge hosts; omit for native-runtime devices.',
)
def deploy(artifact_path, device_ip, key, device_id, docker_image):
    """Deploy a model artifact to an edge device.

    Example:
        edgecraft deploy ./model.pt "$DEVICE_HOST" -k "$SSH_KEY_PATH" -i "$EDGE_DOCKER_IMAGE"
    """
    from edgecraft.tools.deploy import EdgeRunner

    console.print(f"[bold green]📦 Deploying to edge device[/bold green]")
    console.print(f"Artifact: {artifact_path}")
    console.print(f"Device: {device_ip}")
    console.print()

    key = os.path.expanduser(key) if key else key
    runner = EdgeRunner(ssh_key_path=key)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Deploying...", total=None)

        result = runner.deploy(
            artifact_path=artifact_path,
            device_id=device_id,
            device_ip=device_ip,
            ssh_key=key,
            docker_image=docker_image,
        )

        progress.update(task, completed=True)

    console.print()
    console.print(f"[bold]Status:[/bold] {result.get('status')}")

    if result.get("error"):
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
    else:
        console.print("[bold green]✓ Deployment complete[/bold green]")



@profile_commands.command("devices")
@click.argument('device_ip')
@click.option(
    '--ssh-key',
    '-k',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Explicit SSH private key path (used with IdentitiesOnly=yes).',
)
@click.option('--device-id', default='jetson_orin_agx',
              help='Device identifier (e.g. jetson_orin_agx)')
@click.option(
    '--docker-image',
    required=True,
    help='Pre-built Docker image on the edge host (local tag).',
)
@click.option('--families', default=None,
              help='Comma-separated family IDs to profile (e.g. ultralytics,timm)')
@click.option('--modalities', default=None,
              help='Optional comma-separated modalities (e.g. vision,audio,text). '
                   'If omitted, profile all modalities.')
@click.option('--runtimes', default=None,
              help='Optional comma-separated runtimes (e.g. pytorch,onnxruntime,tensorrt). '
                   'If omitted, auto-traverse declared runtimes per model.')
@click.option('--imgsz', default=640, type=int, help='Input image size')
@click.option('--timeout', default=600, type=int,
              help='Per-model timeout (seconds). TensorRT builds use 2700s when default.')
@click.option('--dry-run', is_flag=True, help='Print run/skip plan only')
@click.option(
    '--scope',
    type=click.Choice(['all', 'failed'], case_sensitive=False),
    default='all',
    show_default=True,
    help='Profile scope: all planned requests, or only requests with failed records in profile store.',
)
def profile_devices(
    device_ip,
    ssh_key,
    device_id,
    docker_image,
    families,
    modalities,
    runtimes,
    imgsz,
    timeout,
    dry_run,
    scope,
):
    """Profile registered models on a remote edge device via SSH.

    Profiles registered models with random weights on selected runtimes.
    Outputs latency, memory, throughput, and energy (Jetson).

    Example:
        edgecraft profile devices "$DEVICE_HOST" -k "$SSH_KEY_PATH" --docker-image "$EDGE_DOCKER_IMAGE"
    """
    from edgecraft.models.offline_profiler import OfflineProfiler
    from edgecraft.models.specs import RuntimeId
    from edgecraft.models.profile_store import get_profile_store
    from edgecraft.models import initialize_registries
    from edgecraft.utils.docker_preflight import run_docker_image_preflight
    from edgecraft.utils.network import ensure_edge_runner_active, test_ssh_connection

    initialize_registries()
    ssh_key = os.path.expanduser(ssh_key)

    effective_image = docker_image.strip()
    if not effective_image:
        console.print(
            "[bold red]Error:[/bold red] --docker-image must be a non-empty image name."
        )
        raise click.Abort()

    family_ids = [f.strip() for f in families.split(',')] if families else None
    modality_ids = [m.strip() for m in modalities.split(',') if m.strip()] if modalities else None
    runtime_ids = None
    runtime_map = {
        "onnxruntime": RuntimeId.ONNXRUNTIME,
        "tensorrt": RuntimeId.TENSORRT,
        "pytorch": RuntimeId.PYTORCH,
    }
    if runtimes:
        runtime_ids = []
        for r in [x.strip().lower() for x in runtimes.split(",") if x.strip()]:
            if r not in runtime_map:
                console.print(f"[bold red]Error:[/bold red] unsupported runtime '{r}'")
                raise click.Abort()
            runtime_ids.append(runtime_map[r])

    console.print(f"[bold green]Profilers[/bold green]")
    console.print(f"  Device: {device_ip}")
    console.print(f"  Device ID: {device_id}")
    console.print(f"  Docker image: {effective_image}")
    console.print(f"  Families: {family_ids or 'all'}")
    console.print(f"  Modalities: {modality_ids or 'all'}")
    console.print(f"  Runtimes: {[r.value for r in runtime_ids] if runtime_ids else 'auto(all declared)'}")
    console.print(f"  Scope: {scope.lower()}")
    console.print()

    # Test SSH connection before profiling
    console.print("[dim]Testing SSH connection...[/dim]")
    is_connected, message = test_ssh_connection(device_ip, ssh_key)
    if not is_connected:
        console.print(f"[bold red]SSH connection failed: {message}[/bold red]")
        raise click.Abort()
    console.print("[dim]Connection successful.[/dim]\n")

    console.print("[dim]Checking edge runner status...[/dim]")
    is_runner_active, runner_message = ensure_edge_runner_active(device_ip, ssh_key)
    if not is_runner_active:
        console.print(f"[bold red]Edge runner check failed: {runner_message}[/bold red]")
        raise click.Abort()
    console.print("[dim]Edge runner is active.[/dim]\n")

    console.print("[dim]Docker image preflight on edge (local tag, arch, imports)...[/dim]")
    dp = run_docker_image_preflight(
        device_ip=device_ip,
        ssh_key=ssh_key,
        docker_image=effective_image,
        device_id=device_id,
        flavor="offline_profiler",
        profile_runtime_ids=runtime_ids,
        total_timeout_sec=20,
    )
    if not dp.ok:
        console.print(f"[bold red]Docker preflight failed:[/bold red] {dp.message}")
        raise click.Abort()
    console.print("[dim]Docker image preflight OK.[/dim]\n")

    profiler = OfflineProfiler()
    run_list, skipped = profiler.build_profile_plan(
        device_id=device_id,
        family_ids=family_ids,
        modalities=modality_ids,
        runtime_ids=runtime_ids,
        imgsz=imgsz,
    )
    original_run_count = len(run_list)
    scope = scope.lower()
    if scope == "failed":
        profile_store = get_profile_store()
        failed_only = []
        for req in run_list:
            req.environment_versions["docker_image"] = effective_image
            existing = profile_store.get_by_request(req)
            if existing is not None and existing.status != "success":
                failed_only.append(req)
        run_list = failed_only

    console.print(Rule("[bold]Profile Plan[/bold]", style="blue"))
    run_title = f"Will Run ({len(run_list)})"
    if scope == "failed":
        run_title = f"Will Run Failed-Only ({len(run_list)}/{original_run_count})"
    run_table = Table(title=run_title)
    run_table.add_column("Model", style="cyan")
    run_table.add_column("Runtime", style="green")
    display_run_list = sorted(
        run_list,
        key=lambda req: _model_display_sort_key(req.model_id, req.runtime_id.value),
    )
    for req in display_run_list:
        run_table.add_row(req.model_id, req.runtime_id.value)
    console.print(run_table)

    skip_table = Table(title=f"Skipped ({len(skipped)})")
    skip_table.add_column("Model", style="yellow")
    skip_table.add_column("Runtime", style="magenta")
    skip_table.add_column("Reason", style="dim")
    display_skipped = sorted(
        skipped,
        key=lambda item: _model_display_sort_key(item.get("model_id", "unknown"), item.get("runtime", "-")),
    )
    for item in display_skipped:
        skip_table.add_row(item["model_id"], item["runtime"], item["reason"])
    console.print(skip_table)
    console.print()

    # Coverage report by family/runtime for quick runtime blind-spot analysis.
    coverage = defaultdict(lambda: {"run": 0, "skip": 0, "reasons": Counter()})
    for req in run_list:
        fam = req.model_id.split(":", 1)[0]
        key = (fam, req.runtime_id.value)
        coverage[key]["run"] += 1
    for item in skipped:
        model_id = item.get("model_id", "")
        fam = model_id.split(":", 1)[0] if ":" in model_id else "unknown"
        runtime = item.get("runtime") or "-"
        key = (fam, runtime)
        coverage[key]["skip"] += 1
        coverage[key]["reasons"][item.get("reason", "unknown")] += 1

    cov_table = Table(title="Coverage by Family / Runtime")
    cov_table.add_column("Family", style="cyan")
    cov_table.add_column("Runtime", style="green")
    cov_table.add_column("Run", justify="right")
    cov_table.add_column("Skip", justify="right")
    cov_table.add_column("Coverage", justify="right")
    cov_table.add_column("Top Skip Reasons", style="dim")
    for (fam, runtime), stat in sorted(coverage.items(), key=lambda x: (x[0][0], x[0][1])):
        total = stat["run"] + stat["skip"]
        cov = f"{(100.0 * stat['run'] / total):.1f}%" if total else "0.0%"
        top_reasons = ", ".join(
            f"{reason}:{count}" for reason, count in stat["reasons"].most_common(3)
        ) or "-"
        cov_table.add_row(
            fam,
            runtime,
            str(stat["run"]),
            str(stat["skip"]),
            cov,
            top_reasons,
        )
    console.print(cov_table)
    console.print()

    if dry_run:
        console.print("[green]Dry run complete. No profiling executed.[/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Profiling models...", total=None)
        profile_store = get_profile_store()
        results = []
        for req in run_list:
            req.environment_versions["docker_image"] = effective_image
            per_timeout = timeout
            if req.runtime_id == RuntimeId.TENSORRT and timeout <= 600:
                per_timeout = 2700
            result = profiler.profile_remote(
                request=req,
                device_ip=device_ip,
                ssh_key=ssh_key,
                docker_image=effective_image,
                timeout=per_timeout,
            )
            results.append(result)
            profile_store.store(result)
        progress.update(task, completed=True)

    # Summary table
    console.print()
    console.print(Rule("[bold]Profile Results[/bold]", style="blue"))

    table = Table(title="Latency / Memory / Throughput")
    table.add_column("Model", style="cyan")
    table.add_column("Runtime", style="green")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Throughput (fps)", justify="right")
    table.add_column("Power (W)", justify="right")
    table.add_column("Status", style="dim")

    for r in results:
        mem = f"{r.gpu_memory_mb:.0f}" if r.gpu_memory_mb else f"{r.peak_memory_mb:.0f}"
        power = f"{r.power_w:.1f}" if r.power_w else "—"
        status = r.status if r.status != "success" else "✓"
        table.add_row(
            r.model_id,
            r.runtime_id.value,
            f"{r.latency_avg_ms:.1f}" if r.status == "success" else "—",
            mem if r.status == "success" else "—",
            f"{r.throughput_fps:.1f}" if r.status == "success" else "—",
            power,
            status,
        )

    console.print(table)
    console.print()
    success = sum(1 for r in results if r.status == "success")
    console.print(f"[green]{success}/{len(results)} profiles succeeded.[/green]")


@main.command()
def tools():
    """List available tools by modality."""
    from edgecraft.tools.base import ToolRegistry
    from edgecraft.core.modality import Modality

    table = Table(title="Available Tools")
    table.add_column("Modality", style="cyan")
    table.add_column("Tool", style="green")
    table.add_column("Description", style="white")

    for modality in [Modality.VISION, Modality.AUDIO, Modality.TEXT, Modality.MULTIMODAL]:
        tools = ToolRegistry.get_tools_for_modality(modality)
        for name, tool in tools.items():
            table.add_row(
                modality.value,
                name,
                tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
            )

    console.print(table)


@main.command()
@click.option('--host', default='127.0.0.1', show_default=True, help='Host to bind')
@click.option('--port', default=8000, help='Port to listen on')
def serve(host, port):
    """Start EdgeCraft API server."""
    from edgecraft.api.auth import parse_tenant_tokens
    from edgecraft.config.settings import settings as app_settings

    host_text = str(host or "").strip()
    try:
        loopback = ipaddress.ip_address(host_text).is_loopback
    except ValueError:
        loopback = host_text.lower() == "localhost"
    if not loopback and not parse_tenant_tokens(app_settings.API_TENANT_TOKENS):
        raise click.ClickException(
            "a non-loopback API bind requires EDGECRAFT_API_TENANT_TOKENS"
        )

    import uvicorn

    console.print("[bold green]Starting EdgeCraft API[/bold green]")
    console.print(f"[dim]Binding to http://{host_text}:{port}[/dim]")
    uvicorn.run(
        "edgecraft.api:app",
        host=host_text,
        port=int(port),
        reload=False,
        log_level="info",
    )


@main.group()
def task():
    """Submit and inspect serve-mode SaaS tasks."""
    pass


def _print_json(data):
    console.print(json.dumps(data, indent=2, ensure_ascii=False))


@task.command("submit")
@click.option("--server", default="http://localhost:8000", show_default=True, help="EdgeCraft serve URL.")
@click.option("--tenant-id", default="default", show_default=True, help="Tenant id.")
@click.option("--run-id", default=None, help="Optional run id.")
@click.option("--intent", required=True, help="Natural-language synthesis intent.")
@click.option("--dataset-path", required=True, help="Dataset path visible to the server.")
@click.option("--device-ip", required=True, help="Edge device SSH endpoint.")
@click.option(
    "--ssh-key-path",
    required=True,
    help="Explicit SSH key path visible to the server (used with IdentitiesOnly=yes).",
)
@click.option("--docker-image", required=False, default=None, help="Edge Docker image; omit for native-runtime devices.")
@click.option("--iterations", default=24, show_default=True, type=int, help="Maximum admitted trials.")
@click.option("--branching-factor", default=None, type=int, help="Children per expansion.")
@click.option("--api-token", envvar="EDGECRAFT_API_TOKEN", default=None, help="Bearer token for the tenant API.")
def task_submit(server, tenant_id, run_id, intent, dataset_path, device_ip, ssh_key_path, docker_image, iterations, branching_factor, api_token):
    """Submit a SaaS synthesis request to a running serve process."""
    from edgecraft.sdk.client import EdgeCraftClient

    client = EdgeCraftClient(base_url=server, api_key=api_token)
    task_obj = client.synthesize(
        intent=intent,
        dataset_path=dataset_path,
        device_ip=device_ip,
        ssh_key_path=ssh_key_path,
        docker_image=docker_image,
        max_iterations=iterations,
        branching_factor=branching_factor,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    _print_json(task_obj._data)


@task.command("status")
@click.argument("task_id")
@click.option("--server", default="http://localhost:8000", show_default=True, help="EdgeCraft serve URL.")
@click.option("--api-token", envvar="EDGECRAFT_API_TOKEN", default=None, help="Bearer token for the tenant API.")
def task_status(task_id, server, api_token):
    """Show one serve-mode task status."""
    from edgecraft.sdk.client import EdgeCraftClient

    _print_json(EdgeCraftClient(base_url=server, api_key=api_token).get_task(task_id)._data)


@task.command("tree")
@click.argument("task_id")
@click.option("--server", default="http://localhost:8000", show_default=True, help="EdgeCraft serve URL.")
@click.option("--api-token", envvar="EDGECRAFT_API_TOKEN", default=None, help="Bearer token for the tenant API.")
def task_tree(task_id, server, api_token):
    """Show one task's run/tree lifecycle."""
    from edgecraft.sdk.client import EdgeCraftClient

    _print_json(EdgeCraftClient(base_url=server, api_key=api_token).get_tree(task_id))


@task.command("artifacts")
@click.argument("task_id")
@click.option("--server", default="http://localhost:8000", show_default=True, help="EdgeCraft serve URL.")
@click.option("--api-token", envvar="EDGECRAFT_API_TOKEN", default=None, help="Bearer token for the tenant API.")
def task_artifacts(task_id, server, api_token):
    """Show one task's best-trial artifact manifest."""
    from edgecraft.sdk.client import EdgeCraftClient

    _print_json(EdgeCraftClient(base_url=server, api_key=api_token).get_artifacts(task_id))


@task.command("download")
@click.argument("task_id")
@click.argument("artifact_key")
@click.option("--server", default="http://localhost:8000", show_default=True, help="EdgeCraft serve URL.")
@click.option("--out", "output_path", required=True, help="Local output path.")
@click.option("--api-token", envvar="EDGECRAFT_API_TOKEN", default=None, help="Bearer token for the tenant API.")
def task_download(task_id, artifact_key, server, output_path, api_token):
    """Download one managed artifact by key."""
    from edgecraft.sdk.client import EdgeCraftClient

    path = EdgeCraftClient(base_url=server, api_key=api_token).download_artifact(task_id, artifact_key, output_path)
    console.print(f"[green]Downloaded[/green] {artifact_key} -> {path}")


# ============================================================================
# Knowledge Management Commands
# ============================================================================

@main.group()
def knowledge():
    """Manage the knowledge base (cases and RAG index).

    The knowledge base stores past edge AI cases for Case-Based Reasoning (CBR)
    and domain knowledge for Retrieval-Augmented Generation (RAG).

    Use 'edgecraft knowledge build' to populate the case store before running synth.
    """
    pass


@knowledge.command()
@click.option('--limit', '-n', default=50,
              help='Max cases per source')
def build(limit):
    """Build/update the case store by crawling external sources.

    Sources (Kaggle keywords and GitHub repos) are loaded from
    the repository crawl-source configuration — edit it to customise what
    gets crawled.  This command uses LLM to extract structured case
    information from each source.

    Examples:

        edgecraft knowledge build

        edgecraft knowledge build -n 100
    """
    from edgecraft.knowledge.cbr import CaseCrawler

    console.print("[bold]Building knowledge base...[/bold]")
    console.print()

    crawler = CaseCrawler()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Crawling sources...", total=None)

        try:
            results = crawler.build_case_store(
                limit_per_source=limit
            )
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise click.Abort()

    # Display results
    console.print()
    table = Table(title="Build Results")
    table.add_column("Source", style="cyan")
    table.add_column("Cases", style="green", justify="right")

    for src in ['kaggle', 'github', 'jetson']:
        if src in results:
            table.add_row(src.capitalize(), str(results[src]))

    table.add_row("[bold]Total Added[/bold]", f"[bold]{results.get('total_added', 0)}[/bold]")
    console.print(table)

    console.print()
    console.print("[green]Knowledge base build complete.[/green]")


@knowledge.command()
def status():
    """Show current knowledge base statistics.

    Displays the number of cases in the store, grouped by source.

    Example:

        edgecraft knowledge status
    """
    from edgecraft.knowledge.cbr import CaseStore

    store = CaseStore()
    total = store.count()
    by_source = store.count_by_source()

    console.print("[bold]Knowledge Base Status[/bold]")
    console.print()

    table = Table()
    table.add_column("Source", style="cyan")
    table.add_column("Cases", style="green", justify="right")

    for src, count in sorted(by_source.items()):
        table.add_row(src.capitalize(), str(count))

    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(table)

    if total == 0:
        console.print()
        console.print("[yellow]No cases in knowledge base. Run 'edgecraft knowledge build' to populate.[/yellow]")


@knowledge.command()
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def clear(confirm):
    """Clear all cases from the knowledge base.

    This permanently deletes all stored cases. Use with caution.

    Example:

        edgecraft knowledge clear --confirm
    """
    from edgecraft.knowledge.cbr import CaseStore

    if not confirm:
        if not click.confirm('This will delete all cases from the knowledge base. Continue?'):
            raise click.Abort()

    store = CaseStore()
    count = store.clear()

    console.print(f"[green]Cleared {count} cases from knowledge base.[/green]")


@knowledge.command()
@click.option('--source', '-s', type=click.Choice(['kaggle', 'github', 'jetson', 'edgecraft']),
              help='Filter by source')
@click.option('--limit', '-n', default=10, help='Number of cases to show')
def list(source, limit):
    """List cases in the knowledge base.

    Example:

        edgecraft knowledge list -n 20

        edgecraft knowledge list -s kaggle
    """
    from edgecraft.knowledge.cbr import CaseStore

    store = CaseStore()
    cases = store.list_cases(source=source)[:limit]

    if not cases:
        console.print("[yellow]No cases found.[/yellow]")
        return

    table = Table(title=f"Cases ({len(cases)} shown)")
    table.add_column("ID", style="dim", width=20)
    table.add_column("Source", style="cyan", width=8)
    table.add_column("Task", style="green", width=20)
    table.add_column("Model", style="yellow", width=15)
    table.add_column("Intent", width=40)

    for case in cases:
        task = f"{case.modality.value}/{case.task_type.value}"
        intent = case.intent[:37] + "..." if len(case.intent) > 40 else case.intent
        table.add_row(
            case.id,
            case.source,
            task,
            case.model_name or "-",
            intent
        )

    console.print(table)


if __name__ == '__main__':
    main()
