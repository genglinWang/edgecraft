"""Strict, failure-preserving resource accounting.

The accounting unit is a whole request. GPU allocation is charged for the
request wall clock, and stage costs are summed across every candidate,
including candidates that fail validation or execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("required token telemetry file is missing")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid token telemetry JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"token telemetry line {line_number} must be an object")
        rows.append(value)
    return rows


def summarize_token_telemetry(
    path: str | Path,
    *,
    expected_provider: str | None = None,
    expected_model: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate provider usage and mark failed requests without counts as incomplete."""
    telemetry_path = Path(path)
    rows = _rows(telemetry_path)
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") == "failed"]
    unknown_statuses = [row.get("status") for row in rows if row.get("status") not in {"completed", "failed"}]
    if unknown_statuses:
        raise ValueError(f"token telemetry has unknown status values: {unknown_statuses}")
    if not completed:
        raise ValueError("token telemetry has no completed provider response")
    required = (
        "provider_request_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "request_started_at",
        "response_received_at",
    )
    for index, row in enumerate(rows):
        if expected_provider is not None and row.get("provider") != expected_provider:
            raise ValueError(f"token telemetry provider mismatch at row {index}")
        if expected_model is not None and row.get("model") != expected_model:
            raise ValueError(f"token telemetry model mismatch at row {index}")
        if expected_run_id is not None and row.get("experiment_run_id") != expected_run_id:
            raise ValueError(f"token telemetry run ID mismatch at row {index}")
    for index, row in enumerate(completed):
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise ValueError(
                f"token telemetry incomplete at completed row {index}: {', '.join(missing)}"
            )
        values = [row[key] for key in ("input_tokens", "output_tokens", "total_tokens")]
        if not all(isinstance(value, int) and value >= 0 for value in values):
            raise ValueError(f"invalid token counts at completed row {index}")
        if values[0] + values[1] != values[2]:
            raise ValueError(f"inconsistent token total at completed row {index}")
    failed_with_counts = []
    unmetered_failed = []
    for index, row in enumerate(failed):
        values = [row.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
        if all(value is None for value in values):
            unmetered_failed.append(row)
            continue
        if not all(isinstance(value, int) and value >= 0 for value in values):
            raise ValueError(f"partial or invalid token counts at failed row {index}")
        if int(values[0]) + int(values[1]) != int(values[2]):
            raise ValueError(f"inconsistent token total at failed row {index}")
        failed_with_counts.append(row)
    accounted = [*completed, *failed_with_counts]
    complete = not unmetered_failed
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in accounted),
        "output_tokens": sum(int(row["output_tokens"]) for row in accounted),
        "total_tokens": sum(int(row["total_tokens"]) for row in accounted),
        "provider_request_count": len(rows),
        "completed_request_count": len(completed),
        "failed_request_count": len(failed),
        "unmetered_failed_request_count": len(unmetered_failed),
        "token_count_complete": complete,
        "reported_totals_are_lower_bound": not complete,
        "telemetry_sha256": sha256_file(telemetry_path),
    }


def summarize_candidate_llm_usage(rounds: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate candidate-level usage and fail closed on partial telemetry."""
    generated = [
        row
        for row in rounds
        if row.get("candidate_id") and row.get("root_reused") is not True
    ]
    usage_rows = [row.get("llm_usage") for row in generated]
    complete = bool(generated) and all(isinstance(row, Mapping) for row in usage_rows)
    if complete:
        for usage in usage_rows:
            assert isinstance(usage, Mapping)
            values = [usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
            complete = bool(
                usage.get("token_count_complete") is True
                and all(isinstance(value, int) and value >= 0 for value in values)
                and int(values[0]) + int(values[1]) == int(values[2])
            )
            if not complete:
                break
    if not complete:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "provider_request_count": sum(isinstance(row, Mapping) for row in usage_rows),
            "token_count_complete": False,
        }
    typed = [row for row in usage_rows if isinstance(row, Mapping)]
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in typed),
        "output_tokens": sum(int(row["output_tokens"]) for row in typed),
        "total_tokens": sum(int(row["total_tokens"]) for row in typed),
        "provider_request_count": sum(int(row.get("provider_request_count") or 1) for row in typed),
        "token_count_complete": True,
    }


def summarize_stage_resources(
    trials: Iterable[Mapping[str, Any]],
    *,
    request_wall_s: float,
    allocated_gpu_count: int = 1,
) -> dict[str, Any]:
    """Charge allocation wall time and sum all candidate stage observations."""
    if request_wall_s < 0 or allocated_gpu_count < 0:
        raise ValueError("wall time and allocated GPU count must be non-negative")
    device_s = 0.0
    preprocessing_s = 0.0
    train_s = 0.0
    export_s = 0.0
    compile_s = 0.0
    trial_count = 0
    for trial in trials:
        trial_count += 1
        stage_costs = ((trial.get("observations") or {}).get("stage_costs") or {})
        if not isinstance(stage_costs, Mapping):
            continue
        for name, raw in stage_costs.items():
            if not isinstance(raw, Mapping):
                continue
            elapsed = max(0.0, float(raw.get("elapsed_s") or 0.0))
            stage = str(name).lower()
            resource = str(raw.get("resource") or "").lower()
            if resource == "device" or (
                not resource
                and ("edge" in stage or "efficiency" in stage or "device" in stage)
            ):
                device_s += elapsed
            if "loader" in stage or "preprocess" in stage:
                preprocessing_s += elapsed
            # A combined full_train_export timer is dominated by and charged
            # as training; it cannot be relabelled wholesale as compilation.
            if "train_export" in stage or stage.startswith("train"):
                train_s += elapsed
            elif "compile" in stage:
                compile_s += elapsed
            elif "export" in stage:
                export_s += elapsed
    wall_s = float(request_wall_s)
    return {
        "wall_s": wall_s,
        "allocated_gpu_count": int(allocated_gpu_count),
        "gpu_s": wall_s * int(allocated_gpu_count),
        "gpu_accounting": "allocated_gpu_slots_x_request_wall_time",
        "device_s": device_s,
        "device_accounting": "sum_all_candidate_device_and_efficiency_stages",
        "train_s": train_s,
        "export_s": export_s,
        "compile_s": compile_s,
        "preprocessing_s": preprocessing_s,
        "accounted_trial_count": trial_count,
    }


def compact_candidate_failure(
    candidate: Mapping[str, Any],
    *,
    candidate_id: str,
    error: BaseException | str,
    active_slo: Mapping[str, Any] | None = None,
    completed_stage: str = "code_materialized",
) -> dict[str, Any]:
    """Keep a failed materialized candidate in result and cost accounting."""
    error_text = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    slo = dict(active_slo or {})
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
    return {
        "candidate_id": candidate_id,
        "parent_id": candidate.get("parent_id"),
        "completed_stage": completed_stage,
        "artifact_format": metadata.get("export_format") or None,
        "runtime": {"provider": None, "executed": False},
        "task_quality": {"metric": None, "value": None},
        "active_slo": slo,
        "measured_gap": {key: None for key in slo},
        "evaluation_valid": False,
        "slo_feasible": False,
        "runtime_or_contract_failure": error_text[:1000],
    }


def cost_metrics_complete(record: Mapping[str, Any]) -> bool:
    """Return whether a record has auditable token, GPU, and device fields."""
    usage = record.get("llm_usage")
    resources = record.get("resources")
    if not isinstance(usage, Mapping) or not isinstance(resources, Mapping):
        return False
    token_values = [usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
    resource_values = [resources.get(key) for key in ("wall_s", "gpu_s", "device_s")]
    return bool(
        usage.get("token_count_complete") is True
        and all(isinstance(value, int) and value >= 0 for value in token_values)
        and int(token_values[0]) + int(token_values[1]) == int(token_values[2])
        and all(isinstance(value, (int, float)) and value >= 0 for value in resource_values)
    )
