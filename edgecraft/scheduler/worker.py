"""Execution worker for scheduler-owned jobs.

This module deliberately knows nothing about search trees, scoring, or LLM
repair.  It runs a materialized trial workspace and returns scheduler evidence.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from edgecraft.scheduler.types import JobResult, JobSpec, JobStatus
from edgecraft.utils.subprocess_env import candidate_subprocess_env


_MODEL_ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".engine",
    ".onnx",
    ".plan",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".torchscript",
    ".ts",
}


def run_train_stage(spec: JobSpec, gpu_id: int) -> JobResult:
    """Run loader smoke + train.py inside the job workspace."""
    start = time.monotonic()
    result = JobResult(
        job_id=spec.job_id,
        status=JobStatus.SUCCESS,
        stage_reached="train",
        gpu_id=gpu_id,
        edge_device_id=spec.edge_device_id or "",
    )
    workspace = Path(spec.workspace_path or ".").resolve()
    train_script = Path(spec.train_script).resolve() if spec.train_script else workspace / "train.py"
    env = candidate_subprocess_env(
        workspace,
        {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )

    loader_script = workspace / "loader.py"
    if loader_script.exists():
        loader = _run_python(loader_script, ["--smoke"], workspace, env, min(spec.train_timeout_s, 300))
        result.stdout += loader[1]
        result.stderr += loader[2]
        if loader[0] != 0 or _json_status_error(loader[1]):
            result.status = JobStatus.DATA_PREP_FAILED
            result.error_stage = "loader"
            result.error = _error_from_output(loader[1], loader[2], "loader smoke failed")
            result.train_wall_s = time.monotonic() - start
            return result

    if not train_script.exists():
        result.status = JobStatus.TRAIN_FAILED
        result.error_stage = "train"
        result.error = f"train script not found: {train_script}"
        result.train_wall_s = time.monotonic() - start
        return result

    train = _run_python(train_script, [], workspace, env, spec.train_timeout_s)
    result.stdout += train[1]
    result.stderr += train[2]
    train_json = _last_json(train[1])
    if train[0] != 0 or _json_status_error(train[1]):
        result.status = JobStatus.TRAIN_FAILED
        result.error_stage = "train"
        result.error = _error_from_output(train[1], train[2], "train failed")
        result.train_wall_s = time.monotonic() - start
        return result

    result.local_metrics.update(_extract_metrics(train_json))
    artifact = _artifact_from_train_json(train_json, workspace) or _discover_artifact(workspace)
    if artifact is None:
        result.status = JobStatus.TRAIN_FAILED
        result.error_stage = "artifact"
        result.error = "train completed but no deployable artifact was found"
    else:
        result.artifact_paths.update(_declared_artifact_paths(train_json, workspace))
        result.artifact_paths["train"] = str(artifact)
        result.artifact_paths["artifact"] = str(artifact)
    result.train_wall_s = time.monotonic() - start
    return result


def run_edge_stage(
    spec: JobSpec,
    train_result: JobResult,
    artifact_manifest: Optional[Dict[str, Any]] = None,
) -> JobResult:
    """Run the edge benchmark for a successfully trained job."""
    start = time.monotonic()
    train_result.stage_reached = "edge"
    train_result.edge_device_id = spec.edge_device_id or ""
    artifact = train_result.artifact_paths.get("artifact") or train_result.artifact_paths.get("train")
    if not artifact:
        train_result.status = JobStatus.TRAIN_FAILED
        train_result.error_stage = "artifact"
        train_result.error = train_result.error or "missing artifact for edge benchmark"
        return train_result

    try:
        from edgecraft.tools.deploy.edge_runner import EdgeRunner
        from edgecraft.agent.search.verification import build_artifact_fingerprint

        fingerprint = build_artifact_fingerprint(artifact).model_dump(mode="json")

        edge_result = EdgeRunner().run(
            artifact_path=artifact,
            device_id=spec.edge_device_id,
            device_ip=spec.edge_device_ip,
            ssh_key=spec.edge_ssh_key or None,
            workspace_path=spec.workspace_path or None,
            infer_script=spec.infer_script or None,
            dataset_path=spec.dataset_path or None,
            timeout=spec.edge_timeout_s,
            docker_image=spec.edge_docker_image,
            export_format=spec.export_format or None,
            quant_mode=spec.quant_mode or None,
            graph_hash=str(fingerprint.get("graph_hash") or ""),
            require_exact_runtime=bool(spec.require_exact_runtime),
            artifact_manifest=artifact_manifest
            or _edge_artifact_manifest(train_result.artifact_paths, spec.workspace_path),
        )
    except Exception as exc:  # noqa: BLE001
        train_result.status = JobStatus.EDGE_FAILED
        train_result.error_stage = "edge"
        train_result.error = str(exc)
        train_result.edge_wall_s = time.monotonic() - start
        return train_result

    train_result.stdout += str(
        edge_result.get("stdout") or edge_result.get("stdout_tail") or ""
    )
    train_result.stderr += str(
        edge_result.get("stderr") or edge_result.get("stderr_tail") or ""
    )
    train_result.edge_metrics.update(_extract_edge_metrics(edge_result))
    train_result.diagnostics["edge_result"] = {
        key: edge_result.get(key)
        for key in (
            "status",
            "runtime_used",
            "runtime_provider",
            "artifact_used",
            "fallback_reason",
            "measurement_protocol",
            "requested_graph_hash",
            "source_artifact_hash",
            "require_exact_runtime",
            "requested_runtime",
            "artifact_loaded",
            "artifact_present",
            "evaluation_valid",
            "prediction_parity",
            "prediction_mismatch_count",
        )
        if edge_result.get(key) is not None
    }
    if edge_result.get("status") == "success":
        train_result.status = JobStatus.SUCCESS
    else:
        train_result.status = JobStatus.EDGE_FAILED
        train_result.error_stage = "edge"
        train_result.error = str(edge_result.get("error") or "edge benchmark failed")
    train_result.edge_wall_s = time.monotonic() - start
    return train_result


def _run_python(
    script: Path,
    args: list[str],
    cwd: Path,
    env: Dict[str, str],
    timeout_s: int,
) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script), *args],
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\ntimeout after {timeout_s}s"


def _last_json(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _json_status_error(stdout: str) -> bool:
    obj = _last_json(stdout)
    return str(obj.get("status", "")).lower() == "error"


def _error_from_output(stdout: str, stderr: str, default: str) -> str:
    obj = _last_json(stdout)
    return str(obj.get("error") or (stderr or stdout)[-1200:] or default)


def _extract_metrics(obj: Dict[str, Any]) -> Dict[str, float]:
    raw = obj.get("metrics", obj)
    metrics: Dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, (int, float)):
                metrics[str(key)] = float(value)
    return metrics


def _extract_edge_metrics(edge_result: Dict[str, Any]) -> Dict[str, float]:
    metrics = _extract_metrics(edge_result.get("metrics", {}))
    for key in ("latency_ms", "memory_mb", "energy_j"):
        value = edge_result.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _artifact_from_train_json(obj: Dict[str, Any], workspace: Path) -> Optional[Path]:
    for key in ("artifact_path", "model_path", "export_path", "output_path"):
        value = obj.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = workspace / path
        if path.exists():
            return path
    return None


def _declared_artifact_paths(obj: Dict[str, Any], workspace: Path) -> Dict[str, str]:
    """Return existing train-declared artifacts confined to outputs/."""
    declared = obj.get("artifact_paths")
    if not isinstance(declared, dict):
        return {}
    outputs = (workspace / "outputs").resolve()
    paths: Dict[str, str] = {}
    for key, raw in declared.items():
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = Path(raw)
        path = path if path.is_absolute() else workspace / path
        try:
            resolved = path.resolve()
            resolved.relative_to(outputs)
        except (OSError, ValueError):
            continue
        if resolved.is_file() or resolved.is_dir():
            paths[str(key)] = str(resolved)
    return paths


def _edge_artifact_manifest(
    artifact_paths: Dict[str, str],
    workspace: str | Path | None,
    *,
    required_keys: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Project explicit worker artifacts into EdgeRunner's existing contract.

    Model alternatives remain optional. The selected model and explicitly
    declared non-model payloads are required. ``required_keys`` preserves an
    archived ArtifactContract when available; ``None`` supports old snapshots.
    """
    root = Path(workspace or ".").resolve()
    outputs = (root / "outputs").resolve()
    primary_paths: set[Path] = set()
    for key in ("artifact", "train"):
        raw = artifact_paths.get(key)
        if raw:
            try:
                primary_paths.add(Path(raw).resolve())
            except OSError:
                pass

    artifacts: Dict[str, Dict[str, Any]] = {}
    for key, raw in artifact_paths.items():
        path = Path(str(raw))
        path = path if path.is_absolute() else root / path
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(outputs)
        except (OSError, ValueError):
            continue
        if not (resolved.is_file() or resolved.is_dir()):
            continue
        inferred_required = (
            resolved in primary_paths
            or resolved.is_dir()
            or resolved.suffix.lower() not in _MODEL_ARTIFACT_SUFFIXES
        )
        required = (
            resolved in primary_paths or str(key) in required_keys
            if required_keys is not None
            else inferred_required
        )
        artifacts[str(key)] = {
            "path": str(resolved),
            "required_on_edge": bool(required),
            "archive_relative_path": relative.as_posix(),
        }
    return {"artifacts": artifacts}


def _discover_artifact(workspace: Path) -> Optional[Path]:
    outputs = workspace / "outputs"
    for name in ("best.onnx", "best.pt", "best.tflite", "best.engine"):
        path = outputs / name
        if path.exists():
            return path
    if outputs.is_dir():
        for suffix in ("*.onnx", "*.pt", "*.tflite", "*.engine"):
            matches = sorted(outputs.glob(suffix), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]
    return None
