"""PipelineExecutor node: runs one SolutionVariant through the 3-stage pipeline.

Code-centric architecture stages:
1. DataPrep   – workspace creation, write train.py + infer.py + config/data.yaml
2. Train      – execute train.py via subprocess (train + eval + save model)
3. EdgeBench  – ship infer.py + model to edge device, run benchmark

Each stage writes outputs into TrialResult. A failed stage sets
trial.error + error_stage and returns early.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import yaml

from edgecraft.agent.contracts import (
    ArtifactRecord,
    CraftingProgress,
    RuntimeAttempt,
    artifact_kind,
)
from edgecraft.agent.failure_taxonomy import (
    classify_failure,
    classify_repairability,
    next_search_hint,
)
from edgecraft.agent.metrics import MetricRegistry
from edgecraft.agent.modality_handler import ModalityHandlerRegistry
from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.agent.search.component_boundary import declared_component_files
from edgecraft.agent.search.surrogate import SurrogatePredictor
from edgecraft.agent.search.trial_result import (
    DebugAttempt,
    EdgeMetrics,
    LocalMetrics,
    StageReached,
    TrainingTrace,
    TrialResult,
)
from edgecraft.agent.search.verification import (
    EFFICIENCY_MEASUREMENT_CONTRACT_VERSION,
    EFFICIENCY_PROBE_MIN_MEASURE_SECONDS,
    EFFICIENCY_PROBE_MIN_WARMUP_SECONDS,
    EFFICIENCY_PROBE_REPETITIONS,
    EFFICIENCY_PROBE_SESSIONS,
    EFFICIENCY_PROBE_WARMUP,
    Evidence,
    MetricEstimate,
    ResourceCost,
    VerifierPolicy,
    VerificationReport,
    build_artifact_fingerprint,
    build_environment_fingerprint,
    compare_artifact_fingerprints,
)
from edgecraft.agent.state import AgentState
from edgecraft.agent.workspace.manager import WorkspaceManager
from edgecraft.config.settings import edgecraft_env, settings
from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import UserSpec
from edgecraft.knowledge.calibration_store import (
    CalibrationPair,
    CalibrationStore,
    calibration_protocol_fingerprint,
)
from edgecraft.knowledge.compatibility.repro import write_runtime_repro_infer
from edgecraft.scheduler.estimator import WorkloadEstimator
from edgecraft.scheduler.types import JobResult, JobSpec, JobStatus, WorkloadEstimate
from edgecraft.utils.llm import create_chat_llm, llm_credentials_available
from edgecraft.utils.p1_guard import validate_p1_guard_attestation
from edgecraft.utils.trash import move_to_trash
from edgecraft.utils.subprocess_env import candidate_subprocess_env


# ---------------------------------------------------------------------------
# Script execution helper
# ---------------------------------------------------------------------------

P1_GUARD_RUNNER = (
    Path(__file__).resolve().parents[2] / "utils" / "p1_probe_runner.py"
)


def _config_covers_split_manifest(config: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
    """Return true when declarative split paths cover every manifest sample ID."""
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        return False
    for split_name, sample_ids in splits.items():
        configured = config.get(split_name)
        if not isinstance(configured, str) or not configured.strip():
            return False
        if not isinstance(sample_ids, list):
            return False
        prefix = configured.strip().replace("\\", "/").strip("/") + "/"
        if any(not str(sample_id).replace("\\", "/").lstrip("/").startswith(prefix) for sample_id in sample_ids):
            return False
    return True

def _run_script(
    script_path: Path,
    cwd: Path,
    timeout: int = 7200,
    stream_output: bool = True,
    script_args: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    p1_guarded: bool = False,
) -> Dict[str, Any]:
    """Execute a Python script and parse the last JSON line from stdout.

    Args:
        script_path: Path to the Python script.
        cwd: Working directory for the script.
        timeout: Maximum execution time in seconds.
        stream_output: If True, stream stdout/stderr in real-time to logger.
        p1_guarded: Execute the candidate through the controller-owned P1
            construction guard.

    Returns:
        Dict with at least "status" key. On success, includes "metrics".
        On failure, includes "error", "stdout", "stderr".
    """
    import threading

    def _is_noisy_ultralytics_progress(prefix: str, line: str) -> bool:
        """Filter high-frequency Ultralytics progress lines from live logs.

        Keep full stdout/stderr buffers intact for debugging and JSON parsing,
        but suppress repetitive progress-bar updates in console output.
        """
        if prefix != "train.py":
            return False
        lower = line.lower()
        if "traceback" in lower or "error" in lower or "warning" in lower:
            return False
        if "results saved to" in lower or "speed:" in lower or "validate:" in lower:
            return False
        if "━" in line or "────" in line:
            return True
        if re.search(r"\b\d+/\d+\b.*\bit/s\b", line):
            return True
        if re.search(r"\b\d+/\d+\s+\d+(?:\.\d+)?g\b", lower):
            return True
        return False

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    script_name = script_path.name

    def _stream_reader(pipe, lines_list: list, prefix: str, is_stderr: bool = False):
        """Read from pipe line-by-line, log in real-time, and collect output."""
        for line in iter(pipe.readline, ''):
            line = line.rstrip('\n\r')
            if line:
                lines_list.append(line)
                if stream_output:
                    if is_stderr:
                        logger.warning(f"[{prefix}] {line}")
                    else:
                        # Skip JSON output line from logging (it's the result)
                        if not (line.startswith('{') and line.endswith('}')):
                            if not _is_noisy_ultralytics_progress(prefix, line):
                                logger.info(f"[{prefix}] {line}")
        pipe.close()

    started = time.perf_counter()
    try:
        env = candidate_subprocess_env(cwd, env_overrides)
        assigned_gpu = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip() or None
        execution_seed: Optional[int] = None
        command = [sys.executable, "-u", str(script_path), *(script_args or [])]
        if script_name == "train.py":
            seed_text = str(env.get("EDGECRAFT_TRAIN_SEED") or "").strip()
            if p1_guarded:
                command = [
                    sys.executable,
                    "-u",
                    str(P1_GUARD_RUNNER),
                    "--script",
                    str(script_path),
                    "--",
                    *(script_args or []),
                ]
            elif seed_text:
                execution_seed = int(seed_text)
                env["PYTHONHASHSEED"] = seed_text
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "edgecraft.utils.seed_bootstrap",
                    str(script_path),
                    *(script_args or []),
                ]
            if assigned_gpu is not None:
                logger.info(f"[{script_name}] using externally assigned GPU: {assigned_gpu}")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            start_new_session=True,
        )

        # Start reader threads for real-time streaming
        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stdout, stdout_lines, script_name, False)
        )
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, stderr_lines, script_name, True)
        )
        stdout_thread.start()
        stderr_thread.start()

        peak_rss_mb = 0.0
        memory_limit_gb = 0.0
        if script_name == "train.py":
            try:
                memory_limit_gb = max(
                    0.0,
                    float(
                        env.get(
                            "EDGECRAFT_TRAIN_MEMORY_LIMIT_GB",
                            "0",
                        )
                        or 0
                    ),
                )
            except ValueError:
                memory_limit_gb = 0.0

        if memory_limit_gb > 0:
            import psutil

            memory_limit_bytes = int(memory_limit_gb * 1024**3)
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                try:
                    root_process = psutil.Process(process.pid)
                    rss_bytes = root_process.memory_info().rss + sum(
                        child.memory_info().rss for child in root_process.children(recursive=True)
                    )
                    peak_rss_mb = max(peak_rss_mb, rss_bytes / 1024**2)
                except (psutil.Error, OSError):
                    rss_bytes = 0
                if rss_bytes > memory_limit_bytes:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except Exception:  # noqa: BLE001
                        os.killpg(process.pid, signal.SIGKILL)
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)
                    return {
                        "status": "error",
                        "error": (
                            f"Training process tree exceeded memory limit "
                            f"{memory_limit_gb:g} GB (peak {peak_rss_mb:.1f} MB)"
                        ),
                        "error_code": "resource_memory_limit",
                        "stdout": "\n".join(stdout_lines),
                        "stderr": "\n".join(stderr_lines),
                        "assigned_gpu": assigned_gpu,
                        "execution_seed": execution_seed,
                        "timed_out": False,
                        "timeout_s": timeout,
                        "memory_limit_gb": memory_limit_gb,
                        "peak_rss_mb": peak_rss_mb,
                        "elapsed_s": time.perf_counter() - started,
                    }
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.5)

        # Wait for process with the remaining wall-clock budget.
        remaining = max(0.0, timeout - (time.perf_counter() - started))
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                process.kill()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            return {
                "status": "error",
                "error": f"Script timed out after {timeout}s",
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines),
                "assigned_gpu": assigned_gpu,
                "execution_seed": execution_seed,
                "timed_out": True,
                "timeout_s": timeout,
                "memory_limit_gb": memory_limit_gb,
                "peak_rss_mb": peak_rss_mb,
                "elapsed_s": time.perf_counter() - started,
            }

        # Wait for reader threads to finish
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "assigned_gpu": assigned_gpu if "assigned_gpu" in locals() else None,
            "execution_seed": execution_seed if "execution_seed" in locals() else None,
            "timed_out": False,
            "timeout_s": timeout,
            "elapsed_s": time.perf_counter() - started,
        }

    # Contract: last non-empty stdout line must be valid JSON.
    non_empty_stdout = [line for line in stdout_lines if line.strip()]
    if not non_empty_stdout:
        return {
            "status": "error",
            "error": f"JSON contract violation: stdout empty. Exit code: {process.returncode}",
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "assigned_gpu": assigned_gpu,
            "execution_seed": execution_seed,
            "timed_out": False,
            "timeout_s": timeout,
            "elapsed_s": time.perf_counter() - started,
        }
    parsed = None
    json_error = None
    last_line = non_empty_stdout[-1]
    try:
        parsed = json.loads(last_line)
    except json.JSONDecodeError as exc:
        json_error = exc
        joined = "\n".join(stdout_lines)
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(joined):
            if ch not in "[{":
                continue
            try:
                candidate, end = decoder.raw_decode(joined[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
        if parsed is None:
            return {
                "status": "error",
                "error": (
                    f"JSON contract violation: last stdout line is not JSON "
                    f"(exit={process.returncode}): {json_error}"
                ),
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines),
                "assigned_gpu": assigned_gpu,
                "execution_seed": execution_seed,
                "timed_out": False,
                "timeout_s": timeout,
                "elapsed_s": time.perf_counter() - started,
            }

    # Add stdout/stderr to result
    parsed["stdout"] = "\n".join(stdout_lines)
    parsed["stderr"] = "\n".join(stderr_lines)
    parsed["assigned_gpu"] = assigned_gpu
    parsed["execution_seed"] = execution_seed
    parsed["timed_out"] = False
    parsed["timeout_s"] = timeout
    parsed["memory_limit_gb"] = memory_limit_gb
    parsed["peak_rss_mb"] = peak_rss_mb
    parsed["elapsed_s"] = time.perf_counter() - started
    return parsed


def _confirmed_training_seed(result: Dict[str, Any]) -> Optional[int]:
    """Return the seed only when generated code confirms the bootstrap request."""
    requested = result.get("execution_seed")
    reported = result.get("training_seed")
    if requested is None or reported is None:
        return None
    try:
        return int(requested) if int(reported) == int(requested) else None
    except (TypeError, ValueError):
        return None


_REPAIRABLE_ERROR_PATTERNS = [
    ("invalid_argument", r"not a valid .* argument|unexpected keyword argument|got an unexpected keyword argument"),
    ("export_failure", r"export.*failed|onnx.*error|tensorrt.*error|engine.*error"),
    ("metric_key_error", r"results_dict|metrics/|keyerror|attributeerror"),
    ("contract_violation", r"json contract violation|export smoke failed|blocked_unstable_export_api"),
    ("dataset_loading_failure", r"an error occurred while generating the dataset|dataset.*(not found|missing|invalid)|audiofolder|metadata\.csv|torchcodec|no module named 'torchcodec"),
    ("device_mismatch", r"different from other tensors on (cpu|cuda)"),
]

_INFRA_ERROR_PATTERNS = [
    ("ssh_failure", r"connection refused|connection timed out|no route to host|network is unreachable"),
    ("ssh_auth", r"permission denied \(publickey|host key verification failed"),
    ("docker_not_found", r"docker:.*not found|cannot connect to the docker daemon"),
    ("device_offline", r"destination host unreachable|name or service not known"),
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash a declared artifact directory by relative path and file content."""
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _classify_error_signature(error: str, stderr: str) -> str:
    category = classify_failure(error, stderr)
    if category.value != "unknown":
        return category.value
    blob = f"{error}\n{stderr}".lower()
    for name, pattern in _INFRA_ERROR_PATTERNS:
        if re.search(pattern, blob):
            return f"infra_{name}"
    for name, pattern in _REPAIRABLE_ERROR_PATTERNS:
        if re.search(pattern, blob):
            return name
    return "unknown"


def _is_repairable_failure(error: str, stderr: str) -> tuple[bool, str]:
    repairable, category, _ = classify_repairability(error, stderr)
    if category != "unknown":
        return repairable, category
    sig = _classify_error_signature(error, stderr)
    if sig.startswith("infra_"):
        return False, sig
    return (sig != "unknown"), sig


def _collect_failure_diagnostics(
    *,
    ws: Path,
    state: AgentState,
    stage: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect small, read-only diagnostics before LLM repair or failure recording."""
    diag: Dict[str, Any] = {
        "stage": stage,
        "error": result.get("error"),
        "return_code": result.get("return_code"),
        "dataset_signature": {},
        "layout_contract": None,
        "outputs": {},
        "runtime": {},
        "artifact_selection": {},
    }
    dataset_info = state.get("dataset_info") or {}
    if isinstance(dataset_info, dict):
        split_manifest = dataset_info.get("split_manifest")
        sample_observation = _compact_sample_observation(
            dataset_info.get("sample_observation")
        )
        diag["dataset_info"] = {
            "dataset_path": dataset_info.get("dataset_path"),
            "config_path": dataset_info.get("config_path"),
            "modality": dataset_info.get("modality"),
            "task_type": dataset_info.get("task_type"),
            "format": dataset_info.get("format"),
            "description": str(dataset_info.get("description") or "")[:1200],
            "structure_summary": str(dataset_info.get("structure_summary") or "")[:1500],
            "recommended_config": dataset_info.get("recommended_config"),
            "exploration_report_excerpt": str(dataset_info.get("exploration_report") or "")[:800],
            "sample_observation": sample_observation,
        }
        if sample_observation:
            # Loader failures happen before a workspace observation exists. Keep
            # the analyzer evidence on the same path consumed by repair logic.
            diag["sample_observation"] = sample_observation
        if isinstance(split_manifest, dict):
            diag["split_manifest"] = {
                "path": split_manifest.get("path"),
                "content_hash": split_manifest.get("content_hash"),
                "schema": split_manifest.get("schema"),
            }
            try:
                manifest_path = Path(str(split_manifest.get("path") or ""))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                splits = manifest.get("splits") if isinstance(manifest, dict) else None
                if isinstance(splits, dict):
                    diag["split_manifest"]["actual_splits"] = {
                        str(name): list(values[:8])
                        for name, values in splits.items()
                        if isinstance(values, list)
                    }
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                diag["split_manifest"]["read_error"] = str(exc)
    cfg_path = ws / "config" / "data.yaml"
    if cfg_path.exists():
        try:
            import yaml as _yaml

            cfg_text = cfg_path.read_text(encoding="utf-8")
            diag["config_data_yaml_excerpt"] = cfg_text[:3000]
            cfg = _yaml.safe_load(cfg_text) or {}
            if isinstance(cfg, dict):
                diag["layout_contract"] = cfg.get("layout_contract") or cfg.get("task") or cfg.get("format")
                diag["dataset_signature"] = {
                    "task": cfg.get("task"),
                    "format": cfg.get("format"),
                    "modality": cfg.get("modality"),
                    "task_type": cfg.get("task_type"),
                    "file_format": cfg.get("file_format"),
                    "schema_columns": (cfg.get("schema") or {}).get("columns", [])[:30] if isinstance(cfg.get("schema"), dict) else [],
                    "label_column": cfg.get("label_column") or (cfg.get("schema") or {}).get("label_column") if isinstance(cfg.get("schema"), dict) else cfg.get("label_column"),
                    "label_join_key": cfg.get("label_join_key"),
                }
        except Exception as exc:  # noqa: BLE001
            diag["config_error"] = str(exc)
    outputs = ws / "outputs"
    if outputs.exists():
        try:
            diag["outputs"] = {
                str(p.relative_to(ws)): p.stat().st_size
                for p in sorted(outputs.glob("*"))
                if p.is_file()
            }
        except Exception as exc:  # noqa: BLE001
            diag["outputs_error"] = str(exc)
        sample_obs_path = outputs / "sample_observation.json"
        if sample_obs_path.exists():
            try:
                diag["sample_observation"] = json.loads(sample_obs_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                diag["sample_observation_error"] = str(exc)
    current_trial = state.get("current_trial_result")
    if current_trial and getattr(current_trial, "artifact_contract", None):
        try:
            contract = current_trial.artifact_contract
            diag["artifact_selection"] = {
                "primary_artifact": contract.primary_artifact,
                "fallback_artifacts": list(contract.fallback_artifacts),
                "artifact_roles": {
                    name: {
                        "kind": rec.kind,
                        "role": rec.role,
                        "size_bytes": rec.size_bytes,
                    }
                    for name, rec in contract.artifacts.items()
                },
            }
        except Exception as exc:  # noqa: BLE001
            diag["artifact_selection_error"] = str(exc)
    runtime_config = state.get("runtime_config")
    if runtime_config:
        try:
            diag["runtime"] = runtime_config.model_dump(mode="json")
        except Exception:
            diag["runtime"] = {"repr": repr(runtime_config)}
    return diag


def _record_trial_observations(
    trial: TrialResult,
    *,
    loader_smoke: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    """Store factual trial observations in one stable place."""
    observations: Dict[str, Any] = dict(getattr(trial, "observations", {}) or {})
    sample_observation: Optional[Dict[str, Any]] = None
    if loader_smoke:
        observations["loader_smoke"] = loader_smoke
        sample_observation = loader_smoke.get("sample_observation")
        probe = loader_smoke.get("loader_return_probe")
        if not sample_observation and isinstance(probe, dict):
            sample_observation = probe.get("sample_observation")
    if diagnostics:
        observations["diagnostics"] = diagnostics
        if not sample_observation:
            sample_observation = diagnostics.get("sample_observation")
    if sample_observation:
        observations["sample_observation"] = sample_observation
    if getattr(trial, "artifact_contract", None):
        observations["artifact_manifest"] = trial.artifact_contract.model_dump(mode="json")
    if getattr(trial, "runtime_report", None):
        observations["runtime_report"] = trial.runtime_report.model_dump(mode="json")
    trial.observations = observations


def _workspace_hashes(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    hashes: Dict[str, str] = {}
    for name in ("loader.py", "train.py", "infer.py", "config/data.yaml"):
        file_path = path / name
        if file_path.exists() and file_path.is_file():
            try:
                hashes[name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
            except Exception:
                continue
    return hashes


def _reuse_parent_training_result(
    *,
    trial: TrialResult,
    variant: SolutionVariant,
    ws: Path,
    state: AgentState,
) -> Optional[Dict[str, Any]]:
    """Reuse a parent's measured training output when its training inputs are identical."""
    parent_id = str(getattr(variant, "parent_trial_id", "") or "")
    trial_bank = state.get("trial_bank")
    parent = trial_bank.get(parent_id) if parent_id and trial_bank is not None else None
    if parent is None or parent.local_metrics is None or not parent.workspace_path:
        return None

    parent_ws = Path(parent.workspace_path)
    child_hashes = _workspace_hashes(ws)
    parent_hashes = _workspace_hashes(parent_ws)
    for name in ("train.py", "loader.py", "config/data.yaml"):
        if child_hashes.get(name) != parent_hashes.get(name):
            return None

    parent_outputs = (parent_ws / "outputs").resolve()
    child_outputs = (ws / "outputs").resolve()
    primary_raw = parent.artifact_contract.primary_artifact or parent.artifact_paths.get("train")
    if not primary_raw:
        return None
    primary = Path(primary_raw).resolve()
    try:
        primary.relative_to(parent_outputs)
    except ValueError:
        return None
    if not primary.is_file():
        return None

    sources: Dict[Path, Path] = {primary.relative_to(parent_outputs): primary}
    for record in parent.artifact_contract.artifacts.values():
        source = Path(record.path).resolve()
        try:
            relative = source.relative_to(parent_outputs)
        except ValueError:
            continue
        if (source.is_file() or source.is_dir()) and record.kind != "diagnostic":
            sources[relative] = source

    copied: Dict[str, str] = {}
    for relative, source in sources.items():
        destination = child_outputs / relative
        if source.is_dir():
            if destination.exists():
                if (
                    not destination.is_dir()
                    or _sha256_directory(destination) != _sha256_directory(source)
                ):
                    return None
            else:
                shutil.copytree(source, destination)
        else:
            if destination.exists():
                if (
                    not destination.is_file()
                    or _sha256_file(destination) != _sha256_file(source)
                ):
                    return None
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        copied[relative.as_posix()] = str(destination)

    child_primary = child_outputs / primary.name
    if not child_primary.is_file():
        return None
    declared_copies: Dict[str, str] = {}
    provenance_prefix = "train_result.artifact_paths:"
    for record in parent.artifact_contract.artifacts.values():
        provenance = str(record.provenance or "")
        if not provenance.startswith(provenance_prefix):
            continue
        declared_key = provenance[len(provenance_prefix):]
        source = Path(record.path).resolve()
        try:
            relative = source.relative_to(parent_outputs)
        except ValueError:
            continue
        destination = child_outputs / relative
        if declared_key and (destination.is_file() or destination.is_dir()):
            declared_copies[declared_key] = str(destination)
    for key, raw_path in parent.artifact_paths.items():
        source = Path(raw_path).resolve()
        try:
            relative = source.relative_to(parent_outputs)
        except ValueError:
            continue
        destination = child_outputs / relative
        if destination.is_file() or destination.is_dir():
            declared_copies.setdefault(str(key), str(destination))
    if not declared_copies:
        declared_copies = dict(copied)
    declared_copies.setdefault("train", str(child_primary))
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["inherited_training"] = {
        "parent_trial_id": parent.trial_id,
        "training_input_hashes": {
            name: child_hashes.get(name)
            for name in ("train.py", "loader.py", "config/data.yaml")
            if child_hashes.get(name)
        },
        "copied_artifacts": sorted(copied),
        "declared_artifact_keys": sorted(declared_copies),
        "gpu_s": 0.0,
        "decision_authority": "exact_lineage_reuse",
    }
    trial.observations = observations
    return {
        "status": "success",
        "model_path": str(child_primary),
        "artifact_paths": declared_copies,
        "metrics": dict(parent.local_metrics.all_metrics),
        "training_trace": (
            parent.training_trace.model_dump(mode="json")
            if parent.training_trace is not None
            else None
        ),
        "execution_seed": parent.training_seed,
        "training_seed": parent.training_seed,
        "pretrained_source": (parent.observations or {}).get("pretrained_source"),
        "reused_parent_trial_id": parent.trial_id,
        "elapsed_s": 0.0,
        "stdout": "",
        "stderr": "",
    }


def _record_constraint_adherence(trial: TrialResult, ws: Path, state: AgentState) -> None:
    """Record coarse proposal adherence evidence without gating execution."""
    variant = getattr(trial, "variant", None)
    if variant is None:
        return
    changed = [str(x) for x in (getattr(variant, "changed_components", []) or []) if str(x).strip()]
    preserved = [str(x) for x in (getattr(variant, "inherited_components", []) or []) if str(x).strip()]
    hashes = _workspace_hashes(ws)
    files = list(hashes)
    parent_trial = None
    parent_id = str(getattr(variant, "parent_trial_id", "") or "")
    trial_bank = state.get("trial_bank")
    if parent_id and trial_bank is not None:
        parent_trial = trial_bank.get(parent_id)
    parent_path = Path(parent_trial.workspace_path) if parent_trial and parent_trial.workspace_path else None
    parent_hashes = _workspace_hashes(parent_path)
    actual_changed = sorted(
        name
        for name in set(hashes) | set(parent_hashes)
        if hashes.get(name) != parent_hashes.get(name)
    )
    broken: List[str] = []
    for component in preserved:
        names = declared_component_files([component])
        if names and any(
            name not in hashes or (parent_hashes and hashes.get(name) != parent_hashes.get(name))
            for name in names
        ):
            broken.append(component)
    declared_files = declared_component_files(changed)
    actual_changed_set = set(actual_changed)
    missing_declared = sorted(declared_files - actual_changed_set)
    undeclared_changed = sorted(actual_changed_set - declared_files)
    changed_match = bool(
        parent_hashes
        and declared_files
        and not missing_declared
        and not undeclared_changed
    )
    visible_evidence_ids = set()
    if trial_bank is not None:
        for prior_trial in trial_bank.get_all():
            visible_evidence_ids.update(
                item.id
                for item in list(
                    getattr(getattr(prior_trial, "verification_report", None), "evidence", []) or []
                )
            )
    evidence_refs = [str(item) for item in (getattr(variant, "evidence_refs", []) or [])]
    evidence_refs_valid = bool(evidence_refs) and set(evidence_refs).issubset(visible_evidence_ids)
    audit_ready = bool(parent_hashes and evidence_refs)
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["constraint_adherence"] = {
        "declared_changed_component": changed[0] if len(changed) == 1 else "",
        "changed_components": changed,
        "coherent_boundary_count": len(changed),
        "declared_changed_files": sorted(declared_files),
        "actual_changed_files": actual_changed,
        "missing_declared_changed_files": missing_declared,
        "undeclared_changed_files": undeclared_changed,
        "declared_change_observed": bool(parent_hashes and declared_files and not missing_declared),
        "preserved_components_claimed": preserved,
        "preserved_components_broken": broken,
        "evidence_refs": evidence_refs,
        "evidence_refs_valid": evidence_refs_valid,
        "audit_ready": audit_ready,
        "adherence_pass": bool(
            audit_ready
            and len(changed) == 1
            and changed_match
            and not broken
            and evidence_refs_valid
        ),
        "file_hashes": hashes,
        "parent_file_hashes": parent_hashes,
    }
    trial.observations = observations


def _compat_rules_mode() -> str:
    return str(settings.COMPAT_RULES or "off").strip().lower()


def _multifidelity_mode() -> str:
    return str(settings.VERIFIER_MODE or "full").strip().lower()


def _evidence_ladder_enabled() -> bool:
    return _multifidelity_mode() in {"ladder", "evidence_ladder"}


def _manifest_calibration_snapshot(
    path: str,
    *,
    environment_fingerprint: str,
    protocol_fingerprint: str,
    graph_hash: str,
) -> Dict[str, Any]:
    """Read only exact-context, pair-backed calibration from a JSON snapshot."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration manifest must be a JSON object")

    errors: Dict[str, float] = {}
    pair_ids: List[str] = []
    for item in payload.get("pairs", []):
        if not isinstance(item, dict):
            continue
        if (
            item.get("environment_fingerprint") != environment_fingerprint
            or item.get("protocol_fingerprint") != protocol_fingerprint
            or item.get("graph_hash") != graph_hash
            or not item.get("pair_id")
        ):
            continue
        quantity = item.get("quantity")
        error = item.get("absolute_error")
        if not quantity or error is None:
            continue
        canonical = MetricRegistry.canonicalize_name(str(quantity))
        errors[canonical] = max(errors.get(canonical, 0.0), abs(float(error)))
        pair_ids.append(str(item["pair_id"]))

    # Aggregated records are accepted only when they retain the exact context
    # and the IDs of the measured pairs behind every error bound.
    for item in payload.get("records", []):
        if not isinstance(item, dict):
            continue
        ids = [str(value) for value in (item.get("pair_ids") or []) if value]
        if (
            item.get("environment_fingerprint") != environment_fingerprint
            or item.get("protocol_fingerprint") != protocol_fingerprint
            or item.get("graph_hash") != graph_hash
            or not ids
        ):
            continue
        for quantity, error in (item.get("errors") or {}).items():
            if error is None:
                continue
            canonical = MetricRegistry.canonicalize_name(str(quantity))
            errors[canonical] = max(errors.get(canonical, 0.0), abs(float(error)))
        pair_ids.extend(ids)

    pair_ids = sorted(set(pair_ids))
    snapshot_payload = {
        "environment": environment_fingerprint,
        "protocol": protocol_fingerprint,
        "graph": graph_hash,
        "pairs": pair_ids,
    }
    snapshot_id = "cs_" + hashlib.sha256(
        json.dumps(snapshot_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "snapshot_id": snapshot_id,
        "environment_fingerprint": environment_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "graph_hash": graph_hash,
        "pair_ids": pair_ids,
        "pair_count": len(pair_ids),
        "errors": errors if pair_ids else {},
    }


def _calibration_snapshot(
    *,
    environment_fingerprint: str,
    protocol_fingerprint: str,
    graph_hash: str = "",
) -> Dict[str, Any]:
    if not environment_fingerprint or not protocol_fingerprint or not graph_hash:
        return {"snapshot_id": "incomplete_context", "pair_ids": [], "errors": {}}

    primary = str(getattr(settings, "VERIFIER_CALIBRATION_PATH", "") or "").strip()
    seed = str(getattr(settings, "VERIFIER_CALIBRATION_SEED_PATH", "") or "").strip()
    for path in dict.fromkeys(value for value in (primary, seed) if value):
        try:
            if Path(path).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
                snapshot = CalibrationStore(path).snapshot(
                    environment_fingerprint=environment_fingerprint,
                    protocol_fingerprint=protocol_fingerprint,
                    graph_hash=graph_hash,
                ).model_dump(mode="json")
            else:
                snapshot = _manifest_calibration_snapshot(
                    path,
                    environment_fingerprint=environment_fingerprint,
                    protocol_fingerprint=protocol_fingerprint,
                    graph_hash=graph_hash,
                )
            if snapshot.get("pair_ids") and snapshot.get("errors"):
                return snapshot
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Ignoring unusable verifier calibration source {path}: {exc}")

    # Legacy scalar errors remain accepted as diagnostic configuration by old
    # callers, but deliberately never enter a decision-authoritative snapshot.
    if str(getattr(settings, "VERIFIER_CALIBRATION_ERROR", "") or "").strip():
        logger.warning(
            "Ignoring context-free verifier calibration error for P1 decisions; "
            "provide an exact graph/environment/protocol pair ledger instead."
        )
    return {"snapshot_id": "cold_start", "pair_ids": [], "errors": {}}


def _l1_measurement_protocol(
    *,
    edge_result: Dict[str, Any],
    raw_metrics: Dict[str, Any],
    runtime_used: str,
    runtime_provider: Optional[str],
    precision: Optional[str],
    input_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Describe the cheap measurement with the same facts used by qualification."""
    measured_protocol = (
        edge_result.get("measurement_protocol")
        if isinstance(edge_result.get("measurement_protocol"), dict)
        else {}
    )
    return {
        "runtime": _runtime_identity(runtime_used),
        "runtime_provider": runtime_provider,
        "precision": precision,
        "warmup": measured_protocol.get("warmup", raw_metrics.get("warmup")),
        "warmup_unit": measured_protocol.get(
            "warmup_unit", raw_metrics.get("warmup_unit")
        ),
        "min_warmup_seconds": measured_protocol.get("min_warmup_seconds"),
        "actual_warmup": measured_protocol.get("actual_warmup"),
        "actual_warmup_seconds": measured_protocol.get("actual_warmup_seconds"),
        "repetitions": measured_protocol.get(
            "repetitions",
            raw_metrics.get("iterations") or raw_metrics.get("repetitions"),
        ),
        "actual_repetitions": measured_protocol.get("actual_repetitions"),
        "measurement_sessions": measured_protocol.get("measurement_sessions"),
        "latency_statistic": measured_protocol.get("latency_statistic"),
        "latency_uncertainty": measured_protocol.get("latency_uncertainty"),
        "measurement_policy": measured_protocol.get("measurement_policy"),
        "min_measure_seconds": measured_protocol.get("min_measure_seconds"),
        "benchmark_scope": measured_protocol.get("benchmark_scope"),
        "energy_source": measured_protocol.get("energy_source"),
        "energy_scope": measured_protocol.get("energy_scope"),
        "energy_trusted": measured_protocol.get("energy_trusted", False),
        "dvfs_state": measured_protocol.get(
            "dvfs_state", raw_metrics.get("dvfs_state")
        ),
        "thermal_state": measured_protocol.get(
            "thermal_state", raw_metrics.get("thermal_state")
        ),
        "measurement_contract_version": EFFICIENCY_MEASUREMENT_CONTRACT_VERSION,
        "input_profile": measured_protocol.get("input_specs") or input_specs,
        "congruence_status": "pending",
        "trt_timing_cache_id": raw_metrics.get("trt_timing_cache_id"),
        "trt_timing_cache_hit": raw_metrics.get("trt_timing_cache_hit"),
    }


def _append_online_calibration_pairs(trial: TrialResult) -> List[str]:
    """Append congruent cheap/full pairs after the full result exists."""
    path = str(getattr(settings, "VERIFIER_CALIBRATION_PATH", "") or "").strip()
    if not path or Path(path).suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        return []
    audit = (getattr(trial, "observations", {}) or {}).get("l1_l2_congruence") or {}
    if not audit.get("checked") or not audit.get("match"):
        return []
    report = getattr(trial, "verification_report", None)
    evidence = list(getattr(report, "evidence", []) or [])
    cheap = {
        (item.environment_fingerprint, MetricRegistry.canonicalize_name(item.quantity)): item
        for item in evidence
        if item.probe_id == "efficiency"
        and item.outcome == "pass"
        and item.value is not None
        and bool((item.artifact_fingerprint or {}).get("graph_hash"))
    }
    if not cheap:
        return []
    try:
        store = CalibrationStore(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not open verifier calibration store {path}: {exc}")
        return []
    pair_ids: List[str] = []
    for item in evidence:
        key = (item.environment_fingerprint, MetricRegistry.canonicalize_name(item.quantity))
        prior = cheap.get(key)
        if (
            item.probe_id != "full"
            or item.outcome != "pass"
            or item.fidelity != "measured_congruent"
            or item.protocol.get("decision_authority") != "p2_accept"
            or item.value is None
            or prior is None
            or item.protocol.get("efficiency_source") != "evaluator_runtime_replay"
        ):
            continue
        protocol_fingerprint = str(prior.protocol.get("calibration_protocol_fingerprint") or "")
        if not protocol_fingerprint:
            continue
        graph_hash = str((prior.artifact_fingerprint or {}).get("graph_hash") or "")
        try:
            pair_ids.append(store.append(
                CalibrationPair(
                    environment_fingerprint=item.environment_fingerprint,
                    protocol_fingerprint=protocol_fingerprint,
                    quantity=MetricRegistry.canonicalize_name(item.quantity),
                    cheap_value=float(prior.value),
                    full_value=float(item.value),
                    absolute_error=abs(float(item.value) - float(prior.value)),
                    cheap_sigma=prior.sigma,
                    full_sigma=item.sigma,
                    graph_hash=graph_hash,
                    cheap_evidence_id=prior.id,
                    full_evidence_id=item.id,
                    cheap_protocol=prior.protocol,
                    full_protocol=item.protocol,
                )
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not append verifier calibration pair: {exc}")
    if pair_ids:
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["calibration_update"] = {
            "pair_ids": pair_ids,
            "available_to_current_decision": False,
        }
        trial.observations = observations
    return pair_ids


def _runtime_environment_fingerprint(
    state: AgentState,
    *,
    runtime: str,
    precision: Optional[str] = None,
) -> str:
    """Identify the runtime environment that produced measured evidence."""
    return build_environment_fingerprint(
        device_id=str(state.get("target_device", "")),
        runtime=runtime,
        runtime_config=state.get("runtime_config"),
        protocol={"precision": precision},
    )


def _ensure_verification_report(trial: TrialResult) -> VerificationReport:
    report = trial.verification_report
    if report is None:
        report = VerificationReport(
            level="L0",
            status="unknown",
            evidence_source="",
            decision="none",
            source="evidence_verifier",
        )
        trial.verification_report = report
    return report


def _append_evidence(trial: TrialResult, *items: Evidence) -> VerificationReport:
    report = _ensure_verification_report(trial)
    existing = {item.id for item in report.evidence}
    report.evidence.extend(item for item in items if item.id not in existing)
    return report


def _record_stage_execution_evidence(
    trial: TrialResult,
    *,
    stage: str,
    result: Dict[str, Any],
    resource: str = "",
) -> Evidence:
    """Record one process boundary without interpreting why it passed or failed."""
    elapsed = max(0.0, float(result.get("elapsed_s") or 0.0))
    cost = ResourceCost(
        gpu_s=elapsed if resource == "gpu" else 0.0,
        device_s=elapsed if resource == "device" else 0.0,
    )
    item = Evidence(
        probe_id=stage,
        quantity="stage_execution",
        fidelity="proxy",
        outcome=(
            "unknown"
            if result.get("timed_out")
            else "pass" if result.get("status") == "success" else "fail"
        ),
        value=elapsed,
        unit="s",
        protocol={
            "stage": stage,
            "status": result.get("status"),
            "timed_out": bool(result.get("timed_out")),
            "timeout_s": result.get("timeout_s"),
            "error": str(result.get("error") or "")[-500:],
            "requested_training_seed": result.get("execution_seed"),
            "reported_training_seed": result.get("training_seed"),
            "seed_confirmed": _confirmed_training_seed(result) is not None,
            "memory_limit_gb": result.get("memory_limit_gb"),
            "peak_rss_mb": result.get("peak_rss_mb"),
            "decision_authority": "evidence_only",
        },
        resource_cost=cost,
    )
    _append_evidence(trial, item)
    observations = dict(getattr(trial, "observations", {}) or {})
    stage_costs = dict(observations.get("stage_costs") or {})
    stage_costs[stage] = {
        "elapsed_s": elapsed,
        "resource": resource or "none",
        "timed_out": bool(result.get("timed_out")),
        "timeout_s": result.get("timeout_s"),
        "status": result.get("status"),
        "evidence_id": item.id,
    }
    observations["stage_costs"] = stage_costs
    trial.observations = observations
    return item


def _run_host_infer_contract_probe(
    *,
    trial: TrialResult,
    ws: Path,
    artifact_path: str,
) -> Optional[Evidence]:
    """Exercise generated infer.py against the real produced artifact on the host.

    This is interface evidence only. A host/runtime mismatch cannot kill a branch.
    """
    infer_script = ws / "infer.py"
    if not artifact_path or not infer_script.exists():
        return None
    runtime = artifact_kind(artifact_path) or Path(artifact_path).suffix.lstrip(".")
    fingerprint = build_artifact_fingerprint(artifact_path).model_dump(mode="json")
    strict_dataset_eval = bool(
        getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False)
        and _declared_artifact_path(trial, "edge_eval_manifest")
    )
    env_overrides = {
        "EDGE_EVAL_DATASET": "1" if strict_dataset_eval else "0",
        "EDGECRAFT_ARTIFACT_PATH": str(Path(artifact_path).resolve()),
        "EDGECRAFT_RUNTIME": str(runtime),
    }
    if strict_dataset_eval:
        env_overrides["EDGE_EVAL_MANIFEST"] = str(
            _declared_artifact_path(trial, "edge_eval_manifest")
        )
    result = _run_script(
        infer_script,
        ws,
        timeout=int(getattr(settings, "HOST_INFER_TIMEOUT_S", 180)),
        env_overrides=env_overrides,
    )
    host_metrics = MetricRegistry.normalize_metric_dict(result.get("metrics") or {})
    execution = _record_stage_execution_evidence(
        trial,
        stage="component_host_infer",
        result=result,
    )
    contract = Evidence(
        probe_id="component_roundtrip",
        quantity="component_contract",
        fidelity="proxy",
        outcome=(
            "pass"
            if result.get("status") == "success"
            else "fail" if strict_dataset_eval else "unknown"
        ),
        artifact_fingerprint=fingerprint,
        protocol={
            "boundary": "artifact_to_infer",
            "real_artifact": True,
            "dataset_eval": strict_dataset_eval,
            "metrics": host_metrics,
            "execution_evidence_id": execution.id,
            "error": str(result.get("error") or "")[-800:],
            "stderr_tail": str(result.get("stderr") or "")[-800:],
            "decision_authority": "prompt_only",
        },
    )
    _append_evidence(trial, contract)
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["artifact_input_contract"] = {
        "artifact_format": fingerprint.get("artifact_format"),
        "graph_hash": fingerprint.get("graph_hash"),
        "input_specs": fingerprint.get("input_specs") or [],
        "output_specs": fingerprint.get("output_specs") or [],
        "input_shapes": fingerprint.get("input_shapes") or [],
        "evidence_id": contract.id,
    }
    observations["component_roundtrip"] = {
        "status": result.get("status"),
        "dataset_eval": strict_dataset_eval,
        "metrics": host_metrics,
        "evidence_id": contract.id,
        "execution_evidence_id": execution.id,
        "error": result.get("error"),
        "stdout_tail": str(result.get("stdout") or "")[-800:],
        "stderr_tail": str(result.get("stderr") or "")[-800:],
    }
    trial.observations = observations
    return contract


def _declared_artifact_path(trial: TrialResult, declared_key: str) -> Optional[Path]:
    provenance = f"train_result.artifact_paths:{declared_key}"
    for record in trial.artifact_contract.artifacts.values():
        if record.provenance == provenance and record.path:
            path = Path(record.path)
            if path.is_file():
                return path.resolve()
    return None


def _numpy_dtype_for_artifact(dtype: str) -> str:
    key = str(dtype or "").strip().upper().replace("TENSOR(", "").rstrip(")")
    return {
        "BOOL": "bool",
        "DOUBLE": "float64",
        "FLOAT": "float32",
        "FLOAT16": "float16",
        "INT8": "int8",
        "INT16": "int16",
        "INT32": "int32",
        "INT64": "int64",
        "UINT8": "uint8",
        "UINT16": "uint16",
        "UINT32": "uint32",
        "UINT64": "uint64",
    }.get(key, key.lower())


def _artifact_eval_payload_contract(
    *,
    trial: TrialResult,
    artifact_path: str,
) -> Evidence:
    """Compare evaluator-owned NPZ inputs with the produced artifact interface.

    This check knows nothing about dataset semantics.  It only compares names,
    dtypes, ranks, and fixed dimensions that are already present in the artifact
    fingerprint and evaluation payload.
    """
    fingerprint = build_artifact_fingerprint(artifact_path).model_dump(mode="json")
    input_specs = [
        item for item in (fingerprint.get("input_specs") or [])
        if isinstance(item, dict) and item.get("name")
    ]
    manifest_path = _declared_artifact_path(trial, "edge_eval_manifest")
    payload_path = _declared_artifact_path(trial, "edge_eval_payload")
    protocol: Dict[str, Any] = {
        "boundary": "artifact_to_edge_eval_payload",
        "artifact_inputs": input_specs,
        "manifest_path": str(manifest_path or ""),
        "payload_path": str(payload_path or ""),
        "decision_authority": "repair_before_edge",
    }
    mismatches: list[Dict[str, Any]] = []
    observed: Dict[str, Any] = {}
    comparable = bool(
        input_specs
        and manifest_path is not None
        and payload_path is not None
        and payload_path.suffix.lower() == ".npz"
    )
    if comparable:
        try:
            import numpy as np

            with np.load(payload_path, allow_pickle=False) as payload:
                for spec in input_specs:
                    name = str(spec["name"])
                    if name not in payload.files:
                        mismatches.append({
                            "input": name,
                            "kind": "missing_input",
                        })
                        continue
                    array = payload[name]
                    expected_dtype = _numpy_dtype_for_artifact(str(spec.get("dtype") or ""))
                    actual_dtype = str(array.dtype)
                    expected_shape = list(spec.get("shape") or [])
                    actual_shape = [int(value) for value in array.shape]
                    observed[name] = {
                        "dtype": actual_dtype,
                        "shape": actual_shape,
                    }
                    if expected_dtype and actual_dtype != expected_dtype:
                        mismatches.append({
                            "input": name,
                            "kind": "dtype",
                            "expected": expected_dtype,
                            "actual": actual_dtype,
                        })
                    if expected_shape and len(expected_shape) != len(actual_shape):
                        mismatches.append({
                            "input": name,
                            "kind": "rank",
                            "expected": len(expected_shape),
                            "actual": len(actual_shape),
                        })
                    elif expected_shape:
                        for axis, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                            if isinstance(expected, int) and expected > 0 and expected != actual:
                                mismatches.append({
                                    "input": name,
                                    "kind": "shape",
                                    "axis": axis,
                                    "expected": expected,
                                    "actual": actual,
                                })
        except Exception as exc:
            comparable = False
            protocol["inspection_error"] = f"{type(exc).__name__}: {exc}"
    protocol["payload_inputs"] = observed
    protocol["mismatches"] = mismatches[:16]
    evidence = Evidence(
        probe_id="artifact_eval_payload_contract",
        quantity="component_contract",
        fidelity="static",
        outcome="fail" if comparable and mismatches else "pass" if comparable else "unknown",
        artifact_fingerprint=fingerprint,
        protocol=protocol,
    )
    _append_evidence(trial, evidence)
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["artifact_eval_payload_contract"] = {
        "status": evidence.outcome,
        "evidence_id": evidence.id,
        "mismatches": mismatches[:16],
        "artifact_inputs": input_specs,
        "payload_inputs": observed,
    }
    trial.observations = observations
    return evidence


def _run_host_infer_contract_with_repair(
    *,
    trial: TrialResult,
    variant: SolutionVariant,
    ws: Path,
    artifact_path: str,
    payload_contract: Evidence,
) -> tuple[Optional[Evidence], bool]:
    """Repair a proven artifact/payload mismatch before leasing an edge device."""
    host_contract = _run_host_infer_contract_probe(
        trial=trial,
        ws=ws,
        artifact_path=artifact_path,
    )
    host_status = (trial.observations.get("component_roundtrip") or {}).get("status")
    if host_status == "success":
        return host_contract, True
    if host_contract is None:
        return host_contract, payload_contract.outcome != "fail"
    strict_roundtrip = bool(
        getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False)
        and _declared_artifact_path(trial, "edge_eval_manifest")
    )
    # A generic host failure may only reflect a different host runtime.  It is
    # authoritative here only when the artifact/payload facts independently
    # prove an interface mismatch, or when strict evaluation executes the
    # solution-owned manifest and payload as one measured round trip.
    if payload_contract.outcome != "fail" and not strict_roundtrip:
        return host_contract, True

    infer_script = ws / "infer.py"
    max_retries = max(0, int(getattr(settings, "DEBUGGER_MAX_RETRIES", 2)))
    for retry_idx in range(max_retries):
        roundtrip = trial.observations.get("component_roundtrip") or {}
        error = str(roundtrip.get("error") or "artifact/infer contract failed")
        stderr = str(roundtrip.get("stderr_tail") or "")
        stdout = str(roundtrip.get("stdout_tail") or "")
        if not infer_script.exists():
            break
        signature = "artifact_eval_payload_contract"
        patched, summary, before_hash, after_hash, reason = _run_llm_debugger(
            script_path=infer_script,
            stage="component_host_infer",
            variant=variant,
            error=error,
            stdout=stdout,
            stderr=stderr,
            error_signature=signature,
            diagnostic_context={
                "artifact_interface": trial.observations.get(
                    "artifact_input_contract", {}
                ),
                "artifact_eval_payload_contract": payload_contract.protocol,
                "edge_eval_bundle_contract": trial.observations.get(
                    "edge_eval_bundle_contract", {}
                ),
            },
            debug_history=trial.debug_history,
        )
        trial.debug_attempts += 1
        trial.debug_history.append(DebugAttempt(
            stage="component_host_infer",
            attempt=trial.debug_attempts,
            error_signature=signature,
            reason=reason,
            before_hash=before_hash,
            after_hash=after_hash,
            patch_summary=summary,
            patch_applied=patched,
            success=False,
        ))
        if not patched:
            break
        logger.info(
            f"[{variant.trial_id}] Debugger patched infer.py before edge lease "
            f"(attempt {retry_idx + 1}/{max_retries})."
        )
        host_contract = _run_host_infer_contract_probe(
            trial=trial,
            ws=ws,
            artifact_path=artifact_path,
        )
        host_status = (trial.observations.get("component_roundtrip") or {}).get("status")
        if host_status == "success":
            _mark_debug_retry_success(trial, "component_host_infer")
            return host_contract, True
    return host_contract, False


def _artifact_matches_requested_runtime(artifact_path: str, requested_runtime: str) -> bool:
    """Return whether an artifact can satisfy the requested deployment runtime."""
    requested = _runtime_identity(requested_runtime)
    actual = _runtime_identity(artifact_kind(artifact_path))
    if requested == "engine":
        return actual in {"engine", "onnx"}
    return bool(requested and actual and requested == actual)


def _run_evaluation_metric_contract(
    *,
    trial: TrialResult,
    ws: Path,
    state: AgentState,
    artifact_path: str,
) -> None:
    """Run an explicitly supplied benchmark evaluator against the real artifact.

    This hook is absent in normal synthesis.  It keeps formal evaluation metrics
    outside generated model code and never exposes the evaluator to the proposer.
    """
    raw_path = edgecraft_env("EVALUATION_METRIC_SCRIPT", "").strip()
    if not raw_path:
        return
    script = Path(raw_path).expanduser().resolve()
    if not script.is_file() or not artifact_path:
        return
    dataset_info = state.get("dataset_info") or {}
    split_manifest = dataset_info.get("split_manifest") or {}
    result = _run_script(
        script,
        ws,
        timeout=int(edgecraft_env("EVALUATION_METRIC_TIMEOUT_S", "1800")),
        env_overrides={
            "EDGECRAFT_WORKSPACE": str(ws.resolve()),
            "EDGECRAFT_ARTIFACT_PATH": str(Path(artifact_path).resolve()),
            "EDGECRAFT_DATASET_ROOT": str(dataset_info.get("dataset_path") or ""),
            "EDGECRAFT_SPLIT_MANIFEST": str(split_manifest.get("path") or ""),
        },
    )
    execution = _record_stage_execution_evidence(
        trial,
        stage="evaluation_metric_contract",
        result=result,
        resource="gpu",
    )
    observations = dict(getattr(trial, "observations", {}) or {})
    record = {
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "status": result.get("status"),
        "execution_evidence_id": execution.id,
        "error": str(result.get("error") or "")[-800:],
    }
    if result.get("status") == "success" and isinstance(result.get("metrics"), dict):
        measured = MetricRegistry.normalize_metric_dict(result["metrics"])
        if trial.local_metrics is None:
            trial.local_metrics = LocalMetrics()
        trial.local_metrics.all_metrics.update(measured)
        record["metrics"] = measured
        for name, value in measured.items():
            _append_evidence(
                trial,
                Evidence(
                    probe_id="evaluation_metric_contract",
                    quantity=name,
                    fidelity="measured_congruent",
                    outcome="pass",
                    value=float(value),
                    artifact_fingerprint=build_artifact_fingerprint(artifact_path).model_dump(mode="json"),
                    protocol={
                        "script_sha256": record["script_sha256"],
                        "split_manifest_hash": split_manifest.get("content_hash"),
                        "execution_evidence_id": execution.id,
                        "evaluation_only": True,
                    },
                ),
            )
    observations["evaluation_metric_contract"] = record
    trial.observations = observations


def _apply_verifier_policy(
    trial: TrialResult,
    user_spec: Optional[UserSpec],
    *,
    next_probe: str,
) -> str:
    report = _ensure_verification_report(trial)
    decision = VerifierPolicy(
        kappa=float(getattr(settings, "VERIFIER_KAPPA", 2.0)),
    ).decide(report.evidence, user_spec, next_probe=next_probe)
    report.decision = decision.action
    report.decision_basis = list(decision.basis)
    report.next_probe = decision.next_probe
    report.notes = decision.reason
    return decision.action


def _extract_onnx_op_types(artifact_path: str) -> List[str]:
    if not artifact_path or not str(artifact_path).endswith(".onnx"):
        return []
    try:
        import onnx  # type: ignore

        model = onnx.load(str(artifact_path), load_external_data=False)
        return sorted({node.op_type for node in model.graph.node if node.op_type})
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"ONNX op extraction failed for {artifact_path}: {exc}")
        return []


def _runtime_version_from_state(state: AgentState, runtime: str = "") -> str:
    runtime_config = state.get("runtime_config")
    if runtime_config is None:
        return ""
    def config_value(key: str, default: Any = None) -> Any:
        if isinstance(runtime_config, dict):
            return runtime_config.get(key, default)
        return getattr(runtime_config, key, default)

    runtime_name = str(runtime or "").lower()
    if runtime_name in {"engine", "tensorrt"}:
        keys = ("trt_version", "l4t_version")
    elif runtime_name in {"onnx", "onnxruntime"}:
        keys = ("onnxruntime_version",)
    elif runtime_name in {"pt", "pth", "torch", "pytorch", "torchscript"}:
        keys = ("torch_version",)
    elif runtime_name in {"tflite", "litert"}:
        packages = config_value("python_packages", {}) or {}
        grouped = config_value("runtime_python_packages", {}) or {}
        for group_name in ("litert", "tflite", "default"):
            packages = {**packages, **dict(grouped.get(group_name) or {})}
        for package in ("ai_edge_litert", "tflite_runtime", "tensorflow"):
            value = packages.get(package)
            if value and not str(value).startswith("unavailable:"):
                return str(value)
        keys = ()
    else:
        keys = ("trt_version", "onnxruntime_version", "torch_version", "l4t_version")
    for key in keys:
        value = config_value(key)
        if value:
            return str(value)
    return ""


def _compatibility_dataset_id(state: AgentState) -> str:
    """Return a private dataset label for qualification accounting only."""
    info = state.get("dataset_info") or {}
    value = (
        info.get("dataset_id")
        or info.get("dataset_name")
        or info.get("name")
        or info.get("dataset_path")
        or state.get("dataset_path")
        or ""
    )
    return Path(str(value)).name if value else ""


def _source_onnx_path(trial: TrialResult, artifact_path: str) -> str:
    """Find the ONNX graph that produced an edge-built TensorRT artifact."""
    workspace = Path(str(getattr(trial, "workspace_path", "") or ""))
    candidates: List[str] = [artifact_path]
    candidates.extend(str(path) for path in (trial.artifact_paths or {}).values())
    contract = getattr(trial, "artifact_contract", None)
    if contract is not None:
        candidates.extend(str(path) for path in (contract.fallback_artifacts or []))
        candidates.extend(
            str(record.path)
            for record in (contract.artifacts or {}).values()
            if str(getattr(record, "kind", "") or "").lower() == "onnx"
        )
    if workspace.exists():
        candidates.append(str(workspace / "outputs" / "best.onnx"))
    for raw in candidates:
        path = Path(str(raw or ""))
        if not path.is_absolute() and workspace.exists():
            path = workspace / path
        if path.suffix.lower() == ".onnx" and path.is_file():
            return str(path.resolve())
    return ""


def _run_l0_compatibility_check(
    trial: TrialResult,
    state: AgentState,
    artifact_path: str,
) -> bool:
    """Record compatibility-rule evidence; return True only for enforce hard gate."""
    mode = _compat_rules_mode()
    if mode in {"", "off", "0", "false"}:
        return False
    fingerprint_model = build_artifact_fingerprint(artifact_path)
    fingerprint = fingerprint_model.model_dump(mode="json")
    op_types = list(fingerprint.get("op_set") or [])
    op_signatures = list(fingerprint.get("op_signatures") or [])
    subgraph_signatures = list(fingerprint_model.subgraph_signatures)
    if not op_types and not op_signatures and not subgraph_signatures and not fingerprint:
        return False
    try:
        from edgecraft.knowledge.compatibility import get_compatibility_rule_store

        runtime = str(getattr(trial.runtime_report, "requested_runtime", "") or artifact_kind(artifact_path) or "")
        env_fp = build_environment_fingerprint(
            device_id=str(state.get("target_device", "")),
            runtime=runtime,
            runtime_config=state.get("runtime_config"),
            protocol={"precision": getattr(getattr(trial, "variant", None), "quant_mode", None)},
        )
        hits = get_compatibility_rule_store().exact_match(
            device=str(state.get("target_device", "")),
            runtime=runtime,
            version=_runtime_version_from_state(state, runtime),
            precision=str(getattr(getattr(trial, "variant", None), "quant_mode", "") or ""),
            environment_fingerprint=env_fp,
            op_types=op_types,
            op_signatures=op_signatures,
            subgraph_signatures=subgraph_signatures,
            artifact_fingerprint=fingerprint,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Compatibility rule lookup failed: {exc}")
        return False
    if not hits:
        return False
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["compatibility_hits"] = [h.to_prompt_dict() for h in hits[:8]]
    observations["artifact_op_types"] = op_types[:80]
    trial.observations = observations
    static_evidence = [
        Evidence(
            probe_id="static_artifact",
            quantity="compatibility",
            fidelity="static",
            outcome="fail" if hit.can_hard_gate() else "unknown",
            artifact_fingerprint=fingerprint,
            environment_fingerprint=env_fp,
            protocol={
                "verified_rule": hit.can_hard_gate(),
                "rule_id": hit.id,
                "rule_status": hit.status,
            },
        )
        for hit in hits
    ]
    report = _append_evidence(trial, *static_evidence)
    report.level = "L0"
    report.evidence_source = "static"
    report.artifact_fingerprint = fingerprint
    report.artifact_status = "present"
    if mode != "enforce":
        return False
    blocking = [h for h in hits if h.can_hard_gate()]
    if not blocking:
        return False
    trial.error = "L0 compatibility rule blocked this artifact before edge benchmark."
    trial.error_stage = "compatibility_l0"
    report.status = "fail"
    report.runtime_status = str(getattr(trial.runtime_report, "requested_runtime", "") or artifact_kind(artifact_path) or "")
    report.errors = [trial.error]
    report.source = "compatibility_rule_store"
    _apply_verifier_policy(trial, state.get("user_spec"), next_probe="")
    trial.next_search_hint = (
        "Preserve working dataset/training path, but mutate export/runtime or operator usage "
        "to avoid the compatibility rule hit."
    )
    return True


def _observe_compatibility_failure(
    trial: TrialResult,
    state: AgentState,
    *,
    artifact_path: str,
    error_text: str,
    failure_stage: str = "",
    runtime_used: str = "",
) -> None:
    mode = _compat_rules_mode()
    if mode in {"", "off", "0", "false"}:
        return
    try:
        from edgecraft.knowledge.compatibility import get_compatibility_rule_store
        from edgecraft.knowledge.compatibility.rule_store import (
            locate_failure_artifact_predicate,
            runtime_failure_is_observable,
        )

        fingerprint_model = build_artifact_fingerprint(artifact_path)
        fingerprint = fingerprint_model.model_dump(mode="json")
        stage = str(failure_stage or getattr(trial, "error_stage", "") or "").lower()
        if locate_failure_artifact_predicate(error_text, fingerprint):
            stage = "artifact_load"
        runtime = str(
            runtime_used
            or getattr(trial.runtime_report, "runtime_used", "")
            or getattr(trial.runtime_report, "requested_runtime", "")
            or artifact_kind(artifact_path)
            or ""
        )
        if not runtime_failure_is_observable(
            stage=stage,
            runtime=runtime,
            error_text=error_text,
            artifact_fingerprint=fingerprint,
        ):
            return
        env_fp = build_environment_fingerprint(
            device_id=str(state.get("target_device", "")),
            runtime=runtime,
            runtime_config=state.get("runtime_config"),
            protocol={"precision": getattr(getattr(trial, "variant", None), "quant_mode", None)},
        )
        observation = get_compatibility_rule_store().observe_failure(
            device=str(state.get("target_device", "")),
            runtime=runtime,
            version=_runtime_version_from_state(state, runtime),
            precision=str(getattr(getattr(trial, "variant", None), "quant_mode", "") or ""),
            error_text=error_text,
            artifact_path=artifact_path,
            tenant_id=str(state.get("tenant_id", "")),
            dataset_id=_compatibility_dataset_id(state),
            run_id=str(state.get("run_id", "")),
            trial_id=str(getattr(trial, "trial_id", "")),
            stage=stage or "edge_benchmark",
            environment_fingerprint=env_fp,
            artifact_fingerprint=fingerprint,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Compatibility rule observation failed: {exc}")
        return
    if not observation:
        return
    observations = dict(getattr(trial, "observations", {}) or {})
    items = list(observations.get("failure_observations") or [])
    items.append(observation.to_prompt_dict())
    observations["failure_observations"] = items[-8:]
    trial.observations = observations
    if mode == "enforce":
        _attempt_minimal_repro_validation(
            trial=trial,
            state=state,
            observation=observation,
            artifact_path=artifact_path,
        )


def _attempt_minimal_repro_validation(
    *,
    trial: TrialResult,
    state: AgentState,
    observation: Any,
    artifact_path: str,
) -> None:
    """Replay a runtime failure; only a minimal reproduction may create a rule."""
    workspace = Path(str(getattr(trial, "workspace_path", "") or ""))
    if not workspace.exists():
        return
    try:
        from edgecraft.knowledge.compatibility import (
            get_compatibility_rule_store,
            prepare_reproduction_plan,
        )
        from edgecraft.knowledge.compatibility.rule_store import (
            failure_observation_matches,
        )
        from edgecraft.tools.deploy.edge_runner import EdgeRunner

        store = get_compatibility_rule_store()
        repro_dir = workspace / "compatibility_repro" / observation.id
        runtime = str(
            getattr(observation, "runtime", "")
            or getattr(trial.runtime_report, "runtime_used", "")
            or getattr(trial.runtime_report, "requested_runtime", "")
            or artifact_kind(artifact_path)
            or ""
        )
        source_onnx = _source_onnx_path(trial, artifact_path)
        original_fingerprint = build_artifact_fingerprint(
            artifact_path,
            source_artifact_path=source_onnx,
        )
        plan = prepare_reproduction_plan(
            artifact_path=artifact_path,
            runtime=runtime,
            output_dir=str(repro_dir),
            op_type=str(getattr(observation, "op_type", "") or ""),
            artifact_predicate=dict(getattr(observation, "artifact_predicate", {}) or {}),
            input_specs=list(original_fingerprint.input_specs),
            source_onnx_path=source_onnx,
        )
        protocol = {
            "device": state.get("target_device"),
            "runtime": runtime,
            "runtime_version": _runtime_version_from_state(state, runtime),
            "precision": str(getattr(getattr(trial, "variant", None), "quant_mode", "") or ""),
            "environment_fingerprint": observation.environment_fingerprint,
            "synthetic_input_only": True,
            "reproduction_scope": plan.scope,
            "gate_eligible": plan.gate_eligible,
        }
        if not plan.artifact_path or not plan.infer_script:
            store.record_reproduction(
                observation,
                status="unavailable",
                scope=plan.scope,
                protocol={**protocol, "reason": plan.reason},
            )
            observations = dict(getattr(trial, "observations", {}) or {})
            attempts = list(observations.get("compatibility_repro_attempts") or [])
            attempts.append({
                "observation_id": observation.id,
                **plan.to_dict(),
                "status": "unavailable",
            })
            observations["compatibility_repro_attempts"] = attempts[-8:]
            trial.observations = observations
            return

        result = EdgeRunner().run(
            artifact_path=plan.artifact_path,
            device_id=str(state.get("target_device", "")),
            device_ip=str(state.get("device_ip", "")),
            ssh_key=str(state.get("ssh_key", "")),
            workspace_path=str(repro_dir),
            infer_script=plan.infer_script,
            dataset_path=None,
            timeout=min(600, int(getattr(settings, "L1_EDGE_TIMEOUT_S", 600))),
            docker_image=state.get("docker_image"),
            export_format=runtime,
            quant_mode=getattr(getattr(trial, "variant", None), "quant_mode", None) or "fp16",
            graph_hash=build_artifact_fingerprint(
                plan.artifact_path,
                source_artifact_path=source_onnx,
            ).graph_hash,
            require_exact_runtime=True,
        )
        result_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        repro_runtime_used = result.get("runtime_used") or result_metrics.get("runtime_used")
        fallback = _runtime_fallback_evidence(
            result,
            requested_runtime=runtime,
            runtime_used=repro_runtime_used,
        )
        reproduced_text = fallback.get("error_text") or "\n".join(
            str(result.get(key) or "")
            for key in ("error", "stderr_tail", "stderr", "stdout_tail", "stdout")
        )
        rule = None
        if result.get("status") != "success" or fallback:
            fingerprint_model = build_artifact_fingerprint(plan.artifact_path)
            fingerprint = fingerprint_model.model_dump(mode="json")
            signature = next(
                (
                    item
                    for item in fingerprint.get("op_signatures") or []
                    if observation.op_type and observation.op_type.lower() in str(item).lower()
                ),
                "",
            )
            subgraph_signature = next(
                (
                    item
                    for item in fingerprint_model.subgraph_signatures
                    if observation.op_type
                    and str(item.get("op_type") or "").lower() == observation.op_type.lower()
                ),
                {},
            )
            if plan.gate_eligible:
                pattern = dict(plan.pattern)
                if subgraph_signature:
                    pattern.update({
                        "op_signature": signature,
                        "subgraph_signature": subgraph_signature,
                    })
                rule = store.verify_observation(
                    observation,
                    reproduced_error_text=reproduced_text,
                    minimal_repro_path=plan.artifact_path,
                    pattern=pattern,
                    protocol=protocol,
                )
        matched = failure_observation_matches(observation, reproduced_text)
        if not rule:
            status = (
                "not_reproduced"
                if result.get("status") == "success" and not fallback
                else "reproduced_observation" if matched else "different_failure"
            )
            store.record_reproduction(
                observation,
                status=status,
                scope=plan.scope,
                path=plan.artifact_path,
                reproduced_error_text=reproduced_text,
                protocol=protocol,
            )
        observations = dict(getattr(trial, "observations", {}) or {})
        attempts = list(observations.get("compatibility_repro_attempts") or [])
        attempts.append(
            {
                "observation_id": observation.id,
                "repro_path": plan.artifact_path,
                "reproduction_scope": plan.scope,
                "gate_eligible": plan.gate_eligible,
                "reason": plan.reason,
                "status": result.get("status"),
                "requested_runtime_status": "failed" if fallback else result.get("status"),
                "fallback_reason": fallback.get("fallback_reason", ""),
                "verified_rule_id": getattr(rule, "id", "") if rule else "",
                "error_fingerprint_match": matched,
            }
        )
        observations["compatibility_repro_attempts"] = attempts[-8:]
        trial.observations = observations
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Compatibility minimal reproduction failed safely: {exc}")


def _script_supports_l1_efficiency(train_script: Path) -> bool:
    try:
        text = train_script.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return (
        "--probe" in text
        or "probe efficiency" in text
        or "--verify-efficiency" in text
        or "verify_efficiency" in text
    )


def _efficiency_probe_args(train_script: Path) -> List[str]:
    try:
        text = train_script.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return ["--probe", "efficiency"]
    if "--probe" in text or "probe efficiency" in text:
        return ["--probe", "efficiency"]
    return ["--verify-efficiency"]


def _efficiency_probe_result_contract(
    result: Dict[str, Any],
    *,
    candidate_source_sha256: str,
    guard_source_sha256: str,
) -> Dict[str, Any]:
    """Validate candidate output and controller-owned P1 construction evidence."""
    reported_probe = str(result.get("probe") or "").strip().lower()
    raw_steps = result.get("training_steps")
    zero_steps = False
    if raw_steps is not None and not isinstance(raw_steps, bool):
        try:
            zero_steps = float(raw_steps) == 0.0
        except (TypeError, ValueError):
            zero_steps = False

    violations: List[str] = []
    if result.get("status") != "success":
        violations.append("probe execution did not report status=success")
    if reported_probe != "efficiency":
        violations.append("final JSON must report probe=efficiency")
    if not zero_steps:
        violations.append("final JSON must report training_steps=0")
    construction_guard = validate_p1_guard_attestation(
        result.get("p1_guard_attestation"),
        candidate_source_sha256=candidate_source_sha256,
        guard_source_sha256=guard_source_sha256,
    )
    violations.extend(construction_guard["violations"])
    if bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False)):
        artifacts = result.get("artifact_paths")
        if not isinstance(artifacts, dict):
            violations.append(
                "strict efficiency probe must declare artifact_paths"
            )
        else:
            for key in ("edge_eval_manifest", "edge_eval_payload"):
                if not artifacts.get(key):
                    violations.append(f"strict efficiency probe must declare {key}")
    return {
        "valid": not violations,
        "reported_probe": reported_probe,
        "training_steps": raw_steps,
        "construction_guard": construction_guard,
        "violations": violations,
    }


def _configured_probes() -> List[str]:
    raw = str(getattr(settings, "VERIFIER_PROBES", "static_artifact,efficiency,full") or "")
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _run_quality_proxy_probe(
    *,
    trial: TrialResult,
    train_script: Path,
    ws: Path,
    state: AgentState,
) -> None:
    """Collect optional few-step quality evidence; never control execution."""
    if not _evidence_ladder_enabled() or "quality_proxy" not in _configured_probes():
        return
    try:
        text = train_script.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return
    if "--probe" not in text:
        return
    steps = max(1, int(getattr(settings, "VERIFIER_QUALITY_STEPS", 100)))
    started = time.perf_counter()
    result = _run_script(
        train_script,
        ws,
        timeout=int(getattr(settings, "L1_EDGE_TIMEOUT_S", 600)),
        script_args=["--probe", "quality", "--train-steps", str(steps)],
    )
    wall_s = time.perf_counter() - started
    metrics = MetricRegistry.normalize_metric_dict(
        result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    )
    items = [
        Evidence(
            probe_id="quality_proxy",
            quantity=quantity,
            fidelity="proxy",
            outcome="unknown" if result.get("status") == "success" else "fail",
            value=float(value),
            environment_fingerprint=build_environment_fingerprint(
                device_id=str(state.get("target_device", "")),
                runtime="cloud_train",
                runtime_config=state.get("runtime_config"),
            ),
            protocol={
                "train_steps": steps,
                "decision_authority": "rank_only",
                "dataset_subset_hash": result.get("dataset_subset_hash"),
            },
            resource_cost=ResourceCost(gpu_s=wall_s),
        )
        for quantity, value in metrics.items()
        if not MetricRegistry.is_edge_metric(quantity)
    ]
    if items:
        _append_evidence(trial, *items)
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["quality_proxy"] = {
        "attempted": True,
        "status": result.get("status"),
        "train_steps": steps,
        "evidence_ids": [item.id for item in items],
        "error": result.get("error"),
    }
    trial.observations = observations


def _repair_efficiency_probe_contract(
    *,
    trial: TrialResult,
    variant: SolutionVariant,
    train_script: Path,
    result: Dict[str, Any],
    probe_contract: Dict[str, Any],
) -> bool:
    """Give a broken generated probe one bounded, evidence-driven repair."""
    if result.get("timed_out"):
        return False
    error = str(result.get("error") or "; ".join(probe_contract.get("violations") or []))
    patched, summary, before_hash, after_hash, reason = _run_llm_debugger(
        script_path=train_script,
        stage="train",
        variant=variant,
        error=error,
        stdout=str(result.get("stdout") or ""),
        stderr=str(result.get("stderr") or ""),
        error_signature="efficiency_probe_contract",
        diagnostic_context={"efficiency_probe_contract": probe_contract},
        debug_history=trial.debug_history,
    )
    trial.debug_attempts += 1
    trial.debug_history.append(DebugAttempt(
        stage="efficiency_probe",
        attempt=trial.debug_attempts,
        error_signature="efficiency_probe_contract",
        reason=reason,
        before_hash=before_hash,
        after_hash=after_hash,
        patch_summary=summary,
        patch_applied=patched,
        success=False,
    ))
    if patched:
        logger.info(
            f"[{variant.trial_id}] Debugger repaired the efficiency probe before "
            "full training; retrying the probe once."
        )
    return patched


def _run_l1_efficiency_probe(
    *,
    trial: TrialResult,
    state: AgentState,
    variant: SolutionVariant,
    ws: Path,
    train_script: Path,
    user_spec: Optional[UserSpec],
) -> bool:
    """Run the configured cheap measured efficiency probe.

    Returns True only when the current trial should stop before full training.
    The paper profile rejects a candidate that cannot enter P1; the explicitly
    weaker offline/full profile may continue without this ladder stage.
    """
    if not _evidence_ladder_enabled():
        return False
    strict_ladder = bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
    if not _script_supports_l1_efficiency(train_script):
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["l1_efficiency"] = {
            "attempted": False,
            "reason": "train_flag_missing",
            "decision": "reject_candidate_contract" if strict_ladder else "escalate",
        }
        trial.observations = observations
        if strict_ladder:
            report = _ensure_verification_report(trial)
            report.level = "admission"
            report.status = "fail"
            report.evidence_source = "static"
            report.decision = "escalate"
            report.admission_status = "rejected"
            report.errors = [
                "paper-profile candidate does not implement the graph-preserving P1 efficiency probe"
            ]
            report.source = "candidate_admission"
            report.notes = (
                "Every executable paper-profile candidate must expose P1 before full P2 training."
            )
            trial.error = report.errors[0]
            trial.error_stage = "efficiency_probe_contract"
            trial.next_search_hint = (
                "Preserve the candidate graph and add an untrained --probe efficiency path "
                "that emits the strict edge-evaluation bundle."
            )
            trial.crafting_progress = CraftingProgress.SCRIPT_GENERATED
            trial.stage_reached = StageReached.TRAIN
            return True
        return False
    device_ip = state.get("device_ip", "")
    ssh_key = state.get("ssh_key", "")
    if not device_ip or not ssh_key:
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["l1_efficiency"] = {
            "attempted": False,
            "reason": "edge_device_missing",
            "decision": "reject_candidate_contract" if strict_ladder else "escalate",
        }
        trial.observations = observations
        if strict_ladder:
            report = _ensure_verification_report(trial)
            report.level = "admission"
            report.status = "fail"
            report.evidence_source = "static"
            report.decision = "escalate"
            report.admission_status = "rejected"
            report.errors = [
                "paper-profile P1 requires an edge endpoint and explicit SSH private key"
            ]
            report.source = "candidate_admission"
            report.notes = "P1 cannot be bypassed by omitting the physical target."
            trial.error = report.errors[0]
            trial.error_stage = "efficiency_probe_contract"
            trial.next_search_hint = (
                "Supply the target edge endpoint and explicit SSH key, then retry the same candidate."
            )
            trial.crafting_progress = CraftingProgress.SCRIPT_GENERATED
            trial.stage_reached = StageReached.TRAIN
            return True
        return False

    probe_args = _efficiency_probe_args(train_script)
    logger.debug(f"[{variant.trial_id}] efficiency probe: executing train.py {' '.join(probe_args)}")
    probe_wall_s = 0.0
    execution_evidence_ids: List[str] = []
    repaired_probe = False
    probe_repair_attempted = False
    for probe_attempt in range(2):
        candidate_source_sha256 = _sha256_file(train_script)
        guard_source_sha256 = _sha256_file(P1_GUARD_RUNNER)
        probe_started = time.perf_counter()
        result = _run_script(
            train_script,
            ws,
            timeout=int(getattr(settings, "L1_PROBE_TIMEOUT_S", 300)),
            script_args=probe_args,
            p1_guarded=True,
        )
        attempt_wall_s = time.perf_counter() - probe_started
        probe_wall_s += attempt_wall_s
        probe_execution = _record_stage_execution_evidence(
            trial,
            stage="efficiency_probe_export",
            result={**result, "elapsed_s": attempt_wall_s},
            resource="gpu",
        )
        execution_evidence_ids.append(probe_execution.id)
        probe_contract = _efficiency_probe_result_contract(
            result,
            candidate_source_sha256=candidate_source_sha256,
            guard_source_sha256=guard_source_sha256,
        )
        if probe_contract["valid"]:
            if repaired_probe:
                _mark_debug_retry_success(trial, "efficiency_probe")
            break
        if probe_attempt == 0:
            attempts_before = trial.debug_attempts
            repaired_probe = _repair_efficiency_probe_contract(
                trial=trial,
                variant=variant,
                train_script=train_script,
                result=result,
                probe_contract=probe_contract,
            )
            probe_repair_attempted = trial.debug_attempts > attempts_before
            if repaired_probe:
                continue
        break
    contract_evidence = Evidence(
        probe_id="efficiency",
        quantity="probe_contract",
        fidelity="proxy",
        outcome=(
            "pass"
            if probe_contract["valid"]
            else "unknown" if result.get("timed_out") else "fail"
        ),
        protocol={
            **probe_contract,
            "requested_probe": "efficiency",
            "execution_evidence_id": probe_execution.id,
            "elapsed_s": probe_wall_s,
            "timeout_s": result.get("timeout_s"),
            "decision_authority": "audit_only",
            "repair_attempted": probe_repair_attempted,
            "execution_evidence_ids": execution_evidence_ids,
        },
    )
    _append_evidence(trial, contract_evidence)
    if not probe_contract["valid"]:
        strict_bundle_violation = bool(
            getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False)
        ) and any(
            str(item).startswith("strict efficiency probe")
            for item in probe_contract["violations"]
        )
        report = _ensure_verification_report(trial)
        report.level = "admission" if strict_bundle_violation else "L1"
        report.status = "fail" if strict_bundle_violation else "skipped"
        report.evidence_source = "static" if strict_bundle_violation else "measured"
        report.decision = "escalate"
        report.admission_status = "rejected" if strict_bundle_violation else "admitted"
        report.errors = [
            *[str(item) for item in probe_contract["violations"]],
            *([str(result.get("error"))] if result.get("error") else []),
        ]
        report.source = "candidate_admission" if strict_bundle_violation else "efficiency_probe_contract"
        report.notes = (
            "Strict evaluation requires a complete edge-evaluation bundle before "
            "full training. The current trial stops so a child can repair the contract."
            if strict_bundle_violation
            else "The generated efficiency path did not satisfy the executed probe "
            "contract; escalating to full training without granting L1 prune authority."
        )
        if strict_bundle_violation:
            trial.error = "; ".join(str(item) for item in probe_contract["violations"])
            trial.error_stage = "edge_eval_contract"
            trial.next_search_hint = (
                "Preserve the model and export path; make the efficiency probe emit "
                "edge_eval_manifest and edge_eval_payload through artifact_paths."
            )
            trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
            trial.stage_reached = StageReached.TRAIN
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["l1_efficiency"] = {
            "attempted": True,
            "measured": False,
            "decision": "reject_candidate_contract" if strict_bundle_violation else "escalate",
            "probe_contract": probe_contract,
            "probe_contract_evidence_id": contract_evidence.id,
            "execution_evidence_id": probe_execution.id,
            "execution_evidence_ids": execution_evidence_ids,
            "repaired_probe": repaired_probe,
            "repair_attempted": probe_repair_attempted,
            "probe_wall_s": probe_wall_s,
            "error": result.get("error"),
            "stdout_tail": (result.get("stdout", "") or "")[-1000:],
            "stderr_tail": (result.get("stderr", "") or "")[-1000:],
        }
        trial.observations = observations
        return strict_bundle_violation

    raw_model_path = result.get("model_path", "")
    try:
        artifact_path = _resolve_and_stage_model(raw_model_path, ws, variant.trial_id, variant)
    except Exception as exc:  # noqa: BLE001
        report = _ensure_verification_report(trial)
        report.level = "L1"
        report.status = "skipped"
        report.evidence_source = "measured"
        report.decision = "escalate"
        report.errors = [f"L1 artifact resolve failed: {exc}"]
        report.source = "verify_efficiency"
        return False
    _record_workspace_artifacts(trial, ws, variant)
    _record_declared_artifacts(trial, ws, result.get("artifact_paths"))
    probe_bundle_contract = _audit_edge_eval_bundle(
        trial,
        ws,
        state,
        result.get("artifact_paths"),
        require_official_split=False,
    )
    strict_probe_bundle_error = _strict_edge_eval_bundle_error(probe_bundle_contract)
    if strict_probe_bundle_error:
        report = _ensure_verification_report(trial)
        report.level = "admission"
        report.status = "fail"
        report.evidence_source = "static"
        report.decision = "escalate"
        report.admission_status = "rejected"
        report.source = "candidate_admission"
        if probe_bundle_contract is not None:
            report.decision_basis = [probe_bundle_contract.id]
        report.errors = [strict_probe_bundle_error]
        report.notes = (
            "The efficiency probe exposed an invalid edge evaluation bundle "
            "before full training."
        )
        trial.error = strict_probe_bundle_error
        trial.error_stage = "edge_eval_contract"
        trial.next_search_hint = (
            "Preserve the model and export path; repair edge_eval_manifest, "
            "edge_eval_payload, and infer.py as one self-contained contract."
        )
        trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
        trial.stage_reached = StageReached.TRAIN
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["l1_efficiency"] = {
            "attempted": True,
            "measured": False,
            "decision": "reject_candidate_contract",
            "reason": "edge_eval_bundle_contract",
        }
        trial.observations = observations
        return True
    deploy_artifact = _primary_deploy_artifact(trial) or artifact_path
    l1_fp = build_artifact_fingerprint(deploy_artifact).model_dump(mode="json")
    requested_runtime = str(getattr(variant, "export_format", None) or artifact_kind(deploy_artifact) or "")
    env_fp = build_environment_fingerprint(
        device_id=str(state.get("target_device", "")),
        runtime=requested_runtime,
        runtime_config=state.get("runtime_config"),
        protocol={"precision": getattr(variant, "quant_mode", None)},
    )
    trial.runtime_report.requested_runtime = requested_runtime
    if _run_l0_compatibility_check(trial, state, deploy_artifact):
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["l1_efficiency"] = {
            "attempted": True,
            "measured": False,
            "decision": "prune",
            "reason": "verified_static_compatibility_rule",
            "artifact_fingerprint": l1_fp,
            "environment_fingerprint": env_fp,
        }
        trial.observations = observations
        return True

    edge_started = time.perf_counter()
    try:
        evaluator_infer = write_runtime_repro_infer(
            str(ws / ".edgecraft" / "efficiency_infer.py"),
            input_specs=list(l1_fp.get("input_specs") or []),
            warmup=EFFICIENCY_PROBE_WARMUP,
            min_warmup_seconds=EFFICIENCY_PROBE_MIN_WARMUP_SECONDS,
            repetitions=EFFICIENCY_PROBE_REPETITIONS,
            min_measure_seconds=EFFICIENCY_PROBE_MIN_MEASURE_SECONDS,
            measurement_sessions=EFFICIENCY_PROBE_SESSIONS,
        )
        if _use_scheduler_backend():
            from edgecraft.scheduler.shared_runtime import get_shared_scheduler_runtime

            artifact_paths = {
                str(key): str(value)
                for key, value in (trial.artifact_paths or {}).items()
                if value
            }
            artifact_paths.update(
                {"artifact": str(deploy_artifact), "train": str(deploy_artifact)}
            )
            spec = JobSpec(
                job_id=f"{state['run_id']}:{variant.trial_id}:p1",
                tenant_id=state.get("tenant_id") or "default",
                request_id=state["run_id"],
                variant=variant,
                workspace_path=str(ws),
                infer_script=str(evaluator_infer),
                edge_device_id=state.get("target_device", "") or "",
                edge_device_ip=device_ip,
                edge_ssh_key=ssh_key,
                edge_docker_image=state.get("docker_image"),
                dataset_path=str(
                    (state.get("dataset_info") or {}).get("dataset_path")
                    or state.get("dataset_path")
                    or ""
                ),
                export_format=getattr(variant, "export_format", None) or "onnx",
                quant_mode=getattr(variant, "quant_mode", None) or "fp16",
                require_exact_runtime=True,
                edge_only=True,
                prebuilt_artifact_paths=artifact_paths,
                artifact_manifest=trial.artifact_contract.model_dump(mode="json"),
                stage_label="p1_efficiency",
                estimate=WorkloadEstimate(
                    gpu_memory_gb=0.0,
                    gpu_wall_time_s=0.0,
                    edge_wall_time_s=float(getattr(settings, "L1_EDGE_TIMEOUT_S", 600)),
                    confidence=1.0,
                    source="p1_edge_only",
                ),
                edge_timeout_s=int(getattr(settings, "L1_EDGE_TIMEOUT_S", 600)),
                prior_score=float(getattr(variant, "prior_score", 0.5) or 0.5),
            )
            timeout_s = float(settings.SCHEDULER_JOB_TIMEOUT_S or 0.0) or None
            scheduled = get_shared_scheduler_runtime().submit_batch(
                [spec], timeout_s=timeout_s
            )[0]
            edge_result = dict((scheduled.diagnostics or {}).get("edge_result") or {})
            edge_result["metrics"] = dict(scheduled.edge_metrics or {})
            edge_result.setdefault(
                "status",
                "success" if scheduled.status == JobStatus.SUCCESS else "error",
            )
            if scheduled.error:
                edge_result.setdefault("error", scheduled.error)
            edge_result["stdout"] = scheduled.stdout
            edge_result["stderr"] = scheduled.stderr
            edge_result["scheduler_job_id"] = scheduled.job_id
            edge_wall_s = float(scheduled.edge_wall_s or 0.0)
        else:
            from edgecraft.tools.deploy.edge_runner import EdgeRunner

            runner = EdgeRunner()
            edge_result = runner.run(
                artifact_path=deploy_artifact,
                device_id=state.get("target_device", ""),
                device_ip=device_ip,
                ssh_key=ssh_key,
                workspace_path=str(ws),
                infer_script=evaluator_infer,
                dataset_path=str((state.get("dataset_info") or {}).get("dataset_path") or state.get("dataset_path") or ""),
                timeout=int(getattr(settings, "L1_EDGE_TIMEOUT_S", 600)),
                docker_image=state.get("docker_image"),
                export_format=getattr(variant, "export_format", None) or "onnx",
                quant_mode=getattr(variant, "quant_mode", None) or "fp16",
                graph_hash=str(l1_fp.get("graph_hash") or ""),
                artifact_manifest=trial.artifact_contract.model_dump(mode="json"),
                require_exact_runtime=True,
            )
            edge_wall_s = time.perf_counter() - edge_started
    except Exception as exc:  # noqa: BLE001
        report = _ensure_verification_report(trial)
        report.level = "L1"
        report.status = "skipped"
        report.evidence_source = "measured"
        report.decision = "escalate"
        report.artifact_fingerprint = l1_fp
        report.errors = [f"L1 edge measurement failed: {exc}"]
        report.source = "verify_efficiency"
        return False
    if not _use_scheduler_backend():
        edge_wall_s = time.perf_counter() - edge_started
    edge_execution_evidence = _record_stage_execution_evidence(
        trial,
        stage="efficiency_probe_edge",
        result={
            **(edge_result if isinstance(edge_result, dict) else {}),
            "elapsed_s": edge_wall_s,
        },
        resource="device",
    )

    raw_metrics = edge_result.get("metrics", {}) if isinstance(edge_result, dict) else {}
    metrics = MetricRegistry.normalize_metric_dict(raw_metrics if isinstance(raw_metrics, dict) else {})
    artifact_executed, execution_reasons = _artifact_execution_evidence(raw_metrics)
    estimates: Dict[str, MetricEstimate] = {}
    latency = _safe_float(metrics.get("Latency_p95")) if artifact_executed else None
    memory = _safe_float(metrics.get("Memory_mb")) if artifact_executed else None
    energy = _safe_float(metrics.get("Energy_mj")) if artifact_executed else None
    power = _safe_float(metrics.get("Power_w")) if artifact_executed else None
    if latency is not None:
        estimates["Latency"] = MetricEstimate(value=latency, source="L1_edge_benchmark", unit="ms")
    if memory is not None:
        estimates["Memory_mb"] = MetricEstimate(value=memory, source="L1_edge_benchmark", unit="MB")
    if energy is not None:
        estimates["Energy_mj"] = MetricEstimate(value=energy, source="L1_edge_benchmark", unit="mJ")
    if power is not None:
        estimates["Power_w"] = MetricEstimate(value=power, source="L1_edge_benchmark", unit="W")

    runtime_fields = _resolve_edge_runtime_fields(edge_result)
    runtime_used = _runtime_identity(runtime_fields["runtime_used"] or requested_runtime)
    measured_env_fp = _runtime_environment_fingerprint(
        state,
        runtime=runtime_used or requested_runtime,
        precision=getattr(variant, "quant_mode", None),
    )
    sigma_by_metric = {
        "Latency": _safe_float(metrics.get("latency_p95_session_std_ms")),
        "Memory_mb": _safe_float(metrics.get("memory_std_mb")),
        "Energy_mj": _safe_float(metrics.get("energy_std_mj")),
        "Power_w": _safe_float(metrics.get("power_std_w")),
    }
    units = {"Latency": "ms", "Memory_mb": "MB", "Energy_mj": "mJ", "Power_w": "W"}
    values = {"Latency": latency, "Memory_mb": memory, "Energy_mj": energy, "Power_w": power}
    evidence_items: List[Evidence] = []
    base_protocol = _l1_measurement_protocol(
        edge_result=edge_result,
        raw_metrics=raw_metrics,
        runtime_used=runtime_used,
        runtime_provider=runtime_fields["runtime_provider"],
        precision=getattr(variant, "quant_mode", None),
        input_specs=list(l1_fp.get("input_specs") or []),
    )
    source_hash = str(edge_result.get("source_artifact_hash") or "")
    local_hash = str(l1_fp.get("artifact_hash") or "")
    requested_graph_hash = str(edge_result.get("requested_graph_hash") or "")
    graph_hash = str(l1_fp.get("graph_hash") or "")
    fingerprint_verified = bool(
        source_hash
        and local_hash
        and source_hash == local_hash
        and graph_hash
        and requested_graph_hash == graph_hash
        and runtime_fields["artifact_used"]
    )
    runtime_exact = bool(
        edge_result.get("status") == "success"
        and not edge_result.get("fallback_reason")
        and _runtime_identity(runtime_used) == _runtime_identity(requested_runtime)
        and edge_result.get("require_exact_runtime") is True
    )
    measurement_succeeded = bool(
        edge_result.get("status") == "success" and artifact_executed
    )
    construction_guard = dict(probe_contract.get("construction_guard") or {})
    for quantity, value in values.items():
        if value is None:
            continue
        protocol = dict(base_protocol)
        protocol_fp = calibration_protocol_fingerprint(protocol)
        snapshot = _calibration_snapshot(
            environment_fingerprint=measured_env_fp,
            protocol_fingerprint=protocol_fp,
            graph_hash=str(l1_fp.get("graph_hash") or ""),
        )
        pair_ids = list(snapshot.get("pair_ids") or [])
        calibration_error = (snapshot.get("errors") or {}).get(quantity)
        authorized = measurement_succeeded and bool(
            l1_fp.get("graph_hash")
            and pair_ids
            and calibration_error is not None
            and runtime_exact
            and fingerprint_verified
            and construction_guard.get("valid") is True
            and (
                quantity != "Latency"
                or (
                    sigma_by_metric.get("Latency") is not None
                    and int(base_protocol.get("measurement_sessions") or 0) >= 2
                    and base_protocol.get("latency_statistic") == "p95"
                    and base_protocol.get("latency_uncertainty")
                    == "sample_std_across_session_p95"
                )
            )
        )
        protocol.update(
            {
                "calibration_error": calibration_error,
                "calibration_snapshot_id": snapshot.get("snapshot_id"),
                "calibration_pair_ids": pair_ids,
                "calibration_protocol_fingerprint": protocol_fp,
                "congruence_status": (
                    "calibrated_exact_graph" if authorized else "unavailable"
                ),
                "runtime_exact": runtime_exact,
                "fingerprint_verified": fingerprint_verified,
                "p1_construction_guard": construction_guard.get("schema_version"),
                "p1_construction_guard_status": construction_guard.get("status"),
                "p1_candidate_source_sha256": construction_guard.get(
                    "candidate_source_sha256"
                ),
                "p1_guard_source_sha256": construction_guard.get(
                    "guard_source_sha256"
                ),
                "p1_candidate_source_unchanged": construction_guard.get(
                    "candidate_source_unchanged"
                ),
                "p1_guard_source_unchanged": construction_guard.get(
                    "guard_source_unchanged"
                ),
                "p1_process_spawn_policy": construction_guard.get(
                    "process_spawn_policy"
                ),
                "p1_training_mutation_count": construction_guard.get(
                    "training_mutation_count"
                ),
                "decision_authority": "p1_prune" if authorized else "audit_only",
            }
        )
        evidence_items.append(
            Evidence(
                probe_id="efficiency",
                quantity=quantity,
                fidelity="measured_congruent" if authorized else "proxy",
                outcome="pass" if edge_result.get("status") == "success" else "fail",
                value=value,
                sigma=sigma_by_metric.get(quantity),
                unit=units[quantity],
                artifact_fingerprint=l1_fp,
                environment_fingerprint=measured_env_fp,
                protocol=protocol,
            )
        )
    measured = measurement_succeeded and bool(evidence_items)
    report = _ensure_verification_report(trial)
    report.level = "L1"
    report.status = "pass" if measured else "skipped"
    report.evidence_source = "measured"
    report.metric_estimates.update(estimates)
    report.artifact_fingerprint = l1_fp
    report.artifact_status = "present"
    report.runtime_status = runtime_used
    report.errors = (
        []
        if measured
        else [str(edge_result.get("error") or "L1 edge run produced no measurable efficiency metric")]
    )
    report.source = "efficiency_probe"
    _append_evidence(trial, *evidence_items)
    decision = _apply_verifier_policy(trial, user_spec, next_probe="full")
    prune_reason = trial.verification_report.notes if decision == "prune" else None
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["edge_execution_contract"] = {
        "artifact_executed": artifact_executed,
        "reasons": execution_reasons,
    }
    observations["l1_efficiency"] = {
        "attempted": True,
        "measured": measured,
        "decision": decision,
        "probe_contract": probe_contract,
        "probe_contract_evidence_id": contract_evidence.id,
        "execution_evidence_id": probe_execution.id,
        "execution_evidence_ids": execution_evidence_ids,
        "repair_attempted": probe_repair_attempted,
        "repaired_probe": repaired_probe,
        "probe_wall_s": probe_wall_s,
        "edge_wall_s": edge_wall_s,
        "prune_reason": prune_reason,
        "calibration_snapshots": {
            item.quantity: item.protocol.get("calibration_snapshot_id")
            for item in evidence_items
        },
        "artifact_fingerprint": l1_fp,
        "requested_environment_fingerprint": env_fp,
        "environment_fingerprint": measured_env_fp,
        "runtime_exact": runtime_exact,
        "fingerprint_verified": fingerprint_verified,
        "evidence_ids": [item.id for item in evidence_items],
        "edge_execution_evidence_id": edge_execution_evidence.id,
        "metrics": metrics,
    }
    trial.observations = observations
    if not measured or not prune_reason:
        return False

    audited_trials = list(state.get("l1_forced_audit_trial_ids") or [])
    if (
        bool(getattr(settings, "L1_FALSE_PRUNE_SAMPLE", True))
        and trial.trial_id not in audited_trials
    ):
        audited_trials.append(trial.trial_id)
        state["l1_forced_audit_trial_ids"] = audited_trials
        observations["l1_efficiency"]["false_prune_check"] = True
        observations["l1_efficiency"]["decision"] = "escalate_for_false_prune_check"
        trial.verification_report.decision = "escalate"
        trial.verification_report.notes += "; forced full verification for false-prune audit."
        trial.observations = observations
        return False

    trial.error = f"L1 efficiency prune: {prune_reason}"
    trial.error_stage = "l1_efficiency"
    trial.edge_metrics = EdgeMetrics(
        latency_ms=latency,
        latency_mean_ms=_safe_float(metrics.get("Latency_mean")),
        latency_p95_ms=latency,
        latency_p99_ms=_safe_float(metrics.get("Latency_p99")),
        memory_mb=memory,
        runtime_used=runtime_fields["runtime_used"],
        runtime_provider=runtime_fields["runtime_provider"],
        artifact_used=runtime_fields["artifact_used"] or deploy_artifact,
        all_metrics=metrics,
    )
    trial.runtime_report.runtime_used = trial.edge_metrics.runtime_used or "L1_measured"
    trial.runtime_report.runtime_provider = runtime_fields["runtime_provider"]
    trial.runtime_report.artifact_used = runtime_fields["artifact_used"] or deploy_artifact
    trial.stage_reached = StageReached.EDGE_BENCHMARK
    trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
    return True


def _extract_python_code(raw: str) -> str:
    text = (raw or "").strip()
    fenced = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    fenced_any = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if fenced_any:
        return fenced_any.group(1).strip()
    return text



def _compact_sample_observation(sample_obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(sample_obs, dict):
        return {}
    return {
        "num_observed_samples": sample_obs.get("num_observed_samples"),
        "input_summary": sample_obs.get("input_summary"),
        "label_summary": sample_obs.get("label_summary"),
        "split_counts": sample_obs.get("split_counts"),
        "input_keys": sample_obs.get("input_keys"),
        "label_key": sample_obs.get("label_key"),
    }


def _sample_label_counts(sample_obs: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not isinstance(sample_obs, dict):
        return {}
    label_summary = sample_obs.get("label_summary") or {}
    counts = label_summary.get("label_counts") or {}
    out: Dict[str, int] = {}
    if isinstance(counts, dict):
        for k, v in counts.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
    return out


def _build_repair_diagnosis(
    *,
    stage: str,
    component: str,
    result: Optional[Dict[str, Any]] = None,
    trial: Optional[TrialResult] = None,
    sample_observation: Optional[Dict[str, Any]] = None,
    local_metrics: Optional[Dict[str, Any]] = None,
    edge_metrics: Optional[Dict[str, Any]] = None,
    artifact_manifest: Optional[Dict[str, Any]] = None,
    user_spec: Optional[UserSpec] = None,
) -> Dict[str, Any]:
    """Return a compact diagnosis that decides local patch vs next proposal.

    The classifier is deliberately small and evidence-first.  It does not repair
    code itself; it tells PipelineExecutor whether debugger patching is suitable
    and gives ProposalGenerator a concrete next code-space hint.
    """
    result = result or {}
    error = str(result.get("error") or "")
    stderr = str(result.get("stderr") or result.get("stderr_tail") or "")
    stdout = str(result.get("stdout") or result.get("stdout_tail") or "")
    primary_error_tail = str(result.get("primary_error_tail") or "")
    blob = f"{error}\n{stderr}\n{stdout}\n{primary_error_tail}".lower()
    sample_compact = _compact_sample_observation(sample_observation)
    label_counts = _sample_label_counts(sample_observation)
    diagnosis_type = "implementation_error"
    repair_action = "patch_current_script"
    next_hint = "Fix the current implementation error while preserving dataset evidence and artifact contracts."

    if not sample_observation and trial is not None:
        fc = trial.failure_context or {}
        loader_smoke = fc.get("loader_smoke") if isinstance(fc, dict) else None
        if isinstance(loader_smoke, dict):
            sample_observation = loader_smoke.get("sample_observation")
            sample_compact = _compact_sample_observation(sample_observation)
            label_counts = _sample_label_counts(sample_observation)

    # Loader semantics: a smoke may run but show no usable labels or suspicious constant labels.
    if stage in {"loader", "post_metric"} and sample_observation:
        unique = len(label_counts)
        if unique == 0:
            diagnosis_type = "loader_semantics_error"
            repair_action = "next_proposal_only"
            next_hint = "Regenerate loader.py around the real label source; smoke must return sample-level labels, not unlabeled batches."
        elif unique == 1:
            only_label = next(iter(label_counts.keys()))
            if only_label in {"16000", "44100", "48000"} or only_label.replace(".", "", 1).isdigit():
                diagnosis_type = "loader_semantics_error"
                repair_action = "next_proposal_only"
                next_hint = "Loader appears to use sampling rate or a numeric feature as label; mutate label extraction and representation before model tuning."

    # Runtime/API/shape failures are normally local implementation errors.
    if any(tok in blob for tok in ("shape", "mat1 and mat2", "size mismatch", "cannot be multiplied")):
        diagnosis_type = "implementation_error"
        repair_action = "patch_current_script"
        next_hint = "Patch tensor shape/input-rank handling; prefer adaptive pooling or derive dimensions from loader sample observation."
    if any(tok in blob for tok in ("no such file", "not found", "missing", "filenotfound")):
        diagnosis_type = "runtime_contract_error" if stage == "edge_benchmark" else "artifact_contract_error"
        repair_action = "patch_current_script"
        next_hint = "Patch artifact/path/payload contract without changing dataset semantics."
    if "external dataset" in blob or "load_dataset(" in blob or "stub" in blob or "fake" in blob:
        diagnosis_type = "insufficient_dataset_evidence"
        repair_action = "abort_current_trial"
        next_hint = "Reject fabricated fallback; regenerate from local dataset evidence only."

    # Metric-aware post-run diagnosis.
    metrics = {}
    if isinstance(local_metrics, dict):
        metrics.update(local_metrics)
    if isinstance(edge_metrics, dict):
        metrics.update(edge_metrics)
    metrics = MetricRegistry.normalize_metric_dict(metrics)
    if stage == "post_metric" and metrics:
        quality_zero = any(
            str(k).lower() in {"accuracy", "f1score", "f1", "auc", "miou"} and _safe_float(v) == 0.0
            for k, v in metrics.items()
        )
        if quality_zero:
            if len(label_counts) <= 1:
                diagnosis_type = "loader_semantics_error"
                repair_action = "next_proposal_only"
                next_hint = "Quality collapsed with suspicious/constant labels; next child should repair loader label semantics first."
            else:
                diagnosis_type = "metric_failure"
                repair_action = "next_proposal_only"
                next_hint = "Quality metric is zero despite multiple labels; inspect prediction distribution, class imbalance, threshold, and loss weighting."
        latency = _safe_float(metrics.get("Latency") or metrics.get("latency_ms"))
        memory = _safe_float(metrics.get("Memory_mb") or metrics.get("memory_mb"))
        if latency and latency > 1000:
            diagnosis_type = "representation_mismatch" if diagnosis_type == "implementation_error" else diagnosis_type
            repair_action = "next_proposal_only"
            next_hint = "Edge latency is dominated by representation/runtime path; next child should change export/runtime or avoid Python-heavy fallback."
        if memory and memory > 1024 and diagnosis_type == "implementation_error":
            diagnosis_type = "representation_mismatch"
            repair_action = "next_proposal_only"
            next_hint = "Memory is too high for edge; next child should reduce representation/model capacity or export route."
        if user_spec is not None:
            for constraint in user_spec.constraints or []:
                actual = MetricRegistry.lookup_value(constraint.metric, metrics)
                if actual is None:
                    continue
                actual_f = _safe_float(actual)
                target_f = _safe_float(constraint.target)
                if actual_f is None or target_f is None:
                    continue
                violated = (
                    (constraint.comparison == "gte" and actual_f < target_f)
                    or (constraint.comparison == "lte" and actual_f > target_f)
                    or (constraint.comparison == "eq" and abs(actual_f - target_f) > abs(target_f * 0.01))
                )
                if not violated:
                    continue
                metric_l = str(constraint.metric).lower()
                if "latency" in metric_l or "memory" in metric_l:
                    diagnosis_type = "representation_mismatch"
                    next_hint = (
                        f"{constraint.metric} violates target ({actual_f:.4g} vs {target_f:.4g}); "
                        "next child should change representation/export/runtime path rather than patch current code."
                    )
                else:
                    diagnosis_type = "metric_failure"
                    next_hint = (
                        f"{constraint.metric} violates target ({actual_f:.4g} vs {target_f:.4g}); "
                        "next child should repair representation, label handling, training recipe, or prediction distribution."
                    )
                repair_action = "next_proposal_only"
                break

    return {
        "repair_action": repair_action,
        "diagnosis_type": diagnosis_type,
        "failed_component": component,
        "stage": stage,
        "evidence": {
            "error": error[:300],
            "stderr_tail": stderr[-600:],
            "stdout_tail": stdout[-600:],
            "primary_error_tail": (
                primary_error_tail
                if len(primary_error_tail) <= 1200
                else primary_error_tail[:600] + "\n...\n" + primary_error_tail[-600:]
            ),
            "sample_observation": sample_compact,
            "local_metrics": local_metrics or {},
            "edge_metrics": edge_metrics or {},
            "artifact_manifest": artifact_manifest or {},
        },
        "next_search_hint": next_hint,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _artifact_execution_evidence(raw_metrics: Any) -> tuple[bool, list[str]]:
    """Reject canonical metrics only when the payload explicitly did not load its artifact."""
    if not isinstance(raw_metrics, dict):
        return True, []
    reasons: list[str] = []
    artifact_loaded = _safe_float(raw_metrics.get("artifact_loaded"))
    load_error_present = _safe_float(raw_metrics.get("load_error_present"))
    if artifact_loaded is not None and artifact_loaded <= 0:
        reasons.append("artifact_loaded=0")
    if load_error_present is not None and load_error_present > 0:
        reasons.append("load_error_present=1")
    return not reasons, reasons


def _normalize_training_trace(
    raw: Any,
    *,
    max_points: int = 32,
    result_envelope: Optional[Dict[str, Any]] = None,
) -> Optional[TrainingTrace]:
    """Validate compact train.py history without interpreting learning dynamics."""
    if isinstance(raw, list):
        raw = {"points": raw}
    if not isinstance(raw, dict):
        return None
    raw_points = raw.get("points") or raw.get("history") or raw.get("selection_trace") or []
    if not isinstance(raw_points, list):
        return None

    points: List[Dict[str, Any]] = []
    scalar_aliases = {
        "epoch_or_step": ("epoch_or_step",),
        "epoch": ("epoch",),
        "step": ("step",),
        "train_loss": ("train_loss",),
        "learning_rate": ("learning_rate", "lr"),
        "elapsed_s": ("elapsed_s", "elapsed_sec"),
    }
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        point: Dict[str, Any] = {}
        for key, aliases in scalar_aliases.items():
            value = next((item.get(alias) for alias in aliases if item.get(alias) is not None), None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                point[key] = float(value)
        validation = item.get("validation_metrics") or item.get("val_metrics") or {}
        normalized_validation = {
            str(key): float(value)
            for key, value in validation.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        } if isinstance(validation, dict) else {}
        normalized_validation.update({
            str(key)[4:]: float(value)
            for key, value in item.items()
            if str(key).startswith("val_")
            and len(str(key)) > 4
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and str(key)[4:] not in normalized_validation
        })
        if normalized_validation:
            point["validation_metrics"] = normalized_validation
        if point:
            points.append(point)

    if len(points) > max_points:
        indices = [round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
        points = [points[index] for index in indices]
    def _optional_count(name: str) -> Optional[int]:
        value = raw.get(name)
        if value is None and isinstance(raw.get("sample_counts"), dict):
            aliases = {"num_train": "train", "num_val": "val"}
            value = raw["sample_counts"].get(aliases.get(name, name))
        point_key = {"num_train": "num_train_samples", "num_val": "num_val_samples"}[name]
        if value is None:
            value = next(
                (
                    item.get(point_key)
                    for item in reversed(raw_points)
                    if isinstance(item, dict) and item.get(point_key) is not None
                ),
                None,
            )
        if value is None and isinstance(result_envelope, dict):
            resource = result_envelope.get("resource_summary") or {}
            resource_key = {"num_train": "train_samples", "num_val": "val_samples"}[name]
            value = resource.get(resource_key) if isinstance(resource, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0, int(value))
        return None

    best_step = raw.get("best_step")
    if best_step is None:
        best_step = raw.get("best_epoch")
    if best_step is None and isinstance(result_envelope, dict):
        best_step = result_envelope.get("best_step")
        if best_step is None:
            best_step = result_envelope.get("best_epoch")
    num_train = _optional_count("num_train")
    num_val = _optional_count("num_val")
    stopped_reason = str(
        raw.get("stopped_reason")
        or ((result_envelope or {}).get("stopped_reason") if isinstance(result_envelope, dict) else "")
        or ""
    )[:240]
    if not points and num_train is None and num_val is None and best_step is None and not stopped_reason:
        return None
    return TrainingTrace(
        points=points,
        num_train=num_train,
        num_val=num_val,
        best_step=float(best_step) if isinstance(best_step, (int, float)) and not isinstance(best_step, bool) else None,
        stopped_reason=stopped_reason,
    )

def _extract_debugger_decision(raw: str) -> str:
    text = raw or ""
    match = re.search(
        r"^\s*DECISION:\s*(implementation_error|plan_error|insufficient_dataset_evidence)\b",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).lower() if match else "implementation_error"


def _contains_blocked_transformers_export(script_text: str) -> bool:
    text = script_text or ""
    return ("transformers.onnx.export" in text) or ("from transformers.onnx import export" in text)


def _supports_export_only(script_text: str) -> bool:
    """Return whether a generated train script exposes the optional export retry."""
    return bool(re.search(r"['\"]--export-only['\"]", script_text or ""))


_TRAINING_EVIDENCE_KEYS = (
    "metrics",
    "training_trace",
    "best_epoch",
    "best_step",
    "training_seed",
    "pretrained_source",
    "stopped_reason",
    "dataset_meta",
    "resource_summary",
    "sample_counts",
)


def _training_evidence_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep training facts separate from a later artifact-only retry."""
    return {
        key: result[key]
        for key in _TRAINING_EVIDENCE_KEYS
        if key in result and result[key] not in (None, "", [], {})
    }


def _recover_training_trace_sidecar(
    result: Dict[str, Any],
    ws: Path,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Recover partial training evidence after a watchdog or script failure."""
    if result.get("training_trace") not in (None, "", [], {}):
        return result, None
    path = ws / "outputs" / "training_trace.json"
    try:
        if not path.is_file() or path.stat().st_size > 1_000_000:
            return result, None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return result, None
    if _normalize_training_trace(raw, result_envelope=result) is None:
        return result, None
    recovered = dict(result)
    recovered["training_trace"] = raw
    return recovered, {
        "path": str(path),
        "source": "watchdog_sidecar",
        "timed_out": bool(result.get("timed_out")),
    }


def _merge_export_retry_with_training_evidence(
    export_result: Dict[str, Any],
    training_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """Let export retry outputs replace artifacts, never prior training evidence."""
    return {**export_result, **training_evidence}


def _existing_train_checkpoints(ws: Path) -> List[str]:
    """Return real reusable training artifacts already present in a workspace."""
    suffixes = {".pt", ".pth", ".ckpt", ".joblib", ".pkl"}
    roots = [ws / "outputs", ws / "runs"]
    paths = {
        path.relative_to(ws).as_posix()
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }
    return sorted(paths)


_LOCAL_LOAD_DATASET_BUILDERS = {"csv", "json", "parquet", "text", "audiofolder", "imagefolder"}

def _uses_external_load_dataset(text: str) -> bool:
    for m in re.finditer(r"\bload_dataset\(\s*['\"]([^'\"]+)['\"]", text or ""):
        name = m.group(1).strip().lower()
        if name not in _LOCAL_LOAD_DATASET_BUILDERS:
            return True
    return False


def _provenance_guard_violation(script_text: str) -> Optional[str]:
    """Reject patches that fabricate or replace the user's local dataset."""
    text = script_text or ""
    if _uses_external_load_dataset(text):
        return "provenance guard: external load_dataset fallback is forbidden; use local load_from_disk paths"
    if re.search(r"\.to_csv\(\s*(csv_path|data_path|dataset_path|root\s*/)", text):
        return "provenance guard: writing CSV into dataset path is forbidden"
    return None


def _apply_oom_batch_fallback(script_path: Path) -> bool:
    """Halve common batch-size settings in train script after OOM."""
    text = script_path.read_text()
    original = text

    def _halve(match: re.Match) -> str:
        value = int(match.group(2))
        new_val = max(1, value // 2)
        return f"{match.group(1)}{new_val}"

    patterns = [
        r"(batch\s*=\s*)(\d+)",
        r"(per_device_train_batch_size\s*=\s*)(\d+)",
    ]
    changed = False
    for pat in patterns:
        new_text, count = re.subn(pat, _halve, text, count=1)
        if count > 0:
            text = new_text
            changed = True
            break
    if changed and text != original:
        script_path.write_text(text)
        return True
    return False


def _loader_smoke_count(payload: Dict[str, Any], *names: str) -> Optional[int]:
    """Read a small integer count from loader smoke JSON."""
    candidates = [payload]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        candidates.append(metrics)
    for obj in candidates:
        for name in names:
            value = obj.get(name)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _validate_loader_smoke_result(result: Dict[str, Any]) -> Optional[str]:
    """Return a contract error when smoke did not prove a trainable batch."""
    if result.get("status") != "success":
        return None
    num_train = _loader_smoke_count(
        result,
        "num_train",
        "train_count",
        "train_samples",
        "n_train",
        "num_samples",
        "samples",
    )
    if num_train is None:
        return (
            "loader smoke contract failed: success JSON must include "
            "num_train/train_count/train_samples or an equivalent sample count"
        )
    batch_size = _loader_smoke_count(result, "batch_size", "smoke_batch_size") or 1
    required = max(1, batch_size)
    if num_train < required:
        return (
            f"loader smoke contract failed: num_train={num_train} is smaller "
            f"than required trainable batch size {required}"
        )
    return None



def _loader_signature_map(loader_code: str) -> Dict[str, Dict[str, Any]]:
    """Return lightweight signatures for loader entrypoints without importing them."""
    try:
        tree = ast.parse(loader_code)
    except SyntaxError:
        return {}
    signatures: Dict[str, Dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in {"load_train_val", "load_test"}:
            continue
        positional = [arg.arg for arg in (node.args.posonlyargs + node.args.args)]
        keyword_only = [arg.arg for arg in node.args.kwonlyargs]
        signatures[node.name] = {
            "positional": positional,
            "keywords": set(positional + keyword_only),
            "has_vararg": node.args.vararg is not None,
            "has_varkw": node.args.kwarg is not None,
        }
    return signatures


def _positive_literal_sample_limit(value: ast.AST) -> Optional[int]:
    if isinstance(value, ast.Constant):
        raw = value.value
        if isinstance(raw, bool) or raw is None:
            return None
        if isinstance(raw, (int, float)) and raw > 0:
            return int(raw)
    return None


def _declared_sample_limits(train_code: str) -> List[Dict[str, Any]]:
    """Record literal loader sample limits without guessing their runtime role."""
    try:
        tree = ast.parse(train_code)
    except SyntaxError:
        return []
    limits: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if func_name not in {"load_train_val", "load_test"}:
            continue
        for keyword in node.keywords:
            if keyword.arg != "max_samples":
                continue
            limit = _positive_literal_sample_limit(keyword.value)
            if limit is not None:
                limits.append({"loader": func_name, "max_samples": limit, "line": node.lineno})
    return limits


def _validate_loader_train_call_contract(ws: Path) -> Optional[str]:
    """Fail fast when train.py calls loader.py with an incompatible API.

    This is a script-boundary contract, not a dataset adapter: loader.py owns how
    data is read; train.py must call the loader entrypoints using their actual
    signatures instead of guessing positional config/root arguments.
    """
    train_path = ws / "train.py"
    loader_path = ws / "loader.py"
    if not train_path.exists() or not loader_path.exists():
        return None
    try:
        train_code = train_path.read_text()
        loader_code = loader_path.read_text()
        tree = ast.parse(train_code)
    except (OSError, SyntaxError):
        return None
    signatures = _loader_signature_map(loader_code)
    if not signatures:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name not in signatures:
            continue
        sig = signatures[func_name]
        positional = sig.get("positional") or []
        allowed_keywords = sig.get("keywords") or set()
        if len(node.args) > len(positional) and not sig.get("has_vararg"):
            return (
                f"loader/train call contract failed: train.py calls {func_name}() "
                f"with {len(node.args)} positional argument(s), but loader.py defines "
                f"positional parameters {positional}"
            )
        first_param = positional[0] if positional else ""
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                literal = first_arg.value
                if literal.endswith((".yaml", ".yml")) and first_param not in {"config_path", "cfg_path", "data_config"}:
                    return (
                        f"loader/train call contract failed: train.py passes {literal!r} "
                        f"positionally to {func_name}(), but loader.py first parameter "
                        f"is {first_param!r}, not config_path"
                    )
        for kw in node.keywords:
            if kw.arg is None or sig.get("has_varkw"):
                continue
            if kw.arg not in allowed_keywords:
                return (
                    f"loader/train call contract failed: train.py passes unsupported "
                    f"keyword {kw.arg!r} to {func_name}(); loader.py accepts "
                    f"{sorted(allowed_keywords)}"
                )
    return None

def _validate_loader_train_interface(ws: Path) -> Optional[str]:
    """Return a contract error when train.py imports missing loader symbols."""
    train_path = ws / "train.py"
    loader_path = ws / "loader.py"
    if not train_path.exists():
        return None
    try:
        train_code = train_path.read_text()
    except OSError:
        return None

    required: set[str] = set()
    for match in re.finditer(r"^\s*from\s+loader\s+import\s+([^\n#]+)", train_code, flags=re.MULTILINE):
        imported = match.group(1)
        for name in imported.split(","):
            symbol = name.strip().split(" as ")[0].strip()
            if symbol in {"load_train_val", "load_test"}:
                required.add(symbol)
    if required and not loader_path.exists():
        return (
            "loader/train interface contract failed: train.py imports "
            f"{', '.join(sorted(required))} from loader.py, but loader.py is missing"
        )
    try:
        loader_code = loader_path.read_text()
    except OSError:
        return None
    missing = sorted(
        name for name in required
        if not re.search(rf"^\s*def\s+{re.escape(name)}\s*\(", loader_code, flags=re.MULTILINE)
    )
    if missing:
        return (
            "loader/train interface contract failed: train.py imports "
            f"{', '.join(missing)} from loader.py, but loader.py does not define it"
        )
    return None


def _loader_target_count(target: ast.AST) -> Optional[int]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return len(target.elts)
    if isinstance(target, ast.Name):
        return 1
    return None


def _load_train_val_unpack_counts(train_code: str) -> list[int]:
    try:
        tree = ast.parse(train_code)
    except SyntaxError:
        return []
    counts: list[int] = []
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name != "load_train_val":
            continue
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            count = _loader_target_count(target)
            if count and count > 1:
                counts.append(count)
    return sorted(set(counts))


def _probe_loader_return_summary(ws: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    probe = ws / ".edgecraft_loader_return_probe.py"
    probe.write_text('\nimport importlib.util\nimport json\nfrom collections import Counter\nfrom pathlib import Path\n\nLABEL_KEYS = ("label", "labels", "target", "y", "class", "category", "intent", "action", "anomaly", "is_anomaly")\n\nspec = importlib.util.spec_from_file_location(\'edgecraft_loader_probe\', \'loader.py\')\nmodule = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n\n\ndef safe_len(obj):\n    try:\n        return len(obj)\n    except Exception:\n        return None\n\n\ndef summarize(obj, depth=0):\n    if depth > 3:\n        return {\'type\': type(obj).__name__}\n    if hasattr(obj, \'shape\'):\n        summary = {\'type\': type(obj).__name__, \'shape\': [int(x) for x in list(getattr(obj, \'shape\', []))[:8]], \'dtype\': str(getattr(obj, \'dtype\', \'\'))}\n        columns = getattr(obj, \'columns\', None)\n        if columns is not None:\n            summary[\'columns\'] = [str(value) for value in list(columns)[:128]]\n        return summary\n    if isinstance(obj, dict):\n        return {\'type\': \'dict\', \'keys\': list(obj.keys())[:12], \'items\': {str(k): summarize(v, depth + 1) for k, v in list(obj.items())[:4]}}\n    if isinstance(obj, (list, tuple)):\n        return {\'type\': type(obj).__name__, \'len\': len(obj), \'items\': [summarize(x, depth + 1) for x in list(obj)[:4]]}\n    return {\'type\': type(obj).__name__, \'repr\': str(obj)[:160]}\n\n\ndef split_count_summary(loaded):\n    counts = {}\n    if isinstance(loaded, dict):\n        for key, value in loaded.items():\n            n = safe_len(value)\n            if n is not None:\n                counts[str(key)] = int(n)\n    elif isinstance(loaded, (list, tuple)):\n        names = [\'train_x\', \'train_y\', \'val_x\', \'val_y\'] if len(loaded) >= 4 else [\'train\', \'val\', \'test\']\n        for idx, value in enumerate(list(loaded)[:4 if len(loaded) >= 4 else 3]):\n            n = safe_len(value)\n            if n is not None:\n                counts[names[idx] if idx < len(names) else f\'part_{idx}\'] = int(n)\n    return counts\n\n\ndef choose_train_part(loaded):\n    if isinstance(loaded, dict):\n        for key in (\'train\', \'train_dataset\', \'train_split\', \'samples\', \'data\'):\n            if key in loaded:\n                return loaded[key]\n        return loaded\n    if isinstance(loaded, (list, tuple)) and len(loaded) >= 4:\n        first, second = loaded[0], loaded[1]\n        first_len = safe_len(first)\n        second_len = safe_len(second)\n        if first_len is not None and second_len is not None and first_len == second_len:\n            return (first, second)\n    if isinstance(loaded, (list, tuple)) and len(loaded) >= 2:\n        first, second = loaded[0], loaded[1]\n        if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):\n            first_len = safe_len(first)\n            second_len = safe_len(second)\n            if first_len is not None and second_len is not None and first_len == second_len and first_len > 4:\n                return loaded\n        return first\n    return loaded\n\n\ndef normalize_samples(part, limit=8):\n    samples = []\n    if isinstance(part, dict):\n        label_key = next((k for k in part.keys() if str(k).lower() in LABEL_KEYS), None)\n        if label_key is not None and safe_len(part.get(label_key)):\n            n = min(limit, safe_len(part[label_key]) or 0)\n            for i in range(n):\n                x = {k: (v[i] if safe_len(v) and safe_len(v) > i else v) for k, v in part.items() if k != label_key}\n                samples.append((x, part[label_key][i]))\n            return samples\n        return [(dict([item]), None) for item in list(part.items())[:limit]]\n    if isinstance(part, (list, tuple)) and len(part) == 2:\n        xseq, yseq = part\n        lx, ly = safe_len(xseq), safe_len(yseq)\n        if lx is not None and ly is not None and lx == ly:\n            return [(xseq[i], yseq[i]) for i in range(min(limit, lx))]\n    if isinstance(part, (list, tuple)):\n        for item in list(part)[:limit]:\n            if isinstance(item, dict):\n                label_key = next((k for k in item.keys() if str(k).lower() in LABEL_KEYS), None)\n                if label_key is not None:\n                    samples.append(({k: v for k, v in item.items() if k != label_key}, item.get(label_key)))\n                else:\n                    samples.append((item, None))\n            elif isinstance(item, (list, tuple)) and len(item) >= 2:\n                samples.append((item[0], item[1]))\n            else:\n                samples.append((item, None))\n        return samples\n    return [(part, None)]\n\n\ndef build_sample_observation(loaded):\n    part = choose_train_part(loaded)\n    samples = normalize_samples(part, limit=8)\n    labels = [y for _, y in samples if y is not None]\n    label_counts = Counter(str(y) for y in labels)\n    examples = []\n    for x, y in samples[:5]:\n        examples.append({\'input\': summarize(x), \'label\': summarize(y) if y is not None else None})\n    first_x = samples[0][0] if samples else None\n    return {\n        \'status\': \'success\',\n        \'num_observed_samples\': len(samples),\n        \'input_summary\': summarize(first_x) if first_x is not None else {},\n        \'label_summary\': {\n            \'num_labels_observed\': len(labels),\n            \'unique_labels_observed\': len(label_counts),\n            \'label_counts\': dict(label_counts.most_common(20)),\n        },\n        \'split_counts\': split_count_summary(loaded),\n        \'examples\': examples,\n    }\n\ntry:\n    loaded = module.load_train_val(max_samples=128)\nexcept TypeError:\n    loaded = module.load_train_val(\'config/data.yaml\', max_samples=128)\ncount = len(loaded) if isinstance(loaded, (tuple, list)) else 1\nobservation = build_sample_observation(loaded)\nPath(\'outputs\').mkdir(exist_ok=True)\nPath(\'outputs/sample_observation.json\').write_text(json.dumps(observation, indent=2, ensure_ascii=False))\nprint(json.dumps({\'status\': \'success\', \'top_level_count\': count, \'return_summary\': summarize(loaded), \'sample_observation\': observation}))\n')
    source = probe.read_text()
    probe.write_text(source.replace(
        "max_samples=128",
        "max_samples=None",
    ).replace(
        "return [(xseq[i], yseq[i]) for i in range(min(limit, lx))]",
        "indices = range(lx) if lx <= limit else "
        "[round(i * (lx - 1) / (limit - 1)) for i in range(limit)]\n"
        "            return [((xseq.iloc[i] if hasattr(xseq, 'iloc') else xseq[i]), "
        "(yseq.iloc[i] if hasattr(yseq, 'iloc') else yseq[i])) for i in indices]",
    ).replace(
        "for item in list(part)[:limit]:",
        "items = list(part)\n"
        "        indices = range(len(items)) if len(items) <= limit else "
        "[round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]\n"
        "        for index in indices:\n"
        "            item = items[index]",
    ).replace(
        "'keys': list(obj.keys())[:12], 'items':",
        "'keys': [str(k) for k in list(obj.keys())[:64]], 'key_count': len(obj), 'items':",
    ).replace(
        "[summarize(x, depth + 1) for x in list(obj)[:4]]",
        "[summarize(x, depth + 1) for x in list(obj)[:128]]",
    ))
    source = probe.read_text()
    source = source.replace(
        "    elif isinstance(loaded, (list, tuple)):\n"
        "        names = ['train_x', 'train_y', 'val_x', 'val_y'] if len(loaded) >= 4 else ['train', 'val', 'test']\n",
        "    elif isinstance(loaded, (list, tuple)):\n"
        "        names = ['train', 'val', 'test']\n"
        "        if len(loaded) >= 4:\n"
        "            lengths = [safe_len(value) for value in loaded[:4]]\n"
        "            if lengths[0] == lengths[1] and lengths[2] == lengths[3]:\n"
        "                names = ['train_x', 'train_y', 'val_x', 'val_y']\n"
        "            elif lengths[0] == lengths[2] and lengths[1] == lengths[3]:\n"
        "                names = ['train_x', 'val_x', 'train_y', 'val_y']\n"
        "            else:\n"
        "                names = ['part_0', 'part_1', 'part_2', 'part_3']\n",
    )
    source = source.replace(
        "    if isinstance(loaded, (list, tuple)) and len(loaded) >= 4:\n"
        "        first, second = loaded[0], loaded[1]\n"
        "        first_len = safe_len(first)\n"
        "        second_len = safe_len(second)\n"
        "        if first_len is not None and second_len is not None and first_len == second_len:\n"
        "            return (first, second)\n",
        "    if isinstance(loaded, (list, tuple)) and len(loaded) >= 4:\n"
        "        lengths = [safe_len(value) for value in loaded[:4]]\n"
        "        if lengths[0] is not None and lengths[0] == lengths[1] and lengths[2] == lengths[3]:\n"
        "            return (loaded[0], loaded[1])\n"
        "        if lengths[0] is not None and lengths[0] == lengths[2] and lengths[1] == lengths[3]:\n"
        "            return (loaded[0], loaded[2])\n",
    )
    probe.write_text(source)
    try:
        result = _run_script(
            probe,
            ws,
            timeout=int(os.getenv("EDGECRAFT_LOADER_RETURN_PROBE_TIMEOUT_S", "30")),
            stream_output=False,
        )
    finally:
        try:
            move_to_trash(probe)
        except OSError:
            pass
    if result.get("status") != "success":
        error = str(result.get("error") or "loader return probe failed")
        stderr_tail = str(result.get("stderr") or "").strip()[-1200:]
        if stderr_tail:
            error = f"{error}; stderr: {stderr_tail}"
        return None, error
    observation = dict(result.get("sample_observation") or {})
    observation.setdefault("sampling_strategy", "deterministic_spread_over_loader_return")
    observation.setdefault("population_representativeness", "bounded_observation")
    return {
        "top_level_count": result.get("top_level_count"),
        "return_summary": result.get("return_summary"),
        "sample_observation": observation,
    }, None

def _probe_loader_return_count(ws: Path) -> tuple[Optional[int], Optional[str]]:
    summary, error = _probe_loader_return_summary(ws)
    if error:
        return None, error
    raw = (summary or {}).get("top_level_count")
    try:
        return int(raw), None
    except (TypeError, ValueError):
        return None, "loader return probe did not report top_level_count"


def _probe_loader_test_membership(
    ws: Path,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Check that load_test() reads the evaluator-owned frozen test split.

    This probe inspects identifiers only. It does not expose test labels or
    metrics to the search loop, and it does not encode dataset-specific paths.
    """
    probe = ws / ".edgecraft_loader_test_membership.py"
    probe.write_text(
        '''
import importlib.util
import json
from pathlib import Path

import yaml


def as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(item) for item in list(value)]


def identifiers(value):
    if hasattr(value, "columns"):
        columns = {str(column): column for column in value.columns}
        sample_key = "__sample_id__" if "__sample_id__" in columns else "sample_id"
        samples = as_list(value[columns[sample_key]]) if sample_key in columns else []
        sources = as_list(value[columns["source_id"]]) if "source_id" in columns else []
        return samples, sources
    if isinstance(value, dict):
        samples = as_list(value.get("sample_ids"))
        sources = as_list(value.get("source_ids"))
        if samples:
            return samples, sources
        if value.get("sample_id") is not None:
            return [str(value["sample_id"])], [str(value.get("source_id") or "")]
        for key in ("test", "records", "samples", "data"):
            if key in value:
                found = identifiers(value[key])
                if found[0]:
                    return found
        return [], []
    if isinstance(value, tuple):
        for item in value:
            found = identifiers(item)
            if found[0]:
                return found
        return [], []
    if isinstance(value, list):
        samples = []
        sources = []
        for item in value:
            found_samples, found_sources = identifiers(item)
            samples.extend(found_samples)
            sources.extend(found_sources or [""] * len(found_samples))
        return samples, sources
    return [], []


cfg_path = Path("config/data.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
split = cfg.get("split_manifest") or {}
manifest_path = Path(str(split.get("path") or ""))
if not manifest_path.is_absolute():
    manifest_path = (cfg_path.parent / manifest_path).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {str(item) for item in ((manifest.get("splits") or {}).get("test") or [])}
if not expected:
    raise ValueError("frozen split manifest has no test IDs")

spec = importlib.util.spec_from_file_location("edgecraft_test_loader", "loader.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    loaded = module.load_test(config_path=str(cfg_path), max_samples=64)
except TypeError:
    loaded = module.load_test(str(cfg_path), max_samples=64)
sample_ids, source_ids = identifiers(loaded)
if not sample_ids:
    raise ValueError("load_test() did not expose sample_id/source_id identifiers")
sample_outside = sorted(set(sample_ids) - expected)
nonempty_sources = {value for value in source_ids if value}
source_outside = sorted(nonempty_sources - expected)
sample_match = not sample_outside
source_match = bool(nonempty_sources) and not source_outside
if not sample_match and not source_match:
    raise ValueError(
        "load_test() identifiers do not belong to frozen test split; "
        f"sample_examples={sample_outside[:5]}, source_examples={source_outside[:5]}"
    )
print(json.dumps({
    "status": "success",
    "match_mode": "sample_id" if sample_match else "source_id",
    "checked_samples": len(sample_ids),
    "expected_test_ids": len(expected),
    "sample_id_examples": sample_ids[:5],
    "source_id_examples": source_ids[:5],
}))
''',
        encoding="utf-8",
    )
    try:
        result = _run_script(
            probe,
            ws,
            timeout=int(os.getenv("EDGECRAFT_LOADER_TEST_PROBE_TIMEOUT_S", "30")),
            stream_output=False,
        )
    finally:
        try:
            move_to_trash(probe)
        except OSError:
            pass
    if result.get("status") != "success":
        error = str(result.get("error") or "loader test membership probe failed")
        stderr_tail = str(result.get("stderr") or "").strip()[-1200:]
        if stderr_tail:
            error = f"{error}; stderr: {stderr_tail}"
        return None, error
    return {
        "match_mode": result.get("match_mode"),
        "checked_samples": result.get("checked_samples"),
        "expected_test_ids": result.get("expected_test_ids"),
        "sample_id_examples": result.get("sample_id_examples") or [],
        "source_id_examples": result.get("source_id_examples") or [],
    }, None


def _literal_mapping_keys(node: ast.AST, variable: str) -> set[str]:
    """Return mapping keys whose absence would raise at runtime.

    ``mapping.get("key", default)`` is deliberately excluded: it is an
    optional read, while ``mapping["key"]`` is a required contract.
    """
    keys: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            if child.value.id == variable:
                key = child.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


def _function_mapping_requirements(tree: ast.AST) -> Dict[str, Dict[int, set[str]]]:
    requirements: Dict[str, Dict[int, set[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = list(node.args.posonlyargs) + list(node.args.args)
        for index, arg in enumerate(positional):
            keys = _literal_mapping_keys(node, arg.arg)
            if keys:
                requirements.setdefault(node.name, {})[index] = keys
    return requirements


def _return_item_summary(
    return_summary: Dict[str, Any], target: ast.AST
) -> Dict[str, Dict[str, Any]]:
    if isinstance(target, ast.Name):
        return {target.id: return_summary}
    if not isinstance(target, (ast.Tuple, ast.List)):
        return {}
    items = return_summary.get("items") or []
    result: Dict[str, Dict[str, Any]] = {}
    for index, element in enumerate(target.elts):
        if isinstance(element, ast.Name) and index < len(items) and isinstance(items[index], dict):
            result[element.id] = items[index]
    return result


def _loader_result_bindings(
    tree: ast.AST, return_summary: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    bindings: Dict[str, Dict[str, Any]] = {}
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if func_name != "load_train_val":
            continue
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            bindings.update(_return_item_summary(return_summary, target))
    return bindings


def _required_loader_result_keys(
    tree: ast.AST, bindings: Dict[str, Dict[str, Any]]
) -> Dict[str, set[str]]:
    required = {name: _literal_mapping_keys(tree, name) for name in bindings}
    function_requirements = _function_mapping_requirements(tree)
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        func = call.func
        func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        for index, keys in function_requirements.get(func_name, {}).items():
            if index >= len(call.args) or not isinstance(call.args[index], ast.Name):
                continue
            name = call.args[index].id
            if name in bindings:
                required.setdefault(name, set()).update(keys)
    return required


def _mapping_contract_error(
    train_code: str, return_summary: Dict[str, Any]
) -> Optional[str]:
    try:
        tree = ast.parse(train_code)
    except SyntaxError:
        return None
    bindings = _loader_result_bindings(tree, return_summary)
    required = _required_loader_result_keys(tree, bindings)
    for name, keys in sorted(required.items()):
        summary = bindings.get(name) or {}
        if summary.get("type") != "dict" or not keys:
            continue
        observed = {str(key) for key in (summary.get("keys") or [])}
        key_count = summary.get("key_count")
        if isinstance(key_count, int) and key_count > len(observed):
            continue
        missing = sorted(keys - observed)
        if missing:
            return (
                "loader/train mapping contract failed: train.py reads "
                f"{name} keys {sorted(keys)}, but sampled loader output exposes "
                f"keys {sorted(observed)}; missing {missing}"
            )
    return None


def _validate_loader_train_return_contract(
    ws: Path,
    probe_summary: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    train_path = ws / "train.py"
    loader_path = ws / "loader.py"
    if not train_path.exists() or not loader_path.exists():
        return None
    try:
        train_code = train_path.read_text()
    except OSError:
        return None
    summary = probe_summary
    if summary is None:
        summary, error = _probe_loader_return_summary(ws)
        if error:
            return f"loader/train return contract probe failed: {error}"
    expected_counts = _load_train_val_unpack_counts(train_code)
    actual_count = (summary or {}).get("top_level_count")
    if expected_counts and actual_count not in expected_counts:
        return (
            "loader/train return contract failed: train.py unpacks "
            f"load_train_val() into {expected_counts} top-level values, "
            f"but loader.py returns {actual_count}"
        )
    return _mapping_contract_error(train_code, (summary or {}).get("return_summary") or {})


def _record_loader_train_contract_evidence(
    trial: TrialResult,
    ws: Path,
    probe_summary: Optional[Dict[str, Any]],
    probe_error: Optional[str] = None,
) -> tuple[Optional[str], Evidence]:
    if probe_summary is None and probe_error is None:
        probe_summary, probe_error = _probe_loader_return_summary(ws)
    error = _validate_loader_train_contract_from_evidence(ws, probe_summary)
    try:
        declared_sample_limits = _declared_sample_limits((ws / "train.py").read_text())
    except OSError:
        declared_sample_limits = []
    outcome = "fail" if error else "unknown" if probe_error else "pass"
    evidence = Evidence(
        probe_id="component_roundtrip",
        quantity="component_contract",
        fidelity="proxy",
        outcome=outcome,
        protocol={
            "boundary": "loader_to_train",
            "real_dataset_sample": bool(probe_summary),
            "loader_return_summary": (probe_summary or {}).get("return_summary"),
            "declared_sample_limits": declared_sample_limits,
            "error": error or probe_error or "",
            "decision_authority": "prompt_only",
        },
    )
    _append_evidence(trial, evidence)
    return error, evidence


def _record_training_data_coverage(
    trial: TrialResult,
    state: AgentState,
    trace: Optional[TrainingTrace],
) -> Evidence:
    split_info = ((state.get("dataset_info") or {}).get("split_manifest") or {})
    expected: Dict[str, int] = {}
    manifest_path = split_info.get("path") if isinstance(split_info, dict) else None
    if manifest_path:
        try:
            payload = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
            splits = payload.get("splits") or {}
            expected = {
                name: len(items)
                for name, items in splits.items()
                if name in {"train", "val", "validation"} and isinstance(items, list)
            }
        except (OSError, ValueError, yaml.YAMLError):
            expected = {}
    observed = {
        "train": trace.num_train if trace is not None else None,
        "val": trace.num_val if trace is not None else None,
    }
    expected_train = expected.get("train")
    expected_val = expected.get("val", expected.get("validation"))
    comparable = (
        expected_train is not None
        and expected_val is not None
        and observed["train"] is not None
        and observed["val"] is not None
    )
    covered = bool(
        comparable
        and observed["train"] >= expected_train
        and observed["val"] >= expected_val
    )
    evidence = Evidence(
        probe_id="full_train_trace",
        quantity="training_data_coverage",
        fidelity="proxy",
        outcome="pass" if covered else "fail" if comparable else "unknown",
        protocol={
            "expected": {"train": expected_train, "val": expected_val},
            "observed": observed,
            "split_manifest_hash": split_info.get("content_hash") if isinstance(split_info, dict) else None,
            "decision_authority": "prompt_and_report_only",
        },
    )
    _append_evidence(trial, evidence)
    return evidence


def _record_training_result_evidence(
    trial: TrialResult,
    state: AgentState,
    result: Dict[str, Any],
) -> Dict[str, float]:
    """Persist completed training facts even when artifact validation fails."""
    metrics_raw = result.get("metrics", {})
    metrics = MetricRegistry.normalize_metric_dict(
        metrics_raw if isinstance(metrics_raw, dict) else {}
    )
    trial.local_metrics = LocalMetrics(all_metrics=metrics)
    trial.training_seed = _confirmed_training_seed(result)
    trace = _normalize_training_trace(
        result.get("training_trace"),
        result_envelope=result,
    )
    if trace is not None:
        trial.training_trace = trace
        trace_evidence = Evidence(
            probe_id="full_train_trace",
            quantity="training_dynamics",
            fidelity="proxy",
            outcome="pass",
            protocol={
                "point_count": len(trace.points),
                "num_train": trace.num_train,
                "num_val": trace.num_val,
                "best_step": trace.best_step,
                "stopped_reason": trace.stopped_reason,
                "last_point": trace.points[-1] if trace.points else None,
                "decision_authority": "prompt_only",
            },
        )
        _append_evidence(trial, trace_evidence)
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["training_trace_evidence_id"] = trace_evidence.id
        trial.observations = observations

    coverage_evidence = _record_training_data_coverage(trial, state, trace)
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["training_data_coverage_evidence_id"] = coverage_evidence.id
    trial.observations = observations

    pretrained_source = result.get("pretrained_source")
    if isinstance(pretrained_source, dict):
        model_id = str(pretrained_source.get("model_id") or "")[:300]
        revision = str(
            pretrained_source.get("revision")
            or pretrained_source.get("checkpoint")
            or ""
        )[:300]
        if model_id:
            observations = dict(getattr(trial, "observations", {}) or {})
            observations["pretrained_source"] = {
                "model_id": model_id,
                "revision": revision,
                "pinned": revision.lower()
                not in {"", "default", "latest", "main", "master"},
            }
            trial.observations = observations
    return metrics


def _observed_loader_schema(return_summary: Any) -> Dict[str, Any]:
    """Collect executed loader schemas without assigning them semantic roles."""
    tables: list[Dict[str, Any]] = []
    string_lists: list[Dict[str, Any]] = []
    pending: list[tuple[str, Any]] = [("return", return_summary)]
    while pending:
        path, node = pending.pop()
        if not isinstance(node, dict):
            continue
        columns = node.get("columns")
        if isinstance(columns, list):
            tables.append({
                "path": path,
                "columns": [str(column) for column in columns],
            })
        items = node.get("items")
        if isinstance(items, list):
            values = [
                item.get("repr")
                for item in items
                if isinstance(item, dict) and item.get("type") == "str"
            ]
            if values and len(values) == len(items):
                string_lists.append({
                    "path": path,
                    "values": values,
                    "observed_count": len(values),
                    "total_count": node.get("len", len(values)),
                })
            pending.extend(
                (f"{path}[{index}]", item)
                for index, item in enumerate(items)
            )
        elif isinstance(items, dict):
            pending.extend(
                (f"{path}.{key}", item)
                for key, item in items.items()
            )
    observed: Dict[str, Any] = {}
    if tables:
        observed["observed_table_columns"] = sorted(
            tables, key=lambda item: item["path"]
        )
    if string_lists:
        observed["observed_string_lists"] = sorted(
            string_lists, key=lambda item: item["path"]
        )
    return observed


def _record_loader_feature_provenance(
    trial: TrialResult,
    result: Dict[str, Any],
) -> Optional[Evidence]:
    """Preserve a loader's declared feature boundary without interpreting it."""
    fields: Dict[str, Any] = {}
    for name in ("feature_columns", "excluded_columns", "target_columns"):
        value = result.get(name)
        if isinstance(value, list):
            fields[name] = [str(item) for item in value]
    if "feature_columns" not in fields:
        grouped_features = [
            str(item)
            for name in ("categorical_columns", "categorical_cols", "numeric_columns", "numeric_cols")
            for item in (result.get(name) if isinstance(result.get(name), list) else [])
        ]
        if grouped_features:
            fields["feature_columns"] = list(dict.fromkeys(grouped_features))
    fields.update(_observed_loader_schema(
        (result.get("loader_return_probe") or {}).get("return_summary")
    ))
    if not fields:
        return None
    evidence = Evidence(
        probe_id="loader_feature_provenance",
        quantity="feature_provenance",
        fidelity="proxy",
        outcome="pass",
        protocol={
            **fields,
            "decision_authority": "prompt_and_report_only",
        },
    )
    _append_evidence(trial, evidence)
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["feature_provenance"] = {
        **fields,
        "evidence_id": evidence.id,
    }
    trial.observations = observations
    return evidence


def _validate_loader_train_contract_from_evidence(
    ws: Path,
    probe_summary: Optional[Dict[str, Any]],
) -> Optional[str]:
    error = _validate_loader_train_interface(ws) or _validate_loader_train_call_contract(ws)
    if error is None and probe_summary is not None:
        error = _validate_loader_train_return_contract(ws, probe_summary)
    return error


def _select_generated_failure_component(
    ws: Path,
    result: Dict[str, Any],
    *,
    default: Path,
    candidate_names: tuple[str, ...] = ("loader.py", "train.py"),
) -> Path:
    """Return the last generated component named by the failure traceback."""
    failure_text = "\n".join(
        str(result.get(field) or "") for field in ("error", "stderr", "stdout")
    )
    referenced: list[tuple[int, Path]] = []
    for name in candidate_names:
        path = ws / name
        position = failure_text.rfind(name)
        if path.exists() and position >= 0:
            referenced.append((position, path))
    if not referenced:
        return default
    return max(referenced, key=lambda item: item[0])[1]


def _loader_smoke_args(loader_script: Path, ws: Path) -> List[str]:
    """Use the materialized data config when the loader declares that interface."""
    args = ["--smoke"]
    config_path = ws / "config" / "data.yaml"
    if not config_path.is_file():
        return args
    source = loader_script.read_text(encoding="utf-8", errors="ignore")
    if "--config" not in source:
        return args
    return [*args, "--config", str(config_path.relative_to(ws))]


def _run_loader_smoke_with_debugger(
    *,
    loader_script: Path,
    ws: Path,
    state: AgentState,
    variant: SolutionVariant,
    trial: TrialResult,
    max_debug_retries: int,
) -> Dict[str, Any]:
    """Run optional loader.py --smoke and repair loader.py only on failure."""
    if os.getenv("EDGECRAFT_DISABLE_LOADER_SMOKE", "").strip().lower() in {"1", "true", "yes"}:
        return {"status": "success", "skipped": True, "reason": "EDGECRAFT_DISABLE_LOADER_SMOKE"}

    result: Dict[str, Any] = {}
    for retry_idx in range(max_debug_retries + 1):
        interface_error = _validate_loader_train_interface(ws)
        smoke_started = time.time()
        if interface_error:
            result = {"status": "error", "error": interface_error, "stdout": "", "stderr": ""}
        else:
            result = _run_script(
                loader_script,
                ws,
                timeout=int(
                    os.getenv(
                        "EDGECRAFT_LOADER_SMOKE_TIMEOUT_S",
                        str(getattr(settings, "LOADER_STAGE_TIMEOUT_S", 300)),
                    )
                ),
                script_args=_loader_smoke_args(loader_script, ws),
            )
        result.setdefault("elapsed_s", round(time.time() - smoke_started, 3))
        if result.get("status") == "success":
            probe_summary, probe_error = _probe_loader_return_summary(ws)
            if probe_summary:
                result["loader_return_probe"] = probe_summary
            elif probe_error:
                result["loader_return_probe_error"] = probe_error
            contract_error = _validate_loader_smoke_result(result)
            if (
                contract_error is None
                and bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
            ):
                membership, membership_error = _probe_loader_test_membership(ws)
                if membership:
                    result["loader_test_membership"] = membership
                elif membership_error:
                    contract_error = (
                        "loader frozen-test contract failed: " + membership_error
                    )
            if contract_error is None:
                _mark_debug_retry_success(trial, "loader")
                return result
            result["status"] = "error"
            result["error"] = contract_error

        err_msg = "loader smoke failed: " + str(result.get("error") or "loader.py --smoke failed")
        is_repairable, signature = _is_repairable_failure(err_msg, result.get("stderr", ""))
        if retry_idx >= max_debug_retries or not is_repairable:
            result["error"] = err_msg
            return result

        diagnostics = _collect_failure_diagnostics(
            ws=ws,
            state=state,
            stage="loader",
            result=result,
        )
        _record_trial_observations(trial, diagnostics=diagnostics)
        patched, summary, before_hash, after_hash, reason = _run_llm_debugger(
            script_path=loader_script,
            stage="loader",
            variant=variant,
            error=err_msg,
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            error_signature=signature,
            diagnostic_context=diagnostics,
            debug_history=trial.debug_history,
        )
        trial.debug_attempts += 1
        trial.debug_history.append(
            DebugAttempt(
                stage="loader",
                attempt=trial.debug_attempts,
                error_signature=signature,
                reason=reason,
                before_hash=before_hash,
                after_hash=after_hash,
                patch_summary=summary,
                patch_applied=patched,
                success=False,
            )
        )
        if not patched:
            result["error"] = err_msg
            return result
        logger.info(
            f"[{variant.trial_id}] Debugger patched loader.py "
            f"(attempt {retry_idx + 1}/{max_debug_retries}). Retrying."
        )
    return result


def _train_export_smoke_check(
    ws: Path,
    train_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Hard gate after training: ONNX smoke check + contract validation."""
    stdout = train_result.get("stdout", "")
    stderr = train_result.get("stderr", "")

    def fail(error: str) -> Dict[str, Any]:
        return {
            **train_result,
            "status": "error",
            "error": error,
            "stdout": stdout,
            "stderr": stderr,
        }

    expected = ws / "outputs" / "best.onnx"
    pt_fallback = ws / "outputs" / "best.pt"
    reported_model = str(train_result.get("model_path") or "")
    has_pt_fallback = pt_fallback.exists() and (
        reported_model.endswith(".pt") or not reported_model
    )
    infer_path = ws / "infer.py"
    infer_uses_onnx = False
    if infer_path.exists():
        try:
            infer_lower = infer_path.read_text(encoding="utf-8").lower()
            infer_uses_onnx = (
                "onnxruntime" in infer_lower
                and "inferencesession" in infer_lower
                and "outputs/best.onnx" in infer_lower
            )
        except OSError:
            infer_uses_onnx = False
    if infer_uses_onnx:
        try:
            train_lower = (ws / "train.py").read_text(encoding="utf-8").lower()
        except OSError:
            train_lower = ""
        surrogate_onnx_export = (
            ("joblib.dump" in train_lower or "sklearn" in train_lower)
            and any(
                marker in train_lower
                for marker in (
                    "dummylinear",
                    "dummy linear",
                    "pipeline-like model",
                    "fallback torch.onnx.export",
                    "fallback torch-onnx",
                )
            )
        )
        if surrogate_onnx_export:
            return fail(
                "export smoke failed: infer.py uses outputs/best.onnx but train.py "
                "creates a dummy/surrogate ONNX instead of exporting the trained model"
            )

    if not expected.exists():
        if has_pt_fallback and not infer_uses_onnx:
            logger.info("export smoke: outputs/best.onnx missing; using outputs/best.pt fallback")
            return train_result
        if has_pt_fallback and infer_uses_onnx:
            return fail(
                "export smoke failed: infer.py uses outputs/best.onnx but train.py produced only outputs/best.pt"
            )
        return fail("export smoke failed: outputs/best.onnx missing after train")
    if expected.stat().st_size <= 0:
        if has_pt_fallback and not infer_uses_onnx:
            move_to_trash(expected)
            logger.info("export smoke: removed empty outputs/best.onnx; using outputs/best.pt fallback")
            return train_result
        if has_pt_fallback and infer_uses_onnx:
            move_to_trash(expected)
            return fail(
                "export smoke failed: infer.py uses outputs/best.onnx but outputs/best.onnx is empty"
            )
        return fail("export smoke failed: outputs/best.onnx is empty")
    try:
        import onnx  # type: ignore

        model_proto = onnx.load(str(expected))
        onnx.checker.check_model(model_proto)
    except ImportError:
        # Keep hard gate on file existence even when onnx package is unavailable.
        return train_result
    except Exception as exc:
        if has_pt_fallback and not infer_uses_onnx:
            move_to_trash(expected)
            logger.info(
                f"export smoke: removed invalid outputs/best.onnx ({exc}); "
                "using outputs/best.pt fallback"
            )
            return train_result
        if has_pt_fallback and infer_uses_onnx:
            move_to_trash(expected)
            return fail(
                "export smoke failed: infer.py uses outputs/best.onnx but "
                f"outputs/best.onnx is invalid: {exc}"
            )
        return fail(f"export smoke failed: invalid onnx: {exc}")
    return train_result


def _get_debugger_llm():
    if not settings.DEBUGGER_ENABLED:
        return None
    if not llm_credentials_available():
        return None
    return create_chat_llm(temperature=0.0, purpose="debugger")


def _mark_debug_retry_success(trial: TrialResult, stage: str) -> None:
    """Mark the last applied patch for a stage as having passed on retry."""
    for attempt in reversed(trial.debug_history):
        if attempt.stage == stage and getattr(attempt, "patch_applied", False) and not attempt.success:
            attempt.success = True
            return


def _format_recent_debug_history(debug_history: Optional[List[DebugAttempt]], limit: int = 3) -> str:
    """Compact prior repair attempts for the same trial into debugger context."""
    if not debug_history:
        return "(none)"
    lines: list[str] = []
    for attempt in list(debug_history)[-limit:]:
        if attempt.success:
            status = "retry_passed"
        elif getattr(attempt, "patch_applied", False):
            status = "patch_applied_retry_failed_or_pending"
        else:
            status = "not_applied"
        summary = (attempt.patch_summary or "").replace("\n", " ").strip()
        if len(summary) > 240:
            summary = summary[:237] + "..."
        lines.append(
            f"- attempt={attempt.attempt} stage={attempt.stage} "
            f"signature={attempt.error_signature or 'unknown'} status={status} "
            f"reason={attempt.reason or 'unknown'} summary={summary or '(missing)'}"
        )
    return "\n".join(lines)


def _format_debugger_diagnostics(
    diagnostics: Optional[Dict[str, Any]],
    *,
    limit: int = 7000,
) -> str:
    """Keep executable contracts ahead of verbose observational context."""
    source = diagnostics or {}
    priority = (
        "config_data_yaml_excerpt",
        "layout_contract",
        "dataset_signature",
        "split_manifest",
        "error",
        "return_code",
    )
    ordered = {key: source[key] for key in priority if key in source}
    ordered.update({key: value for key, value in source.items() if key not in ordered})
    return json.dumps(ordered, indent=2, ensure_ascii=False)[:limit]


def _run_llm_debugger(
    script_path: Path,
    stage: str,
    variant: SolutionVariant,
    error: str,
    stdout: str,
    stderr: str,
    error_signature: str,
    diagnostic_context: Optional[Dict[str, Any]] = None,
    debug_history: Optional[List[DebugAttempt]] = None,
) -> tuple[bool, str, str, str, str]:
    """Patch a failed script with LLM debugger.

    Returns:
        (patched, summary, before_hash, after_hash, reason)
    """
    original = script_path.read_text()
    before_hash = _sha256_text(original)
    llm = _get_debugger_llm()
    reason = f"signature={error_signature}"
    if llm is None:
        return False, "debugger disabled or missing API key", before_hash, before_hash, reason

    recent_debug_history = _format_recent_debug_history(debug_history)
    reusable_checkpoints = _existing_train_checkpoints(script_path.parent) if stage == "train" else []

    family = (variant.model_family or "").lower()
    if family == "ultralytics":
        family_rules = """CRITICAL Ultralytics API rules (do NOT violate):
- After model.train(), metrics come from: results.results_dict  (NOT results.metrics)
- Correct mAP key: results.results_dict.get("metrics/mAP50-95(B)", 0)
- Correct mAP50 key: results.results_dict.get("metrics/mAP50(B)", 0)
- NEVER use results.metrics, results.metrics_dict, or results.metric
- NEVER use results.results_dict["box_map"] — use .get() with fallback
- model.export() returns a Path; do not assign results_dict to it
- yolov5n is renamed to yolov5nu by Ultralytics — use yolov5nu directly
- Do not import stable_hf_export_onnx or any Hugging Face export helper in Ultralytics scripts.
"""
    elif family in {"transformers", "huggingface"}:
        family_rules = """CRITICAL Transformers rules:
- Keep dataset path conventions at config/data.yaml.
- Export ONNX with CPU model and CPU dummy tensors; avoid mixed CPU/CUDA export.
- Do NOT use transformers.onnx.export(...).
"""
    elif family == "whisper":
        family_rules = """CRITICAL Whisper/audio rules:
- Load local audio through config/data.yaml and manifest CSV when available.
- Use evaluate.load(...) for metrics; do not use datasets.load_metric.
- Keep the final stdout line as JSON.
"""
    elif getattr(variant, "task_type", None) == TaskType.AUDIO_CLASSIFICATION:
        family_rules = """CRITICAL local audio loader rules:
- Preserve the actual local dataset evidence. If a HuggingFace Arrow audio record exposes
  audio["bytes"] or audio["path"] and no audio["array"], do not switch to implicit
  Audio(decode=True) as a blind fix. Decode the observed bytes/path directly.
- For audio bytes, use only the standard library path unless the diagnostics prove a
  dependency is installed and needed: io.BytesIO(audio["bytes"]) + wave.open(...),
  then convert PCM frames to float32 numpy.
- Do not introduce SpeechBrain, wav2vec, HuBERT, Whisper, external load_dataset(...),
  librosa, or hub downloads while debugging a local loader implementation error.
- Keep loader.py responsible for reading real local user data; train.py should not bypass it.
"""
    else:
        family_rules = "CRITICAL family rule: preserve the existing framework intent and avoid cross-family imports.\n"

    loader_contract = (
        "- Loader-stage contract: when --smoke succeeds, the final JSON line "
        "MUST be {\"status\":\"success\", \"num_train\": <int>, "
        "\"num_val\": <int>, \"input_kind\": <str>}. Prefer also reporting "
        "input_shape/target_shape/num_classes/label_sample when known. Do not use status=ok.\n"
        if stage == "loader"
        else "- Keep output contract: last line must print JSON with keys status and metrics. "
             "If loader.py exists, preserve its return contract and adapt train.py to it; "
             "do not bypass loader.py by re-reading raw CSV/JSONL columns guessed from memory.\n"
    )

    diagnostic_text = _format_debugger_diagnostics(diagnostic_context)
    prompt = f"""You are a Python runtime debugger for EdgeCraft.
Repair the script only when the bug is an implementation error. Preserve the
DATASET_PLAN and analyzer evidence. Do not make the script pass by changing the
dataset source, fabricating data, inventing labels, or falling back to an
external public dataset.

Constraints:
{loader_contract}- Do not change dataset path conventions (config/data.yaml).
- Preserve model/task intent; only fix runtime/API/metric extraction errors.
- First decide whether this is patchable. If sample evidence shows wrong label/input semantics, return DEBUG_DECISION: reject_plan_error instead of forcing code to run.
- Use DEBUG_DECISION: insufficient_evidence when local dataset evidence is too weak; never fabricate data or fallback to external datasets.
- For loader path errors, prefer exact config/data.yaml schema.files over reconstructed subdir guesses.
- For F1/AUROC failures, repair label sampling, class weights, and threshold diagnostics before changing to a larger model.
- Prefer minimal edits.
- When the failure reveals a producer/consumer contract mismatch (for example a
  renamed field, dtype, shape, or return value), trace that contract through the
  current script and update every related producer and consumer consistently.
  Do not patch only the traceback line and leave downstream references stale.
- Do NOT use transformers.onnx.export(...). Use stable_hf_export_onnx(...) only for
  tokenizer-based Hugging Face models. For ordinary torch.nn.Module models, keep a
  direct torch.onnx.export path with the model's real tensor inputs.
- Existing training artifacts: {json.dumps(reusable_checkpoints)}. If and only if
  the traceback proves training completed and the remaining failure is artifact
  export, preserve the checkpoint and add a `--export-only` CLI path that reloads
  it, exports the deploy artifact, and prints the normal final JSON. Do not repeat
  dataset preprocessing or training in that path.
- Do NOT create stub/fake/dummy data, write synthetic CSVs, synthesize labels,
  or call load_dataset("speech_commands") / external datasets as a fallback.
- If train.py imports loader.py, respect the actual loader.py function signature;
  prefer supported keyword arguments and avoid positional config paths that could be
  interpreted as val_ratio/test_ratio.
- For PyTorch classification with CrossEntropyLoss, labels must be encoded with one
  global vocabulary, converted to torch.long, shaped [N], and in [0, num_classes-1].
- Do not create dummy/surrogate ONNX artifacts for sklearn/joblib pipelines. If a
  real ONNX conversion is unavailable, preserve the actual best.pt/joblib runtime
  path instead of making infer.py load a fake outputs/best.onnx.
- If the current DATASET_PLAN is wrong, or evidence is insufficient to identify
  real training data/labels, do not patch the script. Return DECISION:
  plan_error or insufficient_dataset_evidence with a concise SUMMARY.

{family_rules}

Context:
- stage: {stage}
- model_family: {variant.model_family}
- model_name: {variant.model_name}
- task_type: {variant.task_type.value}
- dataset_plan:
{variant.dataset_plan or "(missing)"}
- error_signature: {error_signature}
- error: {error}

recent_debug_history_for_this_trial:
{recent_debug_history}

stderr:
{stderr[-5000:]}

stdout:
{stdout[-3000:]}

diagnostic_context:
{diagnostic_text}

Return:
1) One line starting with "DECISION: " and one of:
   implementation_error | plan_error | insufficient_dataset_evidence
2) One short line starting with "SUMMARY: "
3) If and only if DECISION is implementation_error, the full patched python
   script inside a single ```python fenced block.

Original script:
```python
{original}
```"""
    try:
        resp = llm.invoke(
            [
                SystemMessage(content="You are a precise code repair assistant."),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as exc:
        return False, f"LLM debugger call failed: {exc}", before_hash, before_hash, reason

    content = (resp.content or "").strip()
    decision = _extract_debugger_decision(content)
    summary = ""
    m = re.search(r"SUMMARY:\s*(.*)", content)
    if m:
        summary = m.group(1).strip()
    if decision != "implementation_error":
        return False, (summary or f"debugger decision={decision}"), before_hash, before_hash, f"{reason};decision={decision}"
    patched = _extract_python_code(content)
    if not patched or len(patched) < max(80, len(original) // 3):
        return False, (summary or "LLM debugger returned invalid patch"), before_hash, before_hash, reason

    patched = WorkspaceManager.normalize_script_contract(
        patched,
        variant=variant,
        stage=(
            "infer" if stage in {"edge_benchmark", "component_host_infer"} else stage
        ),
    )
    provenance_violation = _provenance_guard_violation(patched)
    if provenance_violation:
        return (
            False,
            (summary or provenance_violation),
            before_hash,
            before_hash,
            f"{reason};{provenance_violation}",
        )
    if _contains_blocked_transformers_export(patched):
        return (
            False,
            (summary or "LLM patch still violates export contract"),
            before_hash,
            before_hash,
            reason,
        )

    script_path.write_text(patched)
    after_hash = _sha256_text(patched)
    if after_hash == before_hash:
        return False, (summary or "no effective patch"), before_hash, after_hash, reason
    return True, (summary or "applied LLM patch"), before_hash, after_hash, reason


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_trial_summary(variant: SolutionVariant, trial: TrialResult) -> None:
    """Print a structured INFO line summarising a completed trial."""
    parts = [f"[{variant.trial_id}]", variant.short_description()]

    if trial.local_metrics and trial.local_metrics.all_metrics:
        metrics_preview = ", ".join(
            f"{k}={v:.4f}" for k, v in list(trial.local_metrics.all_metrics.items())[:2]
        )
        parts.append(metrics_preview)

    if trial.edge_metrics:
        e = trial.edge_metrics
        latency = f"{e.latency_ms:.1f}ms" if e.latency_ms is not None else "not measured"
        memory = f"{e.memory_mb:.0f}MB" if e.memory_mb is not None else "not measured"
        parts.append(f"latency={latency}  mem={memory}")

    parts.append(
        f"stage={trial.stage_reached.value}  "
        f"progress={trial.crafting_progress.value}  "
        f"t={trial.duration_seconds:.0f}s"
    )

    if trial.error:
        parts.append(f"ERROR({trial.error_stage}): {trial.error}")
    if trial.failure_taxonomy:
        parts.append(f"failure={trial.failure_taxonomy}")

    logger.info("  ".join(parts))


def _record_failure(trial: TrialResult, error: str, stderr: str = "", stdout: str = "") -> None:
    category = classify_failure(error, stderr, stdout)
    trial.failure_taxonomy = category.value
    trial.next_search_hint = next_search_hint(category)


def _artifact_priority_for_variant(variant: Optional[SolutionVariant]) -> Dict[str, int]:
    """Prefer executable artifacts in the runtime requested by the variant."""
    requested = ((getattr(variant, "export_format", "") or "") if variant else "").lower().strip(".")
    if requested in {"pt", "pth", "torchscript", "ts"}:
        return {"torchscript": 0, "pt": 1, "onnx": 2, "engine": 3, "tflite": 4, "unknown": 9}
    if requested == "engine":
        return {"engine": 0, "onnx": 1, "torchscript": 2, "pt": 3, "tflite": 4, "unknown": 9}
    if requested == "tflite":
        return {"tflite": 0, "onnx": 1, "torchscript": 2, "pt": 3, "engine": 4, "unknown": 9}
    return {"onnx": 0, "engine": 1, "tflite": 2, "torchscript": 3, "pt": 4, "unknown": 9}


def _runtime_artifact_kind(path: Path) -> str:
    """Disambiguate executable TorchScript archives from weight checkpoints."""
    kind = artifact_kind(str(path))
    if kind != "pt":
        return kind
    fingerprint = build_artifact_fingerprint(str(path))
    return "torchscript" if fingerprint.graph_hash else kind


def _record_workspace_artifacts(
    trial: TrialResult,
    ws: Path,
    variant: Optional[SolutionVariant] = None,
) -> None:
    """Populate artifact contract from files in outputs/.

    This is intentionally generic: it records what exists instead of assuming a
    dataset or model family.  ONNX external data is tracked as a sidecar.
    """
    outputs = ws / "outputs"
    if not outputs.exists():
        return

    variant = variant or getattr(trial, "variant", None)
    priority = _artifact_priority_for_variant(variant)
    records: list[tuple[int, Path, str]] = []
    for path in sorted(p for p in outputs.iterdir() if p.is_file()):
        kind = _runtime_artifact_kind(path)
        role = "sidecar" if kind in {"onnx_external_data", "diagnostic"} else "candidate"
        trial.artifact_contract.artifacts[path.name] = ArtifactRecord(
            path=str(path),
            kind=kind,
            role=role,
            runtime=kind if kind != "onnx_external_data" else None,
            size_bytes=path.stat().st_size,
            required_on_edge=kind == "onnx_external_data",
        )
        if role != "sidecar":
            records.append((priority.get(kind, 9), path, kind))

    if records:
        records.sort(key=lambda item: (item[0], item[1].name))
        primary = records[0][1]
        for _, path, _ in records:
            if path.name in trial.artifact_contract.artifacts:
                trial.artifact_contract.artifacts[path.name].role = "candidate"
        trial.artifact_contract.primary_artifact = str(primary)
        trial.artifact_contract.artifacts[primary.name].role = "primary"
        trial.artifact_contract.artifacts[primary.name].required_on_edge = True
        fallbacks = [str(p) for _, p, _ in records[1:]]
        trial.artifact_contract.fallback_artifacts = fallbacks


def _record_declared_artifacts(
    trial: TrialResult,
    ws: Path,
    declared: Any,
) -> None:
    """Record paths explicitly returned by train.py as runtime artifacts."""
    if not isinstance(declared, dict):
        return

    outputs = (ws / "outputs").resolve()
    invalid: list[dict[str, str]] = []
    recorded: list[str] = []
    model_kinds = {"pt", "onnx", "engine", "torchscript", "tflite"}
    for declared_key, raw_path in declared.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        path = path if path.is_absolute() else ws / path
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(outputs)
        except (OSError, ValueError):
            invalid.append({"key": str(declared_key), "path": raw_path, "reason": "outside_outputs"})
            continue
        if not (resolved.is_file() or resolved.is_dir()):
            invalid.append({"key": str(declared_key), "path": raw_path, "reason": "missing"})
            continue

        kind = artifact_kind(str(resolved)) if resolved.is_file() else "directory"
        record_key = relative.as_posix()
        record = trial.artifact_contract.artifacts.get(record_key) or ArtifactRecord(
            path=str(resolved),
            kind=kind,
        )
        record.path = str(resolved)
        record.kind = kind
        if resolved.is_file():
            record.size_bytes = resolved.stat().st_size
            record.sha256 = _sha256_file(resolved)
        else:
            record.size_bytes = sum(
                path.stat().st_size for path in resolved.rglob("*") if path.is_file()
            )
            record.sha256 = None
        record.provenance = f"train_result.artifact_paths:{declared_key}"
        if kind not in model_kinds:
            record.role = "sidecar"
            record.required_on_edge = True
        trial.artifact_contract.artifacts[record_key] = record
        recorded.append(record_key)

    observations = dict(getattr(trial, "observations", {}) or {})
    observations["declared_artifacts"] = {
        "recorded": recorded,
        "invalid": invalid,
    }
    trial.observations = observations


def _json_absolute_paths(value: Any, *, limit: int = 8) -> list[str]:
    found: list[str] = []
    pending = [value]
    while pending and len(found) < limit:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str) and Path(item).is_absolute():
            found.append(item)
    return found


def _audit_edge_eval_bundle(
    trial: TrialResult,
    ws: Path,
    state: AgentState,
    declared: Any,
    *,
    require_official_split: bool = True,
) -> Optional[Evidence]:
    """Audit an optional solution-owned edge evaluation bundle.

    The payload format belongs to the generated solution. EdgeCraft always
    checks bundle integrity. Full evaluation additionally requires the frozen
    split hash and sample membership; an efficiency probe only needs a valid,
    self-contained input for measuring the same artifact graph.
    """
    if not isinstance(declared, dict) or not declared.get("edge_eval_manifest"):
        return None

    outputs = (ws / "outputs").resolve()
    errors: list[str] = []

    def _declared_path(key: str, *, allow_directory: bool = False) -> Optional[Path]:
        raw = declared.get(key)
        if not isinstance(raw, str) or not raw.strip():
            errors.append(f"missing declared artifact: {key}")
            return None
        path = Path(raw)
        path = path if path.is_absolute() else ws / path
        try:
            resolved = path.resolve()
            resolved.relative_to(outputs)
        except (OSError, ValueError):
            errors.append(f"{key} is outside outputs/")
            return None
        exists = resolved.is_file() or (allow_directory and resolved.is_dir())
        if not exists:
            errors.append(f"{key} does not exist")
            return None
        return resolved

    manifest_path = _declared_path("edge_eval_manifest")
    payload_path = _declared_path("edge_eval_payload", allow_directory=True)
    manifest: Dict[str, Any] = {}
    if manifest_path is not None:
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw_manifest, dict):
                manifest = raw_manifest
            else:
                errors.append("edge evaluation manifest must be a JSON object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid edge evaluation manifest: {exc}")

    split_name = str(manifest.get("split_name") or "")
    sample_ids = manifest.get("sample_ids")
    sample_ids = [str(item) for item in sample_ids] if isinstance(sample_ids, list) else []
    raw_source_ids = manifest.get("source_ids")
    source_ids: Optional[list[str]] = None
    if raw_source_ids is not None:
        if isinstance(raw_source_ids, list):
            source_ids = [str(item) for item in raw_source_ids]
            if len(source_ids) != len(sample_ids):
                errors.append("manifest source_ids does not match sample_ids")
        else:
            errors.append("manifest source_ids must be a list when present")
    if not split_name:
        errors.append("manifest split_name is missing")
    if not sample_ids:
        errors.append("manifest sample_ids is empty")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("manifest sample_ids contains duplicates")
    try:
        declared_count = int(manifest.get("num_samples"))
    except (TypeError, ValueError):
        declared_count = -1
    if declared_count != len(sample_ids):
        errors.append("manifest num_samples does not match sample_ids")

    raw_local_metrics = manifest.get("local_metrics")
    local_metrics: Dict[str, float] = {}
    if raw_local_metrics is not None:
        if not isinstance(raw_local_metrics, dict):
            errors.append("manifest local_metrics must be a JSON object when present")
        else:
            try:
                local_metrics = MetricRegistry.normalize_metric_dict(raw_local_metrics)
            except (TypeError, ValueError):
                errors.append("manifest local_metrics contains non-numeric values")

    manifest_payload = str(manifest.get("payload_path") or "")
    if not manifest_payload:
        errors.append("manifest payload_path is missing")
    elif manifest_path is not None and payload_path is not None:
        candidate = (manifest_path.parent / manifest_payload).resolve()
        if payload_path.is_dir():
            try:
                candidate.relative_to(payload_path)
            except ValueError:
                errors.append("manifest payload_path is outside edge_eval_payload bundle")
            else:
                if not candidate.exists():
                    errors.append("manifest payload_path entry does not exist")
        elif candidate != payload_path:
            errors.append("manifest payload_path does not match edge_eval_payload")

    # A staged solution-owned bundle must not retain paths into the cloud
    # workspace.  JSON is the only payload format whose contents EdgeCraft can
    # inspect without knowing solution-specific schemas; other formats remain
    # opaque and are audited through their manifest.
    if payload_path is not None and payload_path.suffix.lower() == ".json":
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            absolute_paths = _json_absolute_paths(payload)
            if absolute_paths:
                errors.append(
                    "edge evaluation payload contains absolute paths and is not self-contained"
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON edge evaluation payload: {exc}")

    split_info = ((state.get("dataset_info") or {}).get("split_manifest") or {})
    expected_hash = str(split_info.get("content_hash") or "") if isinstance(split_info, dict) else ""
    actual_hash = str(manifest.get("split_manifest_hash") or "")
    official_split_ids: Optional[set[str]] = None
    official_split_id_examples: list[str] = []
    split_manifest_path = split_info.get("path") if isinstance(split_info, dict) else None
    if require_official_split and expected_hash and actual_hash != expected_hash:
        errors.append("split_manifest_hash does not match the immutable split")
    if require_official_split and split_manifest_path:
        try:
            split_payload = json.loads(Path(str(split_manifest_path)).read_text(encoding="utf-8"))
            splits = split_payload.get("splits") if isinstance(split_payload, dict) else None
            selected = splits.get(split_name) if isinstance(splits, dict) else None
            if not isinstance(selected, list):
                errors.append(f"split_name {split_name!r} is not in the split manifest")
            else:
                official_split_ids = {str(item) for item in selected}
                official_split_id_examples = [str(item) for item in selected[:8]]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read immutable split manifest: {exc}")
    if official_split_ids is not None:
        provenance_ids = source_ids if source_ids is not None else sample_ids
        outside = [item for item in provenance_ids if item not in official_split_ids]
        if outside:
            field = "source_ids" if source_ids is not None else "sample_ids"
            detail = f"{len(outside)} {field} are outside split {split_name!r}"
            if source_ids is None:
                detail += (
                    "; derived windows or patches must provide one source_ids entry "
                    "per sample_id, mapped to the immutable split"
                )
            errors.append(detail)

    evidence = Evidence(
        probe_id="edge_eval_bundle_contract",
        quantity="component_contract",
        fidelity="static",
        outcome="fail" if errors else "pass",
        protocol={
            "decision_authority": (
                "strict_evaluation_contract"
                if bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
                and require_official_split
                else "efficiency_bundle_contract"
                if bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
                else "prompt_only"
            ),
            "official_split_membership_required": require_official_split,
            "split_name": split_name,
            "split_manifest_hash": actual_hash,
            "expected_split_manifest_hash": expected_hash,
            "num_samples": len(sample_ids),
            "valid_split_id_examples": official_split_id_examples,
            "provenance_mode": "source_ids" if source_ids is not None else "sample_ids",
            "num_unique_sources": len(set(source_ids or sample_ids)),
            "provenance_contract": {
                "sample_ids": "unique evaluated-item identifiers",
                "source_ids": (
                    "optional one-to-one immutable split identifiers for derived "
                    "windows or patches"
                ),
            },
            "local_metrics": local_metrics,
            "manifest_path": str(manifest_path or ""),
            "payload_path": str(payload_path or ""),
            "errors": errors[:8],
        },
    )
    _append_evidence(trial, evidence)
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["edge_eval_bundle_contract"] = {
        "status": evidence.outcome,
        "evidence_id": evidence.id,
        "split_name": split_name,
        "num_samples": len(sample_ids),
        "provenance_mode": "source_ids" if source_ids is not None else "sample_ids",
        "source_ids_supported_for_derived_samples": True,
        "valid_split_id_examples": official_split_id_examples,
        "local_metrics": local_metrics,
        "errors": errors[:8],
    }
    trial.observations = observations
    return evidence


def _strict_edge_eval_bundle_error(evidence: Optional[Evidence]) -> str:
    """Return the measured bundle-contract failure that must precede edge use."""
    if not bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False)):
        return ""
    if evidence is None:
        return "Strict edge evaluation bundle contract failed: required bundle is missing"
    if evidence.outcome != "fail":
        return ""
    errors = [str(item) for item in (evidence.protocol.get("errors") or []) if item]
    detail = "; ".join(errors[:4]) or "edge evaluation bundle is invalid"
    return f"Strict edge evaluation bundle contract failed: {detail}"


def _filter_untrusted_edge_quality(
    trial: TrialResult,
    metrics: Dict[str, float],
) -> Dict[str, float]:
    """Keep physical edge measurements when the declared eval split is invalid.

    Artifact execution still proves latency, memory, energy, and runtime facts.
    It does not prove task quality when the solution-owned evaluation bundle
    contradicts the immutable split manifest.  Preserve rejected values as
    diagnostic evidence instead of letting them enter scoring or HIL reports.
    """
    observations = dict(getattr(trial, "observations", {}) or {})
    contract = observations.get("edge_eval_bundle_contract") or {}
    parity_status = "unknown"
    parity_reason = "local_metrics_missing"
    comparisons: Dict[str, Dict[str, float | bool]] = {}
    roundtrip = observations.get("component_roundtrip") or {}
    roundtrip_metrics = MetricRegistry.normalize_metric_dict(roundtrip.get("metrics") or {})
    use_roundtrip = bool(
        roundtrip.get("status") == "success"
        and roundtrip.get("dataset_eval") is True
        and roundtrip_metrics
    )
    local_metrics = (
        roundtrip_metrics
        if use_roundtrip
        else MetricRegistry.normalize_metric_dict(contract.get("local_metrics") or {})
    )
    local_metric_source = (
        "host_artifact_roundtrip" if use_roundtrip else "edge_eval_manifest"
    )
    quality_metrics = {
        name: value
        for name, value in metrics.items()
        if not MetricRegistry.is_edge_metric(name)
        and name not in {"artifact_loaded", "artifact_present", "load_error_present"}
    }
    shared = sorted(set(local_metrics) & set(quality_metrics))
    if shared:
        abs_tol = max(0.0, float(getattr(settings, "EDGE_QUALITY_PARITY_ABS_TOL", 0.001)))
        rel_tol = max(0.0, float(getattr(settings, "EDGE_QUALITY_PARITY_REL_TOL", 0.02)))
        for name in shared:
            local_value = float(local_metrics[name])
            edge_value = float(quality_metrics[name])
            tolerance = max(abs_tol, rel_tol * max(abs(local_value), 1e-12))
            delta = abs(edge_value - local_value)
            comparisons[name] = {
                "local": local_value,
                "edge": edge_value,
                "absolute_delta": delta,
                "tolerance": tolerance,
                "match": delta <= tolerance,
            }
        parity_status = "pass" if all(item["match"] for item in comparisons.values()) else "fail"
        parity_reason = "within_tolerance" if parity_status == "pass" else "local_edge_metric_mismatch"
    elif local_metrics:
        parity_reason = "no_shared_quality_metric"

    parity = Evidence(
        probe_id="edge_quality_parity",
        quantity="prediction_parity",
        fidelity="measured_congruent" if comparisons else "proxy",
        outcome=parity_status,
        protocol={
            "decision_authority": "evaluation_validity_only",
            "comparisons": comparisons,
            "reason": parity_reason,
            "local_metric_source": local_metric_source,
            "strict_required": bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False)),
        },
    )
    _append_evidence(trial, parity)
    observations["edge_quality_parity"] = {
        "status": parity_status,
        "reason": parity_reason,
        "evidence_id": parity.id,
        "comparisons": comparisons,
        "local_metric_source": local_metric_source,
    }
    trial.observations = observations

    parity_invalid = parity_status == "fail" or (
        bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
        and parity_status != "pass"
    )
    if contract.get("status") != "fail" and not parity_invalid:
        return metrics

    execution_fields = {"artifact_loaded", "artifact_present", "load_error_present"}
    trusted = {
        name: value
        for name, value in metrics.items()
        if MetricRegistry.is_edge_metric(name) or name in execution_fields
    }
    rejected = {
        name: value
        for name, value in metrics.items()
        if name not in trusted
    }
    observations["edge_quality_provenance"] = {
        "status": "invalid",
        "reason": (
            "edge_eval_bundle_contract_failed"
            if contract.get("status") == "fail"
            else "edge_quality_parity_failed"
        ),
        "contract_evidence_id": contract.get("evidence_id"),
        "parity_evidence_id": parity.id,
        "rejected_metrics": rejected,
    }
    trial.observations = observations
    return trusted


def _record_evaluation_validity(
    trial: TrialResult,
    *,
    artifact_executed: bool,
    raw_metrics: Dict[str, Any],
) -> None:
    """Record load, execution, and semantic evaluation as separate facts."""
    observations = dict(getattr(trial, "observations", {}) or {})
    contract = observations.get("edge_eval_bundle_contract") or {}
    parity = observations.get("edge_quality_parity") or {}
    artifact_loaded = _safe_float(raw_metrics.get("artifact_loaded"))
    load_error = _safe_float(raw_metrics.get("load_error_present"))
    artifact_loadable = bool(
        artifact_executed
        or (artifact_loaded is not None and artifact_loaded > 0 and load_error != 1.0)
    )
    evaluation_valid = bool(
        artifact_executed
        and contract.get("status") == "pass"
        and parity.get("status") == "pass"
    )
    reasons: list[str] = []
    if not artifact_loadable:
        reasons.append("artifact_not_loadable")
    if not artifact_executed:
        reasons.append("runtime_not_executed")
    if contract.get("status") != "pass":
        reasons.append("evaluation_bundle_invalid")
    if parity.get("status") != "pass":
        reasons.append("local_edge_parity_unproven")
    observations["evaluation_validity"] = {
        "artifact_loadable": artifact_loadable,
        "runtime_executed": bool(artifact_executed),
        "evaluation_valid": evaluation_valid,
        "contract_evidence_id": contract.get("evidence_id"),
        "parity_evidence_id": parity.get("evidence_id"),
        "reasons": reasons,
    }
    trial.observations = observations


def _use_scheduler_backend() -> bool:
    return str(settings.EXECUTION_BACKEND or "").strip().lower() == "scheduler"


def _job_result_to_trial(
    *,
    job_result: JobResult,
    trial: TrialResult,
    ws: Path,
    variant: SolutionVariant,
    start_time: float,
) -> TrialResult:
    """Fold scheduler evidence back into the existing TrialResult contract."""
    edge_result = _edge_result_from_job_result(job_result)
    runtime_fields = _resolve_edge_runtime_fields(edge_result)
    trial.stdout = job_result.stdout or ""
    trial.stderr = job_result.stderr or ""
    trial.artifact_paths.update(job_result.artifact_paths or {})
    if job_result.local_metrics:
        trial.local_metrics = LocalMetrics(
            all_metrics=MetricRegistry.normalize_metric_dict(job_result.local_metrics)
        )
    if job_result.edge_metrics:
        metrics = MetricRegistry.normalize_metric_dict(job_result.edge_metrics)
        trial.edge_metrics = EdgeMetrics(
            latency_ms=_safe_float(metrics.get("Latency", job_result.edge_metrics.get("latency_ms"))),
            latency_mean_ms=_safe_float(metrics.get("Latency_mean")),
            latency_p95_ms=_safe_float(metrics.get("Latency_p95")),
            latency_p99_ms=_safe_float(metrics.get("Latency_p99")),
            memory_mb=_safe_float(metrics.get("Memory_mb", job_result.edge_metrics.get("memory_mb"))),
            runtime_used=runtime_fields["runtime_used"],
            runtime_provider=runtime_fields["runtime_provider"],
            artifact_used=runtime_fields["artifact_used"],
            throughput_fps=float(metrics["Throughput"]) if "Throughput" in metrics else None,
            all_metrics=metrics,
        )
    trial.runtime_report.requested_runtime = str(
        edge_result.get("requested_runtime")
        or getattr(variant, "export_format", None)
        or artifact_kind(job_result.artifact_paths.get("artifact", ""))
        or ""
    )
    trial.runtime_report.runtime_used = runtime_fields["runtime_used"]
    trial.runtime_report.runtime_provider = runtime_fields["runtime_provider"]
    trial.runtime_report.artifact_used = runtime_fields["artifact_used"]
    trial.runtime_report.fallback_reason = _runtime_field_to_str(edge_result.get("fallback_reason"))
    _record_workspace_artifacts(trial, ws, variant)
    trial.failure_context["scheduler"] = {
        "job_id": job_result.job_id,
        "status": job_result.status.value,
        "stage_reached": job_result.stage_reached,
        "queue_wait_s": job_result.queue_wait_s,
        "train_wait_s": job_result.train_wait_s,
        "edge_wait_s": job_result.edge_wait_s,
        "train_wall_s": job_result.train_wall_s,
        "edge_wall_s": job_result.edge_wall_s,
        "total_wall_s": job_result.total_wall_s,
        "stage_times": dict(job_result.stage_times or {}),
        "gpu_id": job_result.gpu_id,
        "edge_device_id": job_result.edge_device_id,
        "diagnostics": dict(job_result.diagnostics or {}),
    }
    if edge_result:
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["scheduler_edge_result"] = edge_result
        trial.observations = observations
    trial.duration_seconds = time.time() - start_time

    if job_result.status == JobStatus.SUCCESS and job_result.edge_metrics:
        trial.stage_reached = StageReached.EDGE_BENCHMARK
        trial.crafting_progress = CraftingProgress.EDGE_RUNNABLE
        return trial

    if job_result.status == JobStatus.SUCCESS:
        trial.stage_reached = StageReached.TRAIN
        trial.crafting_progress = CraftingProgress.TRAIN_SUCCEEDED
        if any(a.kind in {"onnx", "engine", "torchscript"} for a in trial.artifact_contract.artifacts.values()):
            trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
        return trial

    if job_result.artifact_paths:
        trial.stage_reached = StageReached.TRAIN
        trial.crafting_progress = CraftingProgress.TRAIN_SUCCEEDED
        if any(a.kind in {"onnx", "engine", "torchscript"} for a in trial.artifact_contract.artifacts.values()):
            trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
    elif job_result.status == JobStatus.DATA_PREP_FAILED:
        trial.stage_reached = StageReached.DATA_PREP
        trial.crafting_progress = CraftingProgress.SCRIPT_GENERATED
    else:
        trial.stage_reached = StageReached.DATA_PREP

    trial.error = job_result.error or f"scheduler job failed: {job_result.status.value}"
    trial.error_stage = job_result.error_stage or job_result.stage_reached or "scheduler"
    _record_failure(trial, trial.error, job_result.stderr, job_result.stdout)
    return trial


def _run_trial_via_scheduler(
    *,
    state: AgentState,
    variant: SolutionVariant,
    trial: TrialResult,
    ws: Path,
    start_time: float,
) -> TrialResult:
    from edgecraft.scheduler.shared_runtime import get_shared_scheduler_runtime

    trial_bank = state.get("trial_bank")
    estimate = WorkloadEstimator().estimate(
        variant,
        state.get("target_device", "") or "",
        trial_bank.get_all() if trial_bank is not None else [],
    )
    spec = JobSpec(
        job_id=f"{state['run_id']}:{variant.trial_id}",
        tenant_id=state.get("tenant_id") or "default",
        request_id=state["run_id"],
        variant=variant,
        workspace_path=str(ws),
        train_script=str(ws / "train.py"),
        infer_script=str(ws / "infer.py"),
        edge_device_id=state.get("target_device", "") or "",
        edge_device_ip=state.get("device_ip", "") or "",
        edge_ssh_key=state.get("ssh_key", "") or "",
        edge_docker_image=state.get("docker_image"),
        dataset_path=str(
            (state.get("dataset_info") or {}).get("dataset_path")
            or state.get("dataset_path")
            or ""
        ),
        export_format=getattr(variant, "export_format", None) or "onnx",
        quant_mode=getattr(variant, "quant_mode", None) or "fp16",
        require_exact_runtime=True,
        skip_edge=not bool(state.get("device_ip")),
        estimate=estimate,
        train_timeout_s=int(settings.FULL_TRAIN_EXPORT_TIMEOUT_S),
        edge_timeout_s=int(settings.EDGE_RUN_TIMEOUT_S),
        prior_score=float(getattr(variant, "prior_score", 0.5) or 0.5),
        stage_label="p2_full",
    )
    timeout_s = float(settings.SCHEDULER_JOB_TIMEOUT_S or 0.0) or None
    job_result = get_shared_scheduler_runtime().submit_batch([spec], timeout_s=timeout_s)[0]
    return _job_result_to_trial(
        job_result=job_result,
        trial=trial,
        ws=ws,
        variant=variant,
        start_time=start_time,
    )


def _edge_result_from_job_result(job_result: JobResult) -> Dict[str, Any]:
    """Reconstruct the runner envelope retained by a scheduler edge stage."""
    result = dict((job_result.diagnostics or {}).get("edge_result") or {})
    result["metrics"] = dict(job_result.edge_metrics or {})
    result.setdefault(
        "status",
        "success" if job_result.status == JobStatus.SUCCESS else "error",
    )
    if job_result.error:
        result.setdefault("error", job_result.error)
    result["stdout"] = job_result.stdout
    result["stderr"] = job_result.stderr
    result["scheduler_job_id"] = job_result.job_id
    return result


def _run_scheduler_full_efficiency_replay(
    *,
    trial: TrialResult,
    state: AgentState,
    variant: SolutionVariant,
    ws: Path,
    artifact_path: str,
    primary_edge_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Schedule a same-artifact evaluator-owned P2 efficiency measurement."""
    from edgecraft.scheduler.shared_runtime import get_shared_scheduler_runtime

    fingerprint = build_artifact_fingerprint(artifact_path).model_dump(mode="json")
    infer_path = write_runtime_repro_infer(
        str(ws / ".edgecraft" / "full_efficiency_infer.py"),
        input_specs=list(fingerprint.get("input_specs") or []),
        warmup=EFFICIENCY_PROBE_WARMUP,
        min_warmup_seconds=EFFICIENCY_PROBE_MIN_WARMUP_SECONDS,
        repetitions=EFFICIENCY_PROBE_REPETITIONS,
        min_measure_seconds=EFFICIENCY_PROBE_MIN_MEASURE_SECONDS,
        measurement_sessions=EFFICIENCY_PROBE_SESSIONS,
    )
    artifact_paths = {
        str(key): str(value)
        for key, value in (trial.artifact_paths or {}).items()
        if value
    }
    artifact_paths.update({"artifact": artifact_path, "train": artifact_path})
    spec = JobSpec(
        job_id=f"{state['run_id']}:{variant.trial_id}:p2-efficiency",
        tenant_id=state.get("tenant_id") or "default",
        request_id=state["run_id"],
        variant=variant,
        workspace_path=str(ws),
        infer_script=str(infer_path),
        edge_device_id=state.get("target_device", "") or "",
        edge_device_ip=state.get("device_ip", "") or "",
        edge_ssh_key=state.get("ssh_key", "") or "",
        edge_docker_image=state.get("docker_image"),
        dataset_path=str(
            (state.get("dataset_info") or {}).get("dataset_path")
            or state.get("dataset_path")
            or ""
        ),
        export_format=getattr(variant, "export_format", None) or artifact_kind(artifact_path),
        quant_mode=getattr(variant, "quant_mode", None) or "fp16",
        require_exact_runtime=True,
        edge_only=True,
        prebuilt_artifact_paths=artifact_paths,
        artifact_manifest=trial.artifact_contract.model_dump(mode="json"),
        stage_label="p2_efficiency_replay",
        estimate=WorkloadEstimate(
            gpu_memory_gb=0.0,
            gpu_wall_time_s=0.0,
            edge_wall_time_s=float(getattr(settings, "L1_EDGE_TIMEOUT_S", 600)),
            confidence=1.0,
            source="p2_edge_only",
        ),
        edge_timeout_s=int(getattr(settings, "L1_EDGE_TIMEOUT_S", 600)),
        prior_score=float(getattr(variant, "prior_score", 0.5) or 0.5),
    )
    timeout_s = float(settings.SCHEDULER_JOB_TIMEOUT_S or 0.0) or None
    job_result = get_shared_scheduler_runtime().submit_batch([spec], timeout_s=timeout_s)[0]
    replay = _edge_result_from_job_result(job_result)
    execution = _record_stage_execution_evidence(
        trial,
        stage="full_efficiency_replay",
        result={**replay, "elapsed_s": float(job_result.edge_wall_s or 0.0)},
        resource="device",
    )
    raw_metrics = replay.get("metrics") if isinstance(replay.get("metrics"), dict) else {}
    artifact_executed, reasons = _artifact_execution_evidence(raw_metrics)
    runtime_fields = _resolve_edge_runtime_fields(replay)
    requested_runtime = str(getattr(variant, "export_format", None) or artifact_kind(artifact_path))
    attested = bool(
        replay.get("status") == "success"
        and artifact_executed
        and replay.get("require_exact_runtime") is True
        and not replay.get("fallback_reason")
        and _runtime_identity(runtime_fields["runtime_used"]) == _runtime_identity(requested_runtime)
        and str(replay.get("source_artifact_hash") or "")
        == str(fingerprint.get("artifact_hash") or "")
        and str(replay.get("requested_graph_hash") or "")
        == str(fingerprint.get("graph_hash") or "")
    )
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["full_efficiency_replay"] = {
        "attempted": True,
        "status": replay.get("status"),
        "artifact_executed": artifact_executed,
        "execution_reasons": reasons,
        "attested_exact_artifact": attested,
        "execution_evidence_id": execution.id,
        "scheduler_job_id": job_result.job_id,
        "elapsed_s": float(job_result.edge_wall_s or 0.0),
        "error": replay.get("error"),
    }
    trial.observations = observations
    if not attested:
        return primary_edge_result
    return _merge_full_efficiency_replay(primary_edge_result, replay)


def _finalize_scheduler_verification(
    *,
    trial: TrialResult,
    state: AgentState,
    variant: SolutionVariant,
    user_spec: Optional[UserSpec],
    ws: Path,
) -> None:
    """Map a scheduler-owned P2 run into the common verifier authority.

    The scheduler owns placement and stage execution only.  It never decides
    feasibility; its measured result is folded back into the same evidence and
    policy objects used by the direct backend.
    """
    scheduler_record = dict((trial.failure_context or {}).get("scheduler") or {})
    edge_result = dict((trial.observations or {}).get("scheduler_edge_result") or {})
    status = str(scheduler_record.get("status") or "")
    _record_stage_execution_evidence(
        trial,
        stage="full_train_export",
        result={
            "status": (
                "success"
                if scheduler_record.get("stage_reached") == "edge"
                or status == JobStatus.SUCCESS.value
                else status
            ),
            "elapsed_s": float(scheduler_record.get("train_wall_s") or 0.0),
        },
        resource="gpu",
    )
    if float(scheduler_record.get("edge_wall_s") or 0.0) > 0.0:
        _record_stage_execution_evidence(
            trial,
            stage="full_edge_benchmark",
            result={
                "status": "success" if status == JobStatus.SUCCESS.value else status,
                "elapsed_s": float(scheduler_record.get("edge_wall_s") or 0.0),
            },
            resource="device",
        )
    if status != JobStatus.SUCCESS.value or trial.edge_metrics is None:
        if edge_result:
            artifact_path = (
                _primary_deploy_artifact(trial)
                or str(trial.artifact_paths.get("artifact") or "")
            )
            requested_runtime = str(
                trial.runtime_report.requested_runtime
                or getattr(variant, "export_format", None)
                or artifact_kind(artifact_path)
                or ""
            )
            runtime_fields = _resolve_edge_runtime_fields(edge_result)
            attempted_runtime = runtime_fields["runtime_used"] or requested_runtime
            attempted_artifact = runtime_fields["artifact_used"] or artifact_path
            failure = _runtime_failure_observation_context(
                edge_result,
                requested_runtime=requested_runtime,
                requested_artifact=artifact_path,
                attempted_runtime=attempted_runtime,
                attempted_artifact=attempted_artifact,
            )
            _observe_compatibility_failure(
                trial,
                state,
                artifact_path=failure["artifact_path"],
                error_text=failure["error_text"] or str(trial.error or ""),
                failure_stage=failure["failure_stage"] or str(trial.error_stage or ""),
                runtime_used=failure["runtime"],
            )
        report = _ensure_verification_report(trial)
        report.level = "L2"
        report.status = "fail"
        report.evidence_source = "measured"
        report.source = "scheduler_full_pipeline"
        _apply_verifier_policy(trial, user_spec, next_probe="")
        return

    artifact_path = _primary_deploy_artifact(trial) or trial.artifact_paths.get("artifact", "")
    fingerprint = (
        build_artifact_fingerprint(artifact_path).model_dump(mode="json")
        if artifact_path and Path(artifact_path).is_file()
        else {}
    )
    _record_declared_artifacts(trial, ws, trial.artifact_paths)
    bundle_contract = _audit_edge_eval_bundle(
        trial,
        ws,
        state,
        trial.artifact_paths,
    )
    contract_errors: List[str] = []
    strict_bundle_error = _strict_edge_eval_bundle_error(bundle_contract)
    if strict_bundle_error:
        contract_errors.append(strict_bundle_error)
    if artifact_path:
        payload_contract = _artifact_eval_payload_contract(
            trial=trial,
            artifact_path=artifact_path,
        )
        host_contract = _run_host_infer_contract_probe(
            trial=trial,
            ws=ws,
            artifact_path=artifact_path,
        )
        host_status = (trial.observations.get("component_roundtrip") or {}).get(
            "status"
        )
        # infer.py may map payload fields to model input names. As in the
        # direct backend, an executed host round trip resolves a static-name
        # mismatch; full device execution and quality parity are still required.
        if payload_contract.outcome == "fail" and host_status != "success":
            contract_errors.append(
                "artifact and evaluation payload have an interface mismatch"
            )
        if (
            bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
            and _declared_artifact_path(trial, "edge_eval_manifest") is not None
            and (host_contract is None or host_status != "success")
        ):
            contract_errors.append(
                "host artifact/evaluator round trip did not complete"
            )
    if contract_errors:
        trial.error = "; ".join(contract_errors)
        trial.error_stage = "edge_eval_contract"
        trial.next_search_hint = (
            "Preserve the trained artifact and repair the evaluation manifest, "
            "payload, and infer.py as one self-contained contract."
        )
        report = _ensure_verification_report(trial)
        report.level = "admission"
        report.status = "fail"
        report.evidence_source = "static"
        report.admission_status = "rejected"
        report.source = "scheduler_candidate_admission"
        report.errors.extend(contract_errors)
        _apply_verifier_policy(trial, user_spec, next_probe="")
        return
    if artifact_path:
        _run_evaluation_metric_contract(
            trial=trial,
            ws=ws,
            state=state,
            artifact_path=artifact_path,
        )
    prior_report = trial.verification_report
    l1_fingerprint = (
        dict(prior_report.artifact_fingerprint or {})
        if prior_report is not None and prior_report.level == "L1"
        else {}
    )
    has_l1_measurement = bool(
        prior_report
        and any(
            item.probe_id == "efficiency" and item.value is not None
            for item in prior_report.evidence
        )
    )
    if has_l1_measurement and artifact_path and fingerprint:
        edge_result = _run_scheduler_full_efficiency_replay(
            trial=trial,
            state=state,
            variant=variant,
            ws=ws,
            artifact_path=artifact_path,
            primary_edge_result=edge_result,
        )
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["scheduler_edge_result"] = edge_result
        trial.observations = observations
        normalized_edge = MetricRegistry.normalize_metric_dict(
            edge_result.get("metrics")
            if isinstance(edge_result.get("metrics"), dict)
            else {}
        )
        if "Energy_mj" in normalized_edge and "Energy_j" not in normalized_edge:
            normalized_edge["Energy_j"] = float(normalized_edge["Energy_mj"]) / 1000.0
        runtime_fields = _resolve_edge_runtime_fields(edge_result)
        trial.edge_metrics = EdgeMetrics(
            latency_ms=_safe_float(normalized_edge.get("Latency")),
            latency_mean_ms=_safe_float(normalized_edge.get("Latency_mean")),
            memory_mb=_safe_float(normalized_edge.get("Memory_mb")),
            latency_p95_ms=_safe_float(normalized_edge.get("Latency_p95")),
            latency_p99_ms=_safe_float(normalized_edge.get("Latency_p99")),
            runtime_used=runtime_fields["runtime_used"],
            runtime_provider=runtime_fields["runtime_provider"],
            artifact_used=runtime_fields["artifact_used"],
            throughput_fps=_safe_float(normalized_edge.get("Throughput")),
            all_metrics=normalized_edge,
        )
        trial.runtime_report.runtime_used = runtime_fields["runtime_used"]
        trial.runtime_report.runtime_provider = runtime_fields["runtime_provider"]
        trial.runtime_report.artifact_used = runtime_fields["artifact_used"]

    if l1_fingerprint and fingerprint:
        congruence = compare_artifact_fingerprints(l1_fingerprint, fingerprint)
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["l1_l2_congruence"] = {
            **congruence,
            "l1_artifact_mismatch": bool(
                congruence.get("checked") and not congruence.get("match")
            ),
            "l1_fingerprint": l1_fingerprint,
            "l2_fingerprint": fingerprint,
        }
        trial.observations = observations
    raw_edge_metrics = (
        dict(edge_result.get("metrics") or {})
        if isinstance(edge_result.get("metrics"), dict)
        else {}
    )
    normalized_edge = MetricRegistry.normalize_metric_dict(raw_edge_metrics)
    if "Energy_mj" in normalized_edge and "Energy_j" not in normalized_edge:
        normalized_edge["Energy_j"] = float(normalized_edge["Energy_mj"]) / 1000.0
    artifact_executed, execution_reasons = _artifact_execution_evidence(
        raw_edge_metrics
    )
    artifact_executed = bool(
        edge_result.get("status") == "success"
        and trial.runtime_report.artifact_used
        and artifact_executed
    )
    normalized_edge = _filter_untrusted_edge_quality(trial, normalized_edge)
    runtime_fields = _resolve_edge_runtime_fields(edge_result)
    trial.edge_metrics = EdgeMetrics(
        latency_ms=_safe_float(normalized_edge.get("Latency")),
        latency_mean_ms=_safe_float(normalized_edge.get("Latency_mean")),
        memory_mb=_safe_float(normalized_edge.get("Memory_mb")),
        latency_p95_ms=_safe_float(normalized_edge.get("Latency_p95")),
        latency_p99_ms=_safe_float(normalized_edge.get("Latency_p99")),
        runtime_used=runtime_fields["runtime_used"],
        runtime_provider=runtime_fields["runtime_provider"],
        artifact_used=runtime_fields["artifact_used"],
        throughput_fps=_safe_float(normalized_edge.get("Throughput")),
        all_metrics=normalized_edge,
    )
    trial.runtime_report.runtime_used = runtime_fields["runtime_used"]
    trial.runtime_report.runtime_provider = runtime_fields["runtime_provider"]
    trial.runtime_report.artifact_used = runtime_fields["artifact_used"]
    _record_evaluation_validity(
        trial,
        artifact_executed=artifact_executed,
        raw_metrics=raw_edge_metrics,
    )

    requested_runtime = str(
        trial.runtime_report.requested_runtime
        or getattr(variant, "export_format", None)
        or artifact_kind(artifact_path)
        or ""
    )
    runtime_used = str(trial.runtime_report.runtime_used or "")
    source_hash = str(edge_result.get("source_artifact_hash") or "")
    local_hash = str(fingerprint.get("artifact_hash") or "")
    graph_hash = str(fingerprint.get("graph_hash") or "")
    requested_graph_hash = str(edge_result.get("requested_graph_hash") or "")
    runtime_exact = bool(
        edge_result.get("require_exact_runtime") is True
        and not edge_result.get("fallback_reason")
        and _runtime_identity(runtime_used) == _runtime_identity(requested_runtime)
    )
    fingerprint_verified = bool(
        graph_hash
        and source_hash
        and local_hash == source_hash
        and requested_graph_hash == graph_hash
        and trial.runtime_report.artifact_used
    )
    edge_metrics = dict(trial.edge_metrics.all_metrics or {})
    strict_quality = bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
    validity = (trial.observations or {}).get("evaluation_validity") or {}
    reported_evaluation_valid = validity.get("evaluation_valid") is True
    evaluation_valid = bool(
        reported_evaluation_valid if strict_quality else artifact_executed
    )
    full_authorized = bool(
        artifact_executed
        and runtime_exact
        and fingerprint_verified
        and evaluation_valid
    )

    properties = dict(fingerprint.get("artifact_properties") or {})
    properties.update(
        {
            "requested_runtime": requested_runtime,
            "runtime_used": runtime_used,
            "runtime_exact": runtime_exact,
            "artifact_used": trial.runtime_report.artifact_used,
            "source_artifact_hash": source_hash,
            "fingerprint_source": "scheduler_executed_staged_artifact",
        }
    )
    fingerprint["artifact_properties"] = properties
    environment = _runtime_environment_fingerprint(
        state,
        runtime=runtime_used or requested_runtime,
        precision=getattr(variant, "quant_mode", None),
    )
    measurement_protocol = (
        dict(edge_result.get("measurement_protocol"))
        if isinstance(edge_result.get("measurement_protocol"), dict)
        else {}
    )
    common_protocol = {
        "decision_authority": "p2_accept" if full_authorized else "audit_only",
        "artifact_executed": artifact_executed,
        "runtime_exact": runtime_exact,
        "fingerprint_verified": fingerprint_verified,
        "evaluation_valid": evaluation_valid,
        "strict_evaluation_required": strict_quality,
        "runtime": runtime_used,
        "requested_runtime": requested_runtime,
        "artifact_used": trial.runtime_report.artifact_used,
        "scheduler_job_id": scheduler_record.get("job_id"),
        "measurement_sessions": measurement_protocol.get("measurement_sessions"),
        "latency_statistic": measurement_protocol.get("latency_statistic"),
        "latency_uncertainty": measurement_protocol.get("latency_uncertainty"),
    }
    evidence: List[Evidence] = []
    for quantity, value in (trial.local_metrics.all_metrics if trial.local_metrics else {}).items():
        evidence.append(
            Evidence(
                probe_id="full",
                quantity=quantity,
                fidelity="measured_congruent" if full_authorized else "proxy",
                outcome="pass" if full_authorized else "unknown",
                value=float(value),
                sigma=(
                    _safe_float(edge_metrics.get("latency_p95_session_std_ms"))
                    if MetricRegistry.canonicalize_name(quantity) == "Latency"
                    else None
                ),
                artifact_fingerprint=fingerprint,
                environment_fingerprint=environment,
                protocol={"training": "full", **common_protocol},
            )
        )
    for quantity, value in edge_metrics.items():
        if not isinstance(value, (int, float)):
            continue
        evidence.append(
            Evidence(
                probe_id="full",
                quantity=quantity,
                fidelity="measured_congruent" if full_authorized else "proxy",
                outcome="pass" if full_authorized else "unknown",
                value=float(value),
                artifact_fingerprint=fingerprint,
                environment_fingerprint=environment,
                protocol={
                    **dict(edge_result.get("measurement_protocol") or {}),
                    **common_protocol,
                },
            )
        )
    report = _append_evidence(trial, *evidence)
    report.level = "L2"
    report.status = "pass" if full_authorized else "fail"
    report.evidence_source = "measured"
    report.artifact_fingerprint = fingerprint
    report.artifact_status = "present" if fingerprint else "missing"
    report.runtime_status = runtime_used
    report.source = "scheduler_full_pipeline"
    if not full_authorized:
        report.errors.append(
            "Scheduler P2 evidence is non-authoritative: execution, exact runtime, "
            "artifact fingerprint, or strict evaluation validity is unproven."
        )
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["edge_execution_contract"] = {
        "artifact_executed": artifact_executed,
        "reasons": execution_reasons,
        "runtime_exact": runtime_exact,
        "fingerprint_verified": fingerprint_verified,
        "evaluation_valid": evaluation_valid,
    }
    observations["evaluation_validity"] = {
        "artifact_loadable": artifact_executed,
        "runtime_executed": artifact_executed,
        "evaluation_valid": evaluation_valid,
        "reasons": (
            []
            if evaluation_valid
            else ["scheduler edge result did not attest evaluator-valid execution"]
        ),
    }
    trial.observations = observations
    _append_online_calibration_pairs(trial)
    _apply_verifier_policy(trial, user_spec, next_probe="")


def _archive_direct_pending_edge_job(
    *,
    state: AgentState,
    variant: SolutionVariant,
    trial: TrialResult,
    ws: Path,
    model_path: str,
) -> Optional[Path]:
    """Persist the full-pipeline edge boundary when explicitly requested."""
    archive_root = str(settings.PENDING_EDGE_ARCHIVE_DIR or "").strip()
    if not archive_root or not model_path:
        return None
    try:
        from edgecraft.scheduler.archive import archive_pending_edge_job

        spec = JobSpec(
            job_id=f"{state['run_id']}:{variant.trial_id}",
            tenant_id=state.get("tenant_id") or "default",
            request_id=state["run_id"],
            variant=variant,
            workspace_path=str(ws),
            train_script=str(ws / "train.py"),
            infer_script=str(ws / "infer.py"),
            edge_device_id=state.get("target_device", "") or "",
            edge_device_ip=state.get("device_ip", "") or "",
            edge_ssh_key=state.get("ssh_key", "") or "",
            edge_docker_image=state.get("docker_image"),
            dataset_path=str(
                (state.get("dataset_info") or {}).get("dataset_path")
                or state.get("dataset_path")
                or ""
            ),
            export_format=getattr(variant, "export_format", None) or "onnx",
            quant_mode=getattr(variant, "quant_mode", None) or "fp16",
        )
        artifact_paths = {
            str(key): str(value)
            for key, value in (trial.artifact_paths or {}).items()
            if value
        }
        for key, record in trial.artifact_contract.artifacts.items():
            if record.required_on_edge or (record.provenance or "").startswith(
                "train_result.artifact_paths:"
            ):
                artifact_paths.setdefault(str(key), record.path)
        artifact_paths.update({"artifact": model_path, "train": model_path})
        train_result = JobResult(
            job_id=spec.job_id,
            status=JobStatus.SUCCESS,
            stage_reached="train",
            local_metrics=dict(trial.local_metrics.all_metrics if trial.local_metrics else {}),
            artifact_paths=artifact_paths,
            edge_device_id=spec.edge_device_id,
        )
        return archive_pending_edge_job(
            spec,
            train_result,
            Path(archive_root),
            artifact_manifest=trial.artifact_contract.model_dump(mode="json"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not archive full-pipeline edge job {variant.trial_id}: {exc}")
        return None


def _complete_direct_pending_edge_job(
    manifest_path: Optional[Path],
    result: Dict[str, Any],
) -> None:
    if manifest_path is None:
        return
    try:
        from edgecraft.scheduler.archive import job_result_from_dict, mark_edge_snapshot_completed

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        job_result = job_result_from_dict(payload["train_result"])
        raw_metrics = result.get("metrics") or {}
        job_result.edge_metrics = {
            str(key): float(value)
            for key, value in raw_metrics.items()
            if isinstance(value, (int, float))
        } if isinstance(raw_metrics, dict) else {}
        if result.get("status") == "success":
            job_result.status = JobStatus.SUCCESS
            job_result.stage_reached = "edge"
        else:
            job_result.status = JobStatus.EDGE_FAILED
            job_result.error_stage = "edge"
            job_result.error = str(result.get("error") or "edge benchmark failed")
        mark_edge_snapshot_completed(manifest_path, job_result, source="pipeline_executor")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not complete full-pipeline edge snapshot {manifest_path}: {exc}")


def _primary_deploy_artifact(trial: TrialResult) -> str:
    """Return the artifact path that should be deployed to the edge runner."""
    if trial.artifact_contract.primary_artifact:
        return trial.artifact_contract.primary_artifact
    for path in trial.artifact_contract.fallback_artifacts:
        if path:
            return path
    return trial.artifact_paths.get("train", "")


def _runtime_field_to_str(value: Any) -> Optional[str]:
    """Normalize runtime metadata from LLM JSON into contract-safe strings."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return str(value[0])
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _resolve_edge_runtime_fields(result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Read one runtime identity from either runner payload location."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    return {
        "runtime_used": _runtime_field_to_str(
            metrics.get("runtime_used") or metrics.get("Runtime") or result.get("runtime_used")
        ),
        "runtime_provider": _runtime_field_to_str(
            metrics.get("runtime_provider")
            or metrics.get("provider")
            or result.get("runtime_provider")
            or result.get("provider")
        ),
        "artifact_used": _runtime_field_to_str(
            metrics.get("artifact_used")
            or metrics.get("artifact_path")
            or result.get("artifact_used")
        ),
    }


def _runtime_identity(value: Any) -> str:
    """Return a stable identity for runtime-attempt comparisons."""
    key = str(_runtime_field_to_str(value) or "").strip().lower().lstrip(".")
    return {
        "ort": "onnx",
        "onnxruntime": "onnx",
        "trt": "engine",
        "tensorrt": "engine",
        "pytorch": "pt",
        "torch": "pt",
        "torchscript": "pt",
        "litert": "tflite",
    }.get(key, key)


def _runtime_fallback_evidence(
    result: Dict[str, Any],
    *,
    requested_runtime: Any,
    runtime_used: Any,
) -> Dict[str, str]:
    """Preserve a failed preferred-runtime attempt when a fallback succeeds."""
    requested = _runtime_identity(requested_runtime)
    used = _runtime_identity(runtime_used)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    reason = str(
        _runtime_field_to_str(result.get("fallback_reason"))
        or _runtime_field_to_str(metrics.get("fallback_reason"))
        or ""
    ).strip()
    if not requested or not used or requested == used or not reason:
        return {}
    error_text = "\n".join(
        str(result.get(key) or "")
        for key in ("primary_error_tail", "stderr_tail", "stderr", "stdout_tail", "stdout")
    ).strip()
    return {
        "requested_runtime": requested,
        "runtime_used": used,
        "fallback_reason": reason,
        "error_text": error_text[-8000:],
    }


def _runtime_failure_observation_context(
    result: Dict[str, Any],
    *,
    requested_runtime: Any,
    requested_artifact: str,
    attempted_runtime: Any,
    attempted_artifact: str,
) -> Dict[str, str]:
    """Keep a failed preferred runtime attached to its own artifact."""
    fallback = _runtime_fallback_evidence(
        result,
        requested_runtime=requested_runtime,
        runtime_used=attempted_runtime,
    )
    if fallback:
        return {
            "runtime": fallback["requested_runtime"],
            "artifact_path": requested_artifact,
            "failure_stage": "runtime_fallback",
            "error_text": fallback["error_text"] or fallback["fallback_reason"],
            "fallback_reason": fallback["fallback_reason"],
        }
    return {
        "runtime": str(_runtime_field_to_str(attempted_runtime) or requested_runtime or ""),
        "artifact_path": attempted_artifact,
        "failure_stage": str(result.get("failed_stage") or ""),
        "error_text": "\n".join(
            str(result.get(key) or "")
            for key in (
                "error",
                "primary_error_tail",
                "stderr_tail",
                "stderr",
                "stdout_tail",
                "stdout",
            )
        ),
        "fallback_reason": str(result.get("fallback_reason") or ""),
    }


def _result_output_text(result: Dict[str, Any], stream: str) -> str:
    """Read process output whether the runner retained it whole or as a tail."""
    output = str(result.get(stream) or result.get(f"{stream}_tail") or "")
    if stream != "stderr":
        return output

    primary_error = str(result.get("primary_error_tail") or "").strip()
    if not primary_error or primary_error in output:
        return output
    return "\n".join(part for part in (output.strip(), primary_error) if part)


def _record_runtime_attempt(
    trial: TrialResult,
    *,
    runtime: str,
    artifact_path: Optional[str],
    provider: Optional[str] = None,
    status: str,
    error: Optional[str] = None,
    fallback_reason: Optional[str] = None,
) -> None:
    trial.runtime_report.attempts.append(
        RuntimeAttempt(
            runtime=runtime,
            artifact_path=artifact_path,
            provider=_runtime_field_to_str(provider),
            status=status,
            error=error,
            fallback_reason=_runtime_field_to_str(fallback_reason),
        )
    )


def _resolve_and_stage_model(
    raw_model_path: str,
    ws: Path,
    trial_id: str,
    variant: Optional[SolutionVariant] = None,
) -> str:
    """Resolve the model path from train.py output and ensure it exists in ws/outputs/.

    train.py may report a relative path (e.g. "outputs/best.pt") or an absolute
    path under a nested Ultralytics run directory.  This helper:
      1. Resolves relative paths against *ws*.
      2. Falls back to searching ``ws/runs/detect/*/weights/best.*``.
      3. Copies the found model into ``ws/outputs/`` so EdgeRunner can locate it.
      4. If the variant needs an ONNX/engine export and only .pt exists, exports it.
    """
    import shutil
    import glob as glob_mod

    staged_dir = ws / "outputs"
    staged_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidate model files
    candidates: list[Path] = []
    if raw_model_path:
        p = Path(raw_model_path)
        if p.is_absolute() and p.is_file():
            candidates.append(p)
        elif (ws / p).is_file():
            candidates.append(ws / p)

    export_fmt = (variant.export_format if variant else "").lower()

    # Generated training code often already exports the desired deploy artifact
    # into outputs/. Prefer that explicit artifact before falling back to a
    # heavier framework checkpoint such as best.pt.
    preferred_existing: list[Path] = []
    if export_fmt == "engine":
        preferred_existing.extend([staged_dir / "best.onnx", staged_dir / "best.engine"])
    elif export_fmt:
        preferred_existing.append(staged_dir / f"best.{export_fmt}")
    preferred_existing.extend([staged_dir / "best.onnx", staged_dir / "best.pt"])
    for existing in preferred_existing:
        if existing.is_file() and existing not in candidates:
            candidates.insert(0, existing)
            break

    # Fallback: search Ultralytics-style output directories
    if not candidates:
        for pattern in ["runs/detect/*/weights/best.*", "runs/*/weights/best.*", "runs/**/weights/best.*"]:
            for hit in sorted(glob_mod.glob(str(ws / pattern), recursive="**" in pattern), reverse=True):
                candidates.append(Path(hit))

    if not candidates:
        fallback = str(staged_dir / "best.pt")
        logger.warning(f"[{trial_id}] _resolve_and_stage_model: no model found, using fallback {fallback}")
        return fallback

    # Stage the primary model
    model = candidates[0]
    dst = staged_dir / model.name
    if not dst.exists():
        shutil.copy2(model, dst)
        logger.debug(f"[{trial_id}] Staged model: {model} → {dst}")

    # If the variant requires a different format, prefer already-produced
    # artifacts. In particular, TensorRT engines are device/version-bound and
    # must be built by EdgeRunner on the target edge device from best.onnx.
    if export_fmt in ("onnx", "engine") and dst.suffix == ".pt":
        target = staged_dir / f"best.{export_fmt}"
        if target.exists():
            return str(target)
        if export_fmt == "engine":
            onnx_target = staged_dir / "best.onnx"
            if onnx_target.exists():
                logger.debug(
                    f"[{trial_id}] Deploy target is engine; using {onnx_target} "
                    "for edge-side TensorRT build."
                )
                return str(onnx_target)
            logger.warning(
                f"[{trial_id}] Deploy target is engine but outputs/best.onnx is missing; "
                "skipping host-side engine export and returning PT artifact."
            )
            return str(dst)
        if not target.exists():
            logger.debug(f"[{trial_id}] Attempting {export_fmt} export from {dst}")
            try:
                export_result = subprocess.run(
                    [
                        sys.executable, "-c",
                        f"from ultralytics import YOLO; "
                        f"m = YOLO('{dst}'); "
                        f"m.export(format='{export_fmt}', half=False, opset=12)",
                    ],
                    capture_output=True, text=True, timeout=300, cwd=str(staged_dir),
                )
                exported = list(staged_dir.glob(f"best.{export_fmt}"))
                if not exported:
                    exported = list(staged_dir.glob("best*." + export_fmt))
                if exported:
                    logger.debug(f"[{trial_id}] Export succeeded: {exported[0]}")
                    return str(exported[0])
                else:
                    logger.warning(
                        f"[{trial_id}] Export to {export_fmt} produced no file. "
                        f"stderr: {export_result.stderr[-500:]}"
                    )
            except Exception as exc:
                logger.warning(f"[{trial_id}] Export to {export_fmt} failed: {exc}")

    return str(dst)


def _reported_workspace_artifact(ws: Path, train_result: Dict[str, Any]) -> str:
    """Return a reported artifact only when it belongs to this workspace."""
    raw_model_path = str(train_result.get("model_path") or "").strip()
    if not raw_model_path:
        return ""

    reported = Path(raw_model_path)
    reported = reported.resolve() if reported.is_absolute() else (ws / reported).resolve()
    try:
        reported.relative_to(ws.resolve())
    except ValueError:
        return ""
    if not reported.is_file():
        return ""
    return str(reported)


def _recover_staged_artifact_after_train_failure(
    trial: TrialResult,
    ws: Path,
    variant: SolutionVariant,
    train_result: Dict[str, Any],
) -> str:
    """Preserve an explicitly reported artifact after a non-success exit.

    The script result, not a workspace scan, defines the artifact boundary. This
    prevents an old or intermediate checkpoint from silently turning an
    unrelated training error into a successful trial.
    """
    reported = _reported_workspace_artifact(ws, train_result)
    if not reported:
        return ""

    recovered = _resolve_and_stage_model(reported, ws, variant.trial_id, variant)
    if not recovered or not Path(recovered).is_file():
        return ""

    trial.artifact_paths["train"] = recovered
    _record_workspace_artifacts(trial, ws, variant)
    deploy_artifact = _primary_deploy_artifact(trial)
    if deploy_artifact:
        trial.artifact_paths["train"] = deploy_artifact
    trial.failure_context["train_artifact_recovery"] = {
        "status": "recovered_real_artifact",
        "artifact_path": trial.artifact_paths.get("train") or recovered,
        "original_error": train_result.get("error", ""),
    }
    _record_trial_observations(trial)
    return trial.artifact_paths.get("train") or recovered


def _full_efficiency_replay_reasons(
    edge_result: Dict[str, Any],
    device_capabilities: Dict[str, Any],
) -> List[str]:
    """Return missing comparable facts that justify one artifact-only replay."""
    raw_metrics = edge_result.get("metrics") if isinstance(edge_result.get("metrics"), dict) else {}
    metrics = MetricRegistry.normalize_metric_dict(raw_metrics)
    protocol = (
        edge_result.get("measurement_protocol")
        if isinstance(edge_result.get("measurement_protocol"), dict)
        else {}
    )
    reasons: List[str] = []
    if MetricRegistry.lookup_value("Latency_p95", metrics) is None:
        reasons.append("latency_p95_missing")
    elif protocol.get("efficiency_source") != "evaluator_runtime_replay":
        reasons.append("latency_protocol_not_evaluator_owned")
    if device_capabilities.get("energy_sampler") and protocol.get("energy_trusted") is not True:
        reasons.append("energy_missing_or_untrusted")
    return reasons


def _merge_full_efficiency_replay(
    edge_result: Dict[str, Any],
    replay_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge evaluator-owned efficiency facts without touching task quality."""
    merged = dict(edge_result)
    primary_raw = (
        dict(edge_result.get("metrics"))
        if isinstance(edge_result.get("metrics"), dict)
        else {}
    )
    replay_raw = (
        dict(replay_result.get("metrics"))
        if isinstance(replay_result.get("metrics"), dict)
        else {}
    )
    replay_metrics = {
        name: value
        for name, value in MetricRegistry.normalize_metric_dict(replay_raw).items()
        if MetricRegistry.is_edge_metric(name)
    }
    replaced = set(replay_metrics)
    preserved = {
        name: value
        for name, value in primary_raw.items()
        if MetricRegistry.canonicalize_name(str(name)) not in replaced
    }
    merged["metrics"] = {**preserved, **replay_metrics}

    primary_protocol = (
        dict(edge_result.get("measurement_protocol"))
        if isinstance(edge_result.get("measurement_protocol"), dict)
        else {}
    )
    replay_protocol = (
        dict(replay_result.get("measurement_protocol"))
        if isinstance(replay_result.get("measurement_protocol"), dict)
        else {}
    )
    merged["measurement_protocol"] = {
        **replay_protocol,
        "efficiency_source": "evaluator_runtime_replay",
        "quality_measurement_protocol": primary_protocol,
        "efficiency_diagnostics": {
            name: value
            for name, value in replay_raw.items()
            if name in {
                "Dynamic_power_w",
                "Dynamic_energy_mj",
                "power_std_w",
                "energy_std_mj",
                "energy_sample_count",
            }
        },
    }
    return merged


def _run_full_efficiency_replay(
    *,
    trial: TrialResult,
    state: AgentState,
    variant: SolutionVariant,
    ws: Path,
    runner: Any,
    model_path: str,
    edge_result: Dict[str, Any],
) -> tuple[Dict[str, Any], float]:
    """Produce comparable efficiency evidence with the evaluator-owned probe."""
    capabilities = runner.device_capabilities(state.get("target_device", ""))
    reasons = _full_efficiency_replay_reasons(edge_result, capabilities)
    if not reasons or not model_path or not Path(model_path).is_file():
        return edge_result, 0.0

    fingerprint = build_artifact_fingerprint(model_path).model_dump(mode="json")
    infer_path = write_runtime_repro_infer(
        str(ws / ".edgecraft" / "full_efficiency_infer.py"),
        input_specs=list(fingerprint.get("input_specs") or []),
        warmup=EFFICIENCY_PROBE_WARMUP,
        min_warmup_seconds=EFFICIENCY_PROBE_MIN_WARMUP_SECONDS,
        repetitions=EFFICIENCY_PROBE_REPETITIONS,
        min_measure_seconds=EFFICIENCY_PROBE_MIN_MEASURE_SECONDS,
        measurement_sessions=EFFICIENCY_PROBE_SESSIONS,
    )
    started = time.perf_counter()
    try:
        replay = runner.run(
            artifact_path=model_path,
            device_id=state.get("target_device", ""),
            device_ip=state.get("device_ip", ""),
            ssh_key=state.get("ssh_key", ""),
            infer_script=infer_path,
            timeout=int(getattr(settings, "L1_EDGE_TIMEOUT_S", 600)),
            docker_image=state.get("docker_image"),
            export_format=getattr(variant, "export_format", None) or artifact_kind(model_path) or "onnx",
            quant_mode=getattr(variant, "quant_mode", None) or "fp16",
            graph_hash=str(fingerprint.get("graph_hash") or ""),
            artifact_manifest=trial.artifact_contract.model_dump(mode="json"),
            require_exact_runtime=True,
        )
    except Exception as exc:  # noqa: BLE001
        replay = {
            "status": "error",
            "error": f"evaluator efficiency replay failed: {exc}",
            "failed_stage": "full_efficiency_replay",
        }
    elapsed_s = time.perf_counter() - started
    execution = _record_stage_execution_evidence(
        trial,
        stage="full_efficiency_replay",
        result={**replay, "elapsed_s": elapsed_s},
        resource="device",
    )
    replay_raw = replay.get("metrics") if isinstance(replay.get("metrics"), dict) else {}
    artifact_executed, execution_reasons = _artifact_execution_evidence(replay_raw)
    replay_protocol = (
        replay.get("measurement_protocol")
        if isinstance(replay.get("measurement_protocol"), dict)
        else {}
    )
    runtime_fields = _resolve_edge_runtime_fields(replay)
    requested_runtime = str(
        getattr(variant, "export_format", None)
        or artifact_kind(model_path)
        or ""
    )
    attested = bool(
        replay.get("status") == "success"
        and artifact_executed
        and replay.get("require_exact_runtime") is True
        and not replay.get("fallback_reason")
        and _runtime_identity(runtime_fields["runtime_used"])
        == _runtime_identity(requested_runtime)
        and str(replay.get("source_artifact_hash") or "")
        == str(fingerprint.get("artifact_hash") or "")
        and str(replay.get("requested_graph_hash") or "")
        == str(fingerprint.get("graph_hash") or "")
    )
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["full_efficiency_replay"] = {
        "attempted": True,
        "reasons": reasons,
        "status": replay.get("status"),
        "artifact_executed": artifact_executed,
        "execution_reasons": execution_reasons,
        "energy_trusted": replay_protocol.get("energy_trusted") is True,
        "attested_exact_artifact": attested,
        "execution_evidence_id": execution.id,
        "elapsed_s": elapsed_s,
        "error": replay.get("error"),
    }
    trial.observations = observations
    if not attested:
        return edge_result, elapsed_s
    return _merge_full_efficiency_replay(edge_result, replay), elapsed_s


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def _knowledge_retrieval_observation(state: AgentState) -> Dict[str, Any]:
    case_ids = list(dict.fromkeys(state.get("retrieved_case_ids") or []))
    rule_ids = list(dict.fromkeys(state.get("retrieved_rule_ids") or []))
    if not case_ids and not rule_ids:
        return {}
    return {
        "case_ids": case_ids,
        "rule_ids": rule_ids,
        "decision_authority": "prompt_evidence_only",
    }

def pipeline_executor_node(state: AgentState) -> AgentState:
    """Pipeline executor node: runs current_variant through the 3-stage pipeline."""
    if state.get("status") in {"failed", "completed"}:
        return state
    variant: Optional[SolutionVariant] = state.get("current_variant")
    if variant is None:
        logger.error("pipeline_executor_node called with no current_variant!")
        state["status"] = "failed"
        state["error"] = "No variant to execute."
        return state

    run_id = state["run_id"]
    start_time = time.time()
    trial = TrialResult(trial_id=variant.trial_id, variant=variant)
    trial.stage_reached = StageReached.PENDING
    trial.crafting_progress = CraftingProgress.PENDING
    retrieval = _knowledge_retrieval_observation(state)
    if retrieval:
        trial.observations["knowledge_retrieval"] = retrieval

    modality: Modality = state["modality"]
    user_spec: Optional[UserSpec] = state.get("user_spec")
    handler = ModalityHandlerRegistry.get(modality)

    # ------------------------------------------------------------------
    # Stage 1: DataPrep
    # ------------------------------------------------------------------
    ws_mgr = WorkspaceManager()
    ws = ws_mgr.create_trial_workspace(
        run_id,
        variant.trial_id,
        tenant_id=state.get("tenant_id") or "default",
    )
    trial.workspace_path = str(ws)

    try:
        di = state.get("dataset_info") or {}
        train_data_path = di.get("config_path") or state.get("dataset_path") or ""
        ws_mgr.materialize_variant(
            variant, ws,
            train_data_path,
            handler,
        )
        split_manifest = di.get("split_manifest") if isinstance(di, dict) else None
        if isinstance(split_manifest, dict) and split_manifest.get("path"):
            cfg_path = ws / "config" / "data.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cfg["split_manifest"] = dict(split_manifest)
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            generated_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (ws / "loader.py", ws / "train.py")
                if path.is_file()
            )
            referenced_by_code = "split_manifest" in generated_text
            manifest_payload = yaml.safe_load(
                Path(split_manifest["path"]).read_text(encoding="utf-8")
            )
            materialized_by_config = (
                isinstance(manifest_payload, dict)
                and _config_covers_split_manifest(cfg, manifest_payload)
            )
            referenced = referenced_by_code or materialized_by_config
            trial.failure_context["split_manifest_audit"] = {
                "content_hash": split_manifest.get("content_hash"),
                "declared": True,
                "referenced_by_generated_code": referenced_by_code,
                "materialized_by_config": materialized_by_config,
                "consumed": referenced,
                "decision_authority": (
                    "evaluation_contract"
                    if settings.REQUIRE_SPLIT_MANIFEST_CONSUMPTION
                    else "audit_only"
                ),
            }
            if settings.REQUIRE_SPLIT_MANIFEST_CONSUMPTION and not referenced:
                raise ValueError(
                    "split manifest is required but neither generated code nor config split paths consume it"
                )
    except Exception as exc:
        trial.error = f"DataPrep failed: {exc}"
        trial.error_stage = "data_prep"
        _record_failure(trial, trial.error)
        if "split manifest is required" in str(exc).lower():
            trial.next_search_hint = (
                "Read config/data.yaml split_manifest and use its exact train/val/test "
                "sample IDs; do not resplit the dataset."
            )
        trial.duration_seconds = time.time() - start_time
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    trial.stage_reached = StageReached.DATA_PREP
    trial.crafting_progress = CraftingProgress.SCRIPT_GENERATED
    _record_constraint_adherence(trial, ws, state)
    if _evidence_ladder_enabled():
        _append_evidence(
            trial,
            Evidence(
                probe_id="static_workspace",
                quantity="workspace_contract",
                fidelity="static",
                outcome="pass",
                protocol={"files": sorted(_workspace_hashes(ws))},
            ),
        )
        report = _ensure_verification_report(trial)
        report.level = "L0"
        report.status = "pass"
        report.evidence_source = "static"
        report.artifact_status = "workspace_materialized"
        report.runtime_status = "not_run"
        report.source = "static_workspace_check"
        _apply_verifier_policy(trial, user_spec, next_probe="efficiency")
        _record_trial_observations(trial)
    logger.debug(f"[{variant.trial_id}] Stage 1 DATA_PREP done: {ws}")

    # ------------------------------------------------------------------
    # Optional LoaderSmoke: validate dataset loading before training.
    # ------------------------------------------------------------------
    max_debug_retries = max(0, int(getattr(settings, "DEBUGGER_MAX_RETRIES", 2)))
    loader_script = ws / "loader.py"
    loader_return_probe: Optional[Dict[str, Any]] = None
    if loader_script.exists():
        logger.debug(f"[{variant.trial_id}] Optional LOADER_SMOKE: executing loader.py --smoke")
        loader_result = _run_loader_smoke_with_debugger(
            loader_script=loader_script,
            ws=ws,
            state=state,
            variant=variant,
            trial=trial,
            max_debug_retries=max_debug_retries,
        )
        loader_execution = _record_stage_execution_evidence(
            trial,
            stage="loader",
            result=loader_result,
        )
        _record_loader_feature_provenance(trial, loader_result)
        loader_contract = Evidence(
            probe_id="component_roundtrip",
            quantity="component_contract",
            fidelity="proxy",
            outcome="pass" if loader_result.get("status") == "success" else "fail",
            protocol={
                "boundary": "dataset_to_loader",
                "real_dataset_sample": bool(loader_result.get("loader_return_probe")),
                "execution_evidence_id": loader_execution.id,
                "decision_authority": "prompt_only",
            },
        )
        _append_evidence(trial, loader_contract)
        trial.failure_context["loader_smoke"] = {
            "status": loader_result.get("status"),
            "error": loader_result.get("error"),
            "stdout_tail": (loader_result.get("stdout", "") or "")[-1200:],
            "stderr_tail": (loader_result.get("stderr", "") or "")[-1200:],
            "loader_return_probe": loader_result.get("loader_return_probe"),
            "loader_return_probe_error": loader_result.get("loader_return_probe_error"),
            "sample_observation": (loader_result.get("loader_return_probe") or {}).get("sample_observation"),
            "elapsed_s": loader_result.get("elapsed_s"),
        }
        _record_trial_observations(
            trial,
            loader_smoke=trial.failure_context["loader_smoke"],
        )
        if loader_result.get("status") != "success":
            _record_workspace_artifacts(trial, ws, variant)
            trial.error = loader_result.get("error", "loader smoke failed")
            trial.error_stage = "loader"
            _record_failure(
                trial,
                trial.error,
                loader_result.get("stderr", ""),
                loader_result.get("stdout", ""),
            )
            trial.failure_context["repair_diagnosis"] = _build_repair_diagnosis(
                stage="loader",
                component="loader.py",
                result=loader_result,
                trial=trial,
                sample_observation=(loader_result.get("loader_return_probe") or {}).get("sample_observation"),
            )
            trial.next_search_hint = trial.failure_context["repair_diagnosis"].get("next_search_hint")
            trial.failure_context["diagnostics"] = _collect_failure_diagnostics(
                ws=ws,
                state=state,
                stage="loader",
                result=loader_result,
            )
            _record_trial_observations(
                trial,
                diagnostics=trial.failure_context["diagnostics"],
            )
            trial.duration_seconds = time.time() - start_time
            _log_trial_summary(variant, trial)
            state["current_trial_result"] = trial
            state["status"] = "scoring"
            return state
        loader_return_probe = loader_result.get("loader_return_probe")

    # ------------------------------------------------------------------
    # Stage 2: Train (subprocess)
    # ------------------------------------------------------------------
    train_script = ws / "train.py"
    if not train_script.exists():
        trial.error = "train.py not found in workspace"
        trial.error_stage = "train"
        _record_failure(trial, trial.error)
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    component_contract_error, component_contract_evidence = (
        _record_loader_train_contract_evidence(
            trial,
            ws,
            loader_return_probe,
            loader_result.get("loader_return_probe_error") if loader_script.exists() else None,
        )
    )
    if component_contract_error:
        _record_workspace_artifacts(trial, ws, variant)
        trial.error = component_contract_error
        trial.error_stage = "train"
        trial.failure_context["component_roundtrip"] = {
            "boundary": "loader_to_train",
            "evidence_id": component_contract_evidence.id,
            "loader_return_summary": (loader_return_probe or {}).get("return_summary"),
            "error": component_contract_error,
        }
        _record_failure(trial, component_contract_error)
        trial.next_search_hint = (
            "Make train.py consume the measured loader.py return structure without "
            "changing the verified dataset split or loader semantics."
        )
        _record_trial_observations(trial)
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    inherited_train_result = _reuse_parent_training_result(
        trial=trial,
        variant=variant,
        ws=ws,
        state=state,
    )
    if inherited_train_result is None and bool(state.get("require_parent_training_reuse")):
        evidence = Evidence(
            probe_id="parent_training_reuse",
            quantity="component_contract",
            fidelity="static",
            outcome="fail",
            protocol={
                "parent_trial_id": str(getattr(variant, "parent_trial_id", "") or ""),
                "decision_authority": "replay_safety_only",
                "reason": "parent training inputs or artifacts are not congruent",
            },
        )
        _append_evidence(trial, evidence)
        trial.error = (
            "Artifact-only edge replay requires congruent parent training evidence; "
            "refusing to retrain or substitute another artifact."
        )
        trial.error_stage = "training_reuse"
        trial.stage_reached = StageReached.TRAIN
        trial.failure_context["parent_training_reuse"] = {
            "evidence_id": evidence.id,
            "parent_trial_id": str(getattr(variant, "parent_trial_id", "") or ""),
        }
        _record_trial_observations(trial)
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state
    if inherited_train_result is None and _run_l1_efficiency_probe(
        trial=trial,
        state=state,
        variant=variant,
        ws=ws,
        train_script=train_script,
        user_spec=user_spec,
    ):
        _record_trial_observations(trial)
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    if inherited_train_result is None:
        _run_quality_proxy_probe(
            trial=trial,
            train_script=train_script,
            ws=ws,
            state=state,
        )
    else:
        logger.info(
            f"[{variant.trial_id}] Reusing measured training artifact from "
            f"{inherited_train_result['reused_parent_trial_id']}; train inputs are unchanged."
        )

    if inherited_train_result is None and _use_scheduler_backend():
        logger.debug(
            f"[{variant.trial_id}] EXECUTION_BACKEND=scheduler: submitting full P2 job"
        )
        trial = _run_trial_via_scheduler(
            state=state,
            variant=variant,
            trial=trial,
            ws=ws,
            start_time=start_time,
        )
        _finalize_scheduler_verification(
            trial=trial,
            state=state,
            variant=variant,
            user_spec=user_spec,
            ws=ws,
        )
        _record_trial_observations(trial)
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    if inherited_train_result is None:
        logger.debug(f"[{variant.trial_id}] Stage 2 TRAIN: executing train.py")
    else:
        logger.debug(f"[{variant.trial_id}] Stage 2 TRAIN: reused parent artifact")
    train_result: Dict[str, Any] = inherited_train_result or {}
    completed_training_evidence: Dict[str, Any] = {}
    retry_script_args: List[str] = []
    train_stage_started = time.perf_counter()
    retry_range = range(0) if inherited_train_result is not None else range(max_debug_retries + 1)
    for retry_idx in retry_range:
        train_loader_contract_error = (
            _validate_loader_train_contract_from_evidence(ws, loader_return_probe)
            if retry_idx > 0
            else None
        )
        if train_loader_contract_error:
            train_result = {
                "status": "error",
                "error": train_loader_contract_error,
                "stdout": "",
                "stderr": "",
            }
        else:
            blocked_api = _contains_blocked_transformers_export(train_script.read_text())
            if blocked_api:
                train_result = {
                    "status": "error",
                    "error": "blocked_unstable_export_api: use stable_hf_export_onnx template path",
                    "stdout": "",
                    "stderr": "",
                }
            else:
                artifact_only_retry = retry_script_args == ["--export-only"]
                train_result = _run_script(
                    train_script,
                    ws,
                    timeout=int(getattr(settings, "FULL_TRAIN_EXPORT_TIMEOUT_S", 7200)),
                    script_args=retry_script_args,
                )
                retry_script_args = []
                if train_result.get("status") == "success":
                    if artifact_only_retry:
                        train_result = _merge_export_retry_with_training_evidence(
                            train_result,
                            completed_training_evidence,
                        )
                    else:
                        completed_training_evidence = _training_evidence_from_result(train_result)
                    train_result = _train_export_smoke_check(ws, train_result)
        trial.stdout = train_result.get("stdout", "")
        trial.stderr = train_result.get("stderr", "")
        if train_result.get("status") == "success":
            _mark_debug_retry_success(trial, "train")
            break

        if _reported_workspace_artifact(ws, train_result):
            logger.warning(
                f"[{variant.trial_id}] train.py reported a real artifact with a "
                "non-success result; preserving the execution evidence and skipping repair."
            )
            break

        err_msg = train_result.get("error", "train.py failed")
        err_blob = f"{err_msg}\n{train_result.get('stderr', '')}\n{train_result.get('stdout', '')}".lower()
        if "out of memory" in err_blob and retry_idx < max_debug_retries:
            if _apply_oom_batch_fallback(train_script):
                trial.debug_attempts += 1
                trial.debug_history.append(
                    DebugAttempt(
                        stage="train",
                        attempt=trial.debug_attempts,
                        error_signature="oom_downscale",
                        reason="auto_batch_halving_after_oom",
                        before_hash="",
                        after_hash="",
                        patch_summary="Halved batch size after CUDA OOM and retried.",
                        patch_applied=True,
                        success=False,
                    )
                )
                logger.warning(
                    f"[{variant.trial_id}] CUDA OOM detected; halved batch size and retrying train.py."
                )
                continue
        is_repairable, signature = _is_repairable_failure(err_msg, train_result.get("stderr", ""))
        if retry_idx >= max_debug_retries or not is_repairable:
            break
        diagnostics = _collect_failure_diagnostics(
            ws=ws,
            state=state,
            stage="train",
            result=train_result,
        )
        _record_trial_observations(trial, diagnostics=diagnostics)
        repair_script = _select_generated_failure_component(
            ws,
            train_result,
            default=train_script,
        )
        repair_diagnosis = _build_repair_diagnosis(
            stage="train",
            component=repair_script.name,
            result=train_result,
            trial=trial,
            sample_observation=diagnostics.get("sample_observation") if isinstance(diagnostics, dict) else None,
        )
        if repair_diagnosis.get("repair_action") != "patch_current_script":
            trial.failure_context["repair_diagnosis"] = repair_diagnosis
            trial.next_search_hint = repair_diagnosis.get("next_search_hint")
            break

        reusable_checkpoints = _existing_train_checkpoints(ws)
        patched, summary, before_hash, after_hash, reason = _run_llm_debugger(
            script_path=repair_script,
            stage="train",
            variant=variant,
            error=err_msg,
            stdout=train_result.get("stdout", ""),
            stderr=train_result.get("stderr", ""),
            error_signature=signature,
            diagnostic_context=diagnostics,
            debug_history=trial.debug_history,
        )
        reason = f"{reason}; component={repair_script.name}"
        trial.debug_attempts += 1
        trial.debug_history.append(
            DebugAttempt(
                stage="train",
                attempt=trial.debug_attempts,
                error_signature=signature,
                reason=reason,
                before_hash=before_hash,
                after_hash=after_hash,
                patch_summary=summary,
                patch_applied=patched,
                success=False,
            )
        )
        if not patched:
            break
        supported_export_only_after = _supports_export_only(train_script.read_text())
        if reusable_checkpoints and supported_export_only_after:
            retry_script_args = ["--export-only"]
            trial.failure_context["artifact_aware_retry"] = {
                "mode": "export_only",
                "reused_artifacts": reusable_checkpoints,
                "debug_attempt": trial.debug_attempts,
            }
            logger.info(
                f"[{variant.trial_id}] Debugger added --export-only and preserved "
                f"{reusable_checkpoints[0]}; retrying export without training."
            )
        else:
            logger.info(
                f"[{variant.trial_id}] Debugger patched {repair_script.name} "
                f"(attempt {retry_idx + 1}/{max_debug_retries}). Retrying."
            )

    _record_stage_execution_evidence(
        trial,
        stage="full_train_export",
        result={
            **train_result,
            "elapsed_s": time.perf_counter() - train_stage_started,
            "timeout_s": train_result.get("timeout_s")
            or int(getattr(settings, "FULL_TRAIN_EXPORT_TIMEOUT_S", 7200)),
        },
        resource="" if inherited_train_result is not None else "gpu",
    )
    train_wall_s = time.perf_counter() - train_stage_started
    train_result, trace_recovery = _recover_training_trace_sidecar(train_result, ws)
    if trace_recovery is not None:
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["training_trace_recovery"] = trace_recovery
        trial.observations = observations
    metrics: Dict[str, float] = {}
    if train_result.get("status") == "success" or _training_evidence_from_result(train_result):
        metrics = _record_training_result_evidence(trial, state, train_result)

    if train_result.get("status") != "success":
        recovered_artifact = _recover_staged_artifact_after_train_failure(trial, ws, variant, train_result)
        if recovered_artifact:
            logger.warning(
                f"[{variant.trial_id}] train.py exited with an error after producing "
                f"a real artifact; staged {recovered_artifact} and continuing to edge."
            )
            train_result = {
                **train_result,
                "status": "success",
                "model_path": recovered_artifact,
                "metrics": train_result.get("metrics", {}),
            }
        else:
            _record_workspace_artifacts(trial, ws, variant)
            if trial.artifact_contract.primary_artifact:
                trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
            trial.error = train_result.get("error", "train.py failed")
            trial.error_stage = "train"
            _record_failure(
                trial,
                trial.error,
                train_result.get("stderr", ""),
                train_result.get("stdout", ""),
            )
            diagnostics = _collect_failure_diagnostics(
                ws=ws,
                state=state,
                stage="train",
                result=train_result,
            )
            repair_diagnosis = trial.failure_context.get("repair_diagnosis") or _build_repair_diagnosis(
                stage="train",
                component="train.py",
                result=train_result,
                trial=trial,
                sample_observation=diagnostics.get("sample_observation") if isinstance(diagnostics, dict) else None,
            )
            trial.next_search_hint = repair_diagnosis.get("next_search_hint") or trial.next_search_hint
            trial.failure_context.update({
                "assigned_gpu": train_result.get("assigned_gpu"),
                "stderr_tail": (train_result.get("stderr", "") or "")[-1200:],
                "stdout_tail": (train_result.get("stdout", "") or "")[-1200:],
                "repair_diagnosis": repair_diagnosis,
                "diagnostics": diagnostics,
            })
            _record_trial_observations(trial, diagnostics=diagnostics)
            trial.duration_seconds = time.time() - start_time
            _log_trial_summary(variant, trial)
            state["current_trial_result"] = trial
            state["status"] = "scoring"
            return state

    raw_model_path = train_result.get("model_path", "")
    resolved = _resolve_and_stage_model(raw_model_path, ws, variant.trial_id, variant)
    trial.artifact_paths["train"] = resolved
    _record_workspace_artifacts(trial, ws, variant)
    declared_artifacts = train_result.get("artifact_paths")
    _record_declared_artifacts(trial, ws, declared_artifacts)
    bundle_contract = _audit_edge_eval_bundle(trial, ws, state, declared_artifacts)
    _record_trial_observations(trial)
    strict_bundle_error = _strict_edge_eval_bundle_error(bundle_contract)
    if strict_bundle_error:
        trial.error = strict_bundle_error
        trial.error_stage = "edge_eval_contract"
        trial.next_search_hint = (
            "Preserve the trained model and artifact path; make edge_eval_manifest, "
            "edge_eval_payload, and infer.py agree on one self-contained payload contract."
        )
        trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
        trial.stage_reached = StageReached.TRAIN
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state
    deploy_artifact = _primary_deploy_artifact(trial)
    if deploy_artifact:
        trial.artifact_paths["train"] = deploy_artifact
        trial.runtime_report.requested_runtime = getattr(variant, "export_format", None) or artifact_kind(deploy_artifact)
        previous_report = getattr(trial, "verification_report", None)
        if previous_report is not None and previous_report.level == "L1":
            l1_fp = previous_report.artifact_fingerprint or {}
            l2_fp = build_artifact_fingerprint(deploy_artifact).model_dump(mode="json")
            congruence = compare_artifact_fingerprints(l1_fp, l2_fp)
            observations = dict(getattr(trial, "observations", {}) or {})
            observations["l1_l2_congruence"] = {
                **congruence,
                "l1_artifact_mismatch": bool(congruence.get("checked") and not congruence.get("match")),
                "l1_fingerprint": l1_fp,
                "l2_fingerprint": l2_fp,
            }
            if (observations.get("l1_efficiency") or {}).get("false_prune_check"):
                observations["l1_efficiency"]["false_prune_l2_completed"] = True
            trial.observations = observations
        payload_contract = _artifact_eval_payload_contract(
            trial=trial,
            artifact_path=deploy_artifact,
        )
        host_contract, local_contract_valid = _run_host_infer_contract_with_repair(
            trial=trial,
            variant=variant,
            ws=ws,
            artifact_path=deploy_artifact,
            payload_contract=payload_contract,
        )
        if not local_contract_valid:
            trial.error = (
                "Artifact and evaluator payload have a measured interface mismatch; "
                "local infer repair did not produce an executable round trip."
            )
            trial.error_stage = "artifact"
            trial.runtime_report.artifact_used = deploy_artifact
            trial.runtime_report.fallback_reason = "artifact_eval_payload_contract"
            trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
            trial.stage_reached = StageReached.TRAIN
            trial.duration_seconds = time.time() - start_time
            _log_trial_summary(variant, trial)
            state["current_trial_result"] = trial
            state["status"] = "scoring"
            return state
        requested_runtime = str(getattr(variant, "export_format", None) or "")
        if (
            host_contract is not None
            and host_contract.outcome != "pass"
            and not _artifact_matches_requested_runtime(deploy_artifact, requested_runtime)
        ):
            actual_runtime = artifact_kind(deploy_artifact)
            mismatch = Evidence(
                probe_id="artifact_runtime_contract",
                quantity="artifact_contract",
                fidelity="static",
                outcome="fail",
                artifact_fingerprint=build_artifact_fingerprint(deploy_artifact).model_dump(mode="json"),
                protocol={
                    "requested_runtime": requested_runtime,
                    "produced_artifact_kind": actual_runtime,
                    "host_contract_evidence_id": host_contract.id,
                    "decision_authority": "pipeline_contract",
                },
            )
            _append_evidence(trial, mismatch)
            trial.error = (
                "Requested deployment artifact was not produced: "
                f"requested={requested_runtime}, available={actual_runtime}; "
                "generated infer.py could not execute the fallback artifact."
            )
            trial.error_stage = "artifact"
            trial.runtime_report.requested_runtime = requested_runtime
            trial.runtime_report.artifact_used = deploy_artifact
            trial.runtime_report.fallback_reason = "requested_artifact_missing"
            _record_runtime_attempt(
                trial,
                runtime=requested_runtime,
                artifact_path=deploy_artifact,
                status="failed",
                error=trial.error,
                fallback_reason="requested_artifact_missing",
            )
            observations = dict(getattr(trial, "observations", {}) or {})
            observations["artifact_runtime_contract"] = {
                "status": "fail",
                "requested_runtime": requested_runtime,
                "produced_artifact_kind": actual_runtime,
                "host_contract_evidence_id": host_contract.id,
                "evidence_id": mismatch.id,
            }
            trial.observations = observations
            trial.crafting_progress = CraftingProgress.TRAIN_SUCCEEDED
            trial.stage_reached = StageReached.TRAIN
            trial.duration_seconds = time.time() - start_time
            _log_trial_summary(variant, trial)
            state["current_trial_result"] = trial
            state["status"] = "scoring"
            return state
        _run_evaluation_metric_contract(
            trial=trial,
            ws=ws,
            state=state,
            artifact_path=deploy_artifact,
        )
    trial.crafting_progress = CraftingProgress.TRAIN_SUCCEEDED
    if any(a.kind in {"onnx", "engine", "torchscript"} for a in trial.artifact_contract.artifacts.values()):
        trial.crafting_progress = CraftingProgress.EXPORT_SUCCEEDED
    trial.stage_reached = StageReached.TRAIN
    logger.debug(f"[{variant.trial_id}] Stage 2 TRAIN done: metrics={metrics}")

    if deploy_artifact and _run_l0_compatibility_check(trial, state, deploy_artifact):
        _record_trial_observations(trial)
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    # ------------------------------------------------------------------
    # Surrogate evidence: useful as a prior, never authoritative edge evidence.
    # ------------------------------------------------------------------
    surrogate = SurrogatePredictor()
    trial_bank = state.get("trial_bank")
    past_trials = trial_bank.get_all() if trial_bank else []

    lat_est, mem_est, surrogate_confidence = surrogate.predict(
        variant, state.get("target_device", ""), past_trials
    )
    proxy_fp = build_artifact_fingerprint(deploy_artifact).model_dump(mode="json") if deploy_artifact else {}
    proxy_env = build_environment_fingerprint(
        device_id=str(state.get("target_device", "")),
        runtime=str(getattr(variant, "export_format", "") or artifact_kind(deploy_artifact)),
        runtime_config=state.get("runtime_config"),
        protocol={"precision": getattr(variant, "quant_mode", None)},
    )
    _append_evidence(
        trial,
        Evidence(
            probe_id="surrogate_prior",
            quantity="Latency",
            fidelity="proxy",
            outcome="unknown",
            value=lat_est,
            unit="ms",
            artifact_fingerprint=proxy_fp,
            environment_fingerprint=proxy_env,
            protocol={"confidence": surrogate_confidence, "decision_authority": "rank_only"},
        ),
        Evidence(
            probe_id="surrogate_prior",
            quantity="Memory_mb",
            fidelity="proxy",
            outcome="unknown",
            value=mem_est,
            unit="MB",
            artifact_fingerprint=proxy_fp,
            environment_fingerprint=proxy_env,
            protocol={"confidence": surrogate_confidence, "decision_authority": "rank_only"},
        ),
    )

    # ------------------------------------------------------------------
    # Stage 3: EdgeBenchmark (via EdgeRunner with workspace's infer.py)
    # ------------------------------------------------------------------
    device_id = state.get("target_device", "")
    device_ip = state.get("device_ip", "")
    ssh_key = state.get("ssh_key", "")

    if not device_ip:
        logger.warning(f"[{variant.trial_id}] No device_ip; measured edge evidence is unavailable.")
        trial.error = "Measured edge evidence unavailable: device_ip_missing"
        trial.error_stage = "edge_benchmark"
        trial.runtime_report.fallback_reason = "device_ip_missing"
        _record_runtime_attempt(
            trial,
            runtime="surrogate",
            artifact_path=deploy_artifact or None,
            status="skipped",
            fallback_reason="device_ip_missing",
        )
        _apply_verifier_policy(trial, user_spec, next_probe="full_edge")
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    # Import EdgeRunner lazily
    try:
        from edgecraft.tools.deploy.edge_runner import EdgeRunner
        runner = EdgeRunner()
    except ImportError as e:
        logger.warning(f"[{variant.trial_id}] EdgeRunner not available: {e}")
        trial.error = f"Measured edge evidence unavailable: {e}"
        trial.error_stage = "edge_benchmark"
        trial.runtime_report.fallback_reason = "edge_runner_unavailable"
        _record_runtime_attempt(
            trial,
            runtime="surrogate",
            artifact_path=deploy_artifact or None,
            status="skipped",
            fallback_reason="edge_runner_unavailable",
        )
        _apply_verifier_policy(trial, user_spec, next_probe="full_edge")
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    logger.debug(f"[{variant.trial_id}] Stage 3 EDGE_BENCHMARK: deploying to {device_ip}")

    # Get the model path for deployment
    model_path = _primary_deploy_artifact(trial)
    infer_script = ws / "infer.py"
    dataset_for_edge = (
        (state.get("dataset_info") or {}).get("dataset_path")
        or state.get("dataset_path")
        or ""
    )
    pending_edge_snapshot = _archive_direct_pending_edge_job(
        state=state,
        variant=variant,
        trial=trial,
        ws=ws,
        model_path=model_path,
    )
    max_debug_retries = max(0, int(getattr(settings, "DEBUGGER_MAX_RETRIES", 2)))
    result: Dict[str, Any] = {}
    full_edge_started = time.perf_counter()
    for retry_idx in range(max_debug_retries + 1):
        result = runner.run(
            artifact_path=model_path,
            device_id=device_id,
            device_ip=device_ip,
            ssh_key=ssh_key,
            workspace_path=str(ws),
            infer_script=str(infer_script) if infer_script.exists() else None,
            dataset_path=str(dataset_for_edge) if dataset_for_edge else None,
            docker_image=state.get("docker_image"),
            export_format=getattr(variant, "export_format", None) or "onnx",
            quant_mode=getattr(variant, "quant_mode", None) or "fp16",
            graph_hash=build_artifact_fingerprint(model_path).graph_hash if model_path else "",
            artifact_manifest=trial.artifact_contract.model_dump(mode="json"),
            require_exact_runtime=True,
        )
        if result.get("status") == "success":
            _mark_debug_retry_success(trial, "edge_benchmark")
            break
        if not infer_script.exists():
            break
        err_msg = result.get("error", "edge_benchmark failed")
        stderr = _result_output_text(result, "stderr")
        stdout = _result_output_text(result, "stdout")
        is_repairable, signature = _is_repairable_failure(err_msg, f"{stderr}\n{stdout}")
        if retry_idx >= max_debug_retries or not is_repairable:
            break
        diagnostics = _collect_failure_diagnostics(
            ws=ws,
            state=state,
            stage="edge_benchmark",
            result=result,
        )
        _record_trial_observations(trial, diagnostics=diagnostics)
        repair_diagnosis = _build_repair_diagnosis(
            stage="edge_benchmark",
            component="infer.py",
            result=result,
            trial=trial,
            sample_observation=diagnostics.get("sample_observation") if isinstance(diagnostics, dict) else None,
        )
        if repair_diagnosis.get("repair_action") != "patch_current_script":
            trial.failure_context["repair_diagnosis"] = repair_diagnosis
            trial.next_search_hint = repair_diagnosis.get("next_search_hint")
            break
        patched, summary, before_hash, after_hash, reason = _run_llm_debugger(
            script_path=infer_script,
            stage="edge_benchmark",
            variant=variant,
            error=err_msg,
            stdout=stdout,
            stderr=stderr,
            error_signature=signature,
            diagnostic_context=diagnostics,
            debug_history=trial.debug_history,
        )
        trial.debug_attempts += 1
        trial.debug_history.append(
            DebugAttempt(
                stage="edge_benchmark",
                attempt=trial.debug_attempts,
                error_signature=signature,
                reason=reason,
                before_hash=before_hash,
                after_hash=after_hash,
                patch_summary=summary,
                patch_applied=patched,
                success=False,
            )
        )
        if not patched:
            break
        logger.info(
            f"[{variant.trial_id}] Debugger patched infer.py "
            f"(attempt {retry_idx + 1}/{max_debug_retries}). Retrying edge benchmark."
        )

    if result.get("artifact_bundle_staging"):
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["artifact_bundle_staging"] = result["artifact_bundle_staging"]
        trial.observations = observations

    _record_stage_execution_evidence(
        trial,
        stage="full_edge_benchmark",
        result={
            **result,
            "elapsed_s": time.perf_counter() - full_edge_started,
            "timeout_s": int(getattr(settings, "EDGE_RUN_TIMEOUT_S", 2400)),
        },
        resource="device",
    )
    _complete_direct_pending_edge_job(pending_edge_snapshot, result)

    if result.get("status") != "success":
        trial.error = result.get("error", "edge_benchmark failed")
        trial.error_stage = "edge_benchmark"
        requested_runtime = getattr(variant, "export_format", None) or artifact_kind(model_path) or "unknown"
        runtime_fields = _resolve_edge_runtime_fields(result)
        attempted_runtime = runtime_fields["runtime_used"] or str(requested_runtime)
        attempted_artifact = runtime_fields["artifact_used"] or model_path or ""
        provider = runtime_fields["runtime_provider"]
        failure_observation = _runtime_failure_observation_context(
            result,
            requested_runtime=requested_runtime,
            requested_artifact=model_path or "",
            attempted_runtime=attempted_runtime,
            attempted_artifact=attempted_artifact,
        )
        if _runtime_identity(failure_observation["runtime"]) != _runtime_identity(
            attempted_runtime
        ):
            _record_runtime_attempt(
                trial,
                runtime=failure_observation["runtime"],
                artifact_path=failure_observation["artifact_path"] or None,
                status="failed",
                error=failure_observation["error_text"],
                fallback_reason=failure_observation["fallback_reason"],
            )
        _record_runtime_attempt(
            trial,
            runtime=str(attempted_runtime),
            artifact_path=attempted_artifact or None,
            provider=provider,
            status="failed",
            error=trial.error,
            fallback_reason=result.get("fallback_reason"),
        )
        trial.runtime_report.requested_runtime = str(requested_runtime)
        trial.runtime_report.runtime_used = attempted_runtime
        trial.runtime_report.artifact_used = attempted_artifact or None
        trial.runtime_report.runtime_provider = provider
        trial.runtime_report.fallback_reason = (
            failure_observation["fallback_reason"] or trial.error
        )
        _record_failure(
            trial,
            trial.error,
            _result_output_text(result, "stderr"),
            _result_output_text(result, "stdout"),
        )
        _observe_compatibility_failure(
            trial,
            state,
            artifact_path=failure_observation["artifact_path"],
            failure_stage=failure_observation["failure_stage"],
            runtime_used=failure_observation["runtime"],
            error_text=failure_observation["error_text"],
        )
        diagnostics = _collect_failure_diagnostics(
            ws=ws,
            state=state,
            stage="edge_benchmark",
            result=result,
        )
        repair_diagnosis = trial.failure_context.get("repair_diagnosis") or _build_repair_diagnosis(
            stage="edge_benchmark",
            component="infer.py",
            result=result,
            trial=trial,
            sample_observation=diagnostics.get("sample_observation") if isinstance(diagnostics, dict) else None,
        )
        trial.next_search_hint = repair_diagnosis.get("next_search_hint") or trial.next_search_hint
        trial.failure_context.update({
            "failed_stage": result.get("failed_stage"),
            "return_code": result.get("return_code"),
            "command": result.get("command"),
            "stderr_tail": result.get("stderr_tail") or (result.get("stderr", "") or "")[-1200:],
            "stdout_tail": result.get("stdout_tail") or (result.get("stdout", "") or "")[-1200:],
            "repair_diagnosis": repair_diagnosis,
            "diagnostics": diagnostics,
        })
        _record_trial_observations(trial, diagnostics=diagnostics)
        edge_stderr = _result_output_text(result, "stderr")
        edge_stdout = _result_output_text(result, "stdout")
        if edge_stderr:
            logger.warning(f"[{variant.trial_id}] edge_benchmark stderr:\n{edge_stderr[-2000:]}")
        if edge_stdout:
            logger.debug(f"[{variant.trial_id}] edge_benchmark stdout (tail):\n{edge_stdout[-1000:]}")
        trial.duration_seconds = time.time() - start_time
        _log_trial_summary(variant, trial)
        state["current_trial_result"] = trial
        state["status"] = "scoring"
        return state

    # Preserve task-quality/parity output from infer.py and standardize
    # efficiency facts with the evaluator-owned artifact replay.
    full_edge_wall_s = time.perf_counter() - full_edge_started
    result, efficiency_replay_wall_s = _run_full_efficiency_replay(
        trial=trial,
        state=state,
        variant=variant,
        ws=ws,
        runner=runner,
        model_path=model_path,
        edge_result=result,
    )

    # Parse edge metrics
    edge_metrics_raw: Dict[str, Any] = result.get("metrics", {})
    edge_metrics = MetricRegistry.normalize_metric_dict(
        edge_metrics_raw if isinstance(edge_metrics_raw, dict) else {}
    )
    if "Energy_mj" in edge_metrics and "Energy_j" not in edge_metrics:
        edge_metrics["Energy_j"] = float(edge_metrics["Energy_mj"]) / 1000.0
    elif "Energy_j" in edge_metrics and "Energy_mj" not in edge_metrics:
        edge_metrics["Energy_mj"] = float(edge_metrics["Energy_j"]) * 1000.0
    artifact_executed, execution_reasons = _artifact_execution_evidence(edge_metrics_raw)
    if not artifact_executed:
        edge_metrics = {
            name: value
            for name, value in edge_metrics.items()
            if name in {"artifact_loaded", "artifact_present", "load_error_present"}
        }
    latency = _safe_float(edge_metrics.get("Latency"))
    memory = _safe_float(edge_metrics.get("Memory_mb"))
    throughput = edge_metrics.get("Throughput")
    runtime_fields = _resolve_edge_runtime_fields(result)
    trial.runtime_report.runtime_used = runtime_fields["runtime_used"]
    trial.runtime_report.runtime_provider = runtime_fields["runtime_provider"]
    trial.runtime_report.artifact_used = runtime_fields["artifact_used"]
    trial.runtime_report.requested_runtime = getattr(variant, "export_format", None) or artifact_kind(model_path)
    fallback = _runtime_fallback_evidence(
        result,
        requested_runtime=trial.runtime_report.requested_runtime,
        runtime_used=trial.runtime_report.runtime_used,
    )
    if fallback and _runtime_identity(trial.runtime_report.runtime_used) in {
        "",
        "none",
        "null",
        "unknown",
    }:
        artifact_executed = False
        execution_reasons.append("no successful runtime after preferred-runtime failure")
        edge_metrics = {
            name: value
            for name, value in edge_metrics.items()
            if name in {"artifact_loaded", "artifact_present", "load_error_present"}
        }

    edge_metrics = _filter_untrusted_edge_quality(trial, edge_metrics)
    _record_evaluation_validity(
        trial,
        artifact_executed=artifact_executed,
        raw_metrics=edge_metrics_raw,
    )
    latency = _safe_float(edge_metrics.get("Latency"))
    memory = _safe_float(edge_metrics.get("Memory_mb"))
    throughput = edge_metrics.get("Throughput")

    trial.edge_metrics = EdgeMetrics(
        latency_ms=latency,
        latency_mean_ms=_safe_float(edge_metrics.get("Latency_mean")),
        memory_mb=memory,
        latency_p95_ms=_safe_float(edge_metrics.get("Latency_p95")),
        latency_p99_ms=_safe_float(edge_metrics.get("Latency_p99")),
        runtime_used=trial.runtime_report.runtime_used,
        runtime_provider=trial.runtime_report.runtime_provider,
        artifact_used=trial.runtime_report.artifact_used,
        throughput_fps=float(throughput) if throughput else None,
        all_metrics=edge_metrics,
    )
    if fallback:
        trial.runtime_report.fallback_reason = fallback["fallback_reason"]
        _record_runtime_attempt(
            trial,
            runtime=fallback["requested_runtime"],
            artifact_path=model_path or None,
            status="failed",
            error=fallback["error_text"] or fallback["fallback_reason"],
            fallback_reason=fallback["fallback_reason"],
        )
        observations = dict(getattr(trial, "observations", {}) or {})
        observations["runtime_fallback"] = {
            key: value for key, value in fallback.items() if key != "error_text"
        }
        trial.observations = observations
        _observe_compatibility_failure(
            trial,
            state,
            artifact_path=model_path or "",
            failure_stage="runtime_fallback",
            error_text=fallback["error_text"] or fallback["fallback_reason"],
        )
    observations = dict(getattr(trial, "observations", {}) or {})
    observations["edge_execution_contract"] = {
        "artifact_executed": artifact_executed,
        "reasons": execution_reasons,
        "raw_metrics": edge_metrics_raw if not artifact_executed else {},
    }
    trial.observations = observations
    post_metric_diag = _build_repair_diagnosis(
        stage="post_metric",
        component="metrics",
        result=result,
        trial=trial,
        sample_observation=(trial.failure_context.get("loader_smoke") or {}).get("sample_observation") if isinstance(trial.failure_context, dict) else None,
        local_metrics=trial.local_metrics.all_metrics if trial.local_metrics else {},
        edge_metrics=edge_metrics,
        artifact_manifest=trial.artifact_contract.model_dump(mode="json") if trial.artifact_contract else {},
        user_spec=user_spec,
    )
    if post_metric_diag.get("repair_action") == "next_proposal_only" or post_metric_diag.get("diagnosis_type") in {"metric_failure", "loader_semantics_error", "representation_mismatch"}:
        trial.failure_context["repair_diagnosis"] = post_metric_diag
        trial.next_search_hint = post_metric_diag.get("next_search_hint") or trial.next_search_hint
    if artifact_executed:
        _record_runtime_attempt(
            trial,
            runtime=str(trial.runtime_report.runtime_used or trial.runtime_report.requested_runtime or "unknown"),
            artifact_path=trial.runtime_report.artifact_used or model_path or None,
            provider=trial.runtime_report.runtime_provider,
            status="success",
        )
    else:
        trial.error = "Edge benchmark produced no valid artifact execution evidence."
        trial.error_stage = "edge_benchmark"
    _record_trial_observations(trial)
    full_fp = build_artifact_fingerprint(model_path).model_dump(mode="json") if model_path else {}
    requested_runtime = str(trial.runtime_report.requested_runtime or artifact_kind(model_path) or "")
    runtime_used = str(trial.runtime_report.runtime_used or "")
    source_hash = str(result.get("source_artifact_hash") or "")
    local_hash = str(full_fp.get("artifact_hash") or "")
    graph_hash = str(full_fp.get("graph_hash") or "")
    requested_graph_hash = str(result.get("requested_graph_hash") or "")
    runtime_exact = bool(
        result.get("require_exact_runtime") is True
        and not result.get("fallback_reason")
        and _runtime_identity(runtime_used) == _runtime_identity(requested_runtime)
    )
    fingerprint_verified = bool(
        graph_hash
        and source_hash
        and local_hash
        and source_hash == local_hash
        and requested_graph_hash == graph_hash
        and trial.runtime_report.artifact_used
    )
    strict_quality = bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False))
    validity = (getattr(trial, "observations", {}) or {}).get(
        "evaluation_validity"
    ) or {}
    reported_evaluation_valid = validity.get("evaluation_valid") is True
    evaluation_valid = bool(
        reported_evaluation_valid if strict_quality else artifact_executed
    )
    full_authorized = bool(
        result.get("status") == "success"
        and artifact_executed
        and runtime_exact
        and fingerprint_verified
        and evaluation_valid
    )
    artifact_properties = dict(full_fp.get("artifact_properties") or {})
    artifact_properties.update(
        {
            "requested_runtime": requested_runtime,
            "runtime_used": runtime_used,
            "runtime_exact": runtime_exact,
            "artifact_used": trial.runtime_report.artifact_used,
            "source_artifact_hash": source_hash,
            "fingerprint_source": "executed_staged_artifact",
        }
    )
    full_fp["artifact_properties"] = artifact_properties
    full_env = _runtime_environment_fingerprint(
        state,
        runtime=str(trial.runtime_report.runtime_used or trial.runtime_report.requested_runtime or ""),
        precision=getattr(variant, "quant_mode", None),
    )
    full_evidence: List[Evidence] = []
    authority_protocol = {
        "decision_authority": "p2_accept" if full_authorized else "audit_only",
        "artifact_executed": artifact_executed,
        "runtime_exact": runtime_exact,
        "fingerprint_verified": fingerprint_verified,
        "evaluation_valid": evaluation_valid,
        "strict_evaluation_required": strict_quality,
        "runtime": runtime_used,
        "requested_runtime": requested_runtime,
        "artifact_used": trial.runtime_report.artifact_used,
    }
    for quantity, value in (trial.local_metrics.all_metrics if trial.local_metrics else {}).items():
        full_evidence.append(
            Evidence(
                probe_id="full",
                quantity=quantity,
                fidelity="measured_congruent" if full_authorized else "proxy",
                outcome="pass" if full_authorized else "unknown",
                value=float(value),
                artifact_fingerprint=full_fp,
                environment_fingerprint=full_env,
                protocol={"training": "full", **authority_protocol},
            )
        )
    edge_units = {
        "Latency": "ms",
        "Memory_mb": "MB",
        "Energy_j": "J",
        "Energy_mj": "mJ",
        "Power_w": "W",
    }
    full_measurement_protocol = (
        result.get("measurement_protocol")
        if isinstance(result.get("measurement_protocol"), dict)
        else {}
    )
    for quantity, value in edge_metrics.items():
        full_evidence.append(
            Evidence(
                probe_id="full",
                quantity=quantity,
                fidelity="measured_congruent" if full_authorized else "proxy",
                outcome="pass" if full_authorized else "unknown",
                value=float(value),
                unit=edge_units.get(quantity, ""),
                artifact_fingerprint=full_fp,
                environment_fingerprint=full_env,
                protocol={
                    **authority_protocol,
                    "runtime": trial.runtime_report.runtime_used,
                    "runtime_provider": trial.runtime_report.runtime_provider,
                    "measurement_contract_version": EFFICIENCY_MEASUREMENT_CONTRACT_VERSION,
                    "measurement_sessions": full_measurement_protocol.get("measurement_sessions"),
                    "latency_statistic": full_measurement_protocol.get("latency_statistic"),
                    "latency_uncertainty": full_measurement_protocol.get("latency_uncertainty"),
                    "efficiency_source": full_measurement_protocol.get("efficiency_source", "generated_infer"),
                    "warmup": full_measurement_protocol.get(
                        "warmup",
                        edge_metrics_raw.get("warmup") if isinstance(edge_metrics_raw, dict) else None,
                    ),
                    "warmup_unit": full_measurement_protocol.get("warmup_unit"),
                    "min_warmup_seconds": full_measurement_protocol.get("min_warmup_seconds"),
                    "actual_warmup": full_measurement_protocol.get("actual_warmup"),
                    "actual_warmup_seconds": full_measurement_protocol.get("actual_warmup_seconds"),
                    "repetitions": (
                        full_measurement_protocol.get("actual_repetitions")
                        or full_measurement_protocol.get("repetitions")
                        or (
                            edge_metrics_raw.get("iterations") or edge_metrics_raw.get("repetitions")
                            if isinstance(edge_metrics_raw, dict)
                            else None
                        )
                    ),
                    "measurement_policy": full_measurement_protocol.get("measurement_policy"),
                    "min_measure_seconds": full_measurement_protocol.get("min_measure_seconds"),
                    "benchmark_scope": full_measurement_protocol.get("benchmark_scope"),
                    "energy_source": full_measurement_protocol.get("energy_source"),
                    "energy_scope": full_measurement_protocol.get("energy_scope"),
                    "energy_trusted": full_measurement_protocol.get("energy_trusted", False),
                    "trt_timing_cache_id": edge_metrics_raw.get("trt_timing_cache_id") if isinstance(edge_metrics_raw, dict) else None,
                    "trt_timing_cache_hit": edge_metrics_raw.get("trt_timing_cache_hit") if isinstance(edge_metrics_raw, dict) else None,
                },
                sigma=(
                    _safe_float(edge_metrics.get("latency_p95_session_std_ms"))
                    if MetricRegistry.canonicalize_name(quantity) == "Latency"
                    else None
                ),
            )
        )
    report = _append_evidence(trial, *full_evidence)
    report.level = "L2"
    report.status = "pass" if full_authorized else "fail"
    report.evidence_source = "measured"
    report.artifact_fingerprint = full_fp
    report.artifact_status = "present" if full_fp else "missing"
    report.runtime_status = str(trial.runtime_report.runtime_used or "")
    report.source = "full_pipeline"
    if not full_authorized:
        report.errors.append(
            "P2 evidence is non-authoritative: exact runtime or executed-artifact fingerprint was not verified."
        )
    _append_online_calibration_pairs(trial)
    _apply_verifier_policy(trial, user_spec, next_probe="")
    trial.crafting_progress = (
        CraftingProgress.EDGE_RUNNABLE
        if artifact_executed
        else CraftingProgress.EXPORT_SUCCEEDED
    )
    runtime_name = (trial.runtime_report.runtime_used or "").lower()
    provider_name = (trial.runtime_report.runtime_provider or "").lower()
    if artifact_executed and (
        runtime_name in {"engine", "tensorrt"}
        or "tensorrt" in provider_name
        or "cuda" in provider_name
    ):
        trial.crafting_progress = CraftingProgress.EDGE_OPTIMIZED
    trial.stage_reached = StageReached.EDGE_BENCHMARK
    trial.duration_seconds = time.time() - start_time

    _log_trial_summary(variant, trial)
    state["current_trial_result"] = trial
    state["status"] = "scoring"
    return state
