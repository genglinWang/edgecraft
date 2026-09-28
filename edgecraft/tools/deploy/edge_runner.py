"""Edge Runner integration for EdgeCraft.

In the code-centric architecture, EdgeRunner deploys the workspace's infer.py
script to edge devices rather than using hardcoded benchmark scripts.

The infer.py script is LLM-generated and self-contained, handling:
- Model loading and export (e.g., TensorRT conversion)
- Inference benchmarking
- Metrics output (JSON on stdout)
"""

import os
import subprocess
import json
import tempfile
import shutil
import time
import re
import hashlib
import uuid
from typing import Dict, Any, Optional, List, Iterable
from pathlib import Path

from loguru import logger

from edgecraft.tools.deploy.base import BaseDeployer
from edgecraft.tools.base import ToolRegistry
from edgecraft.config.settings import edgecraft_env, settings
from edgecraft.config.device_manifest import load_declared_devices, supported_formats
from edgecraft.tools.deploy.docker_opts import edge_runner_infer_docker_opts_bash
from edgecraft.tools.deploy.energy_measurement import (
    edge_power_sampler_bash,
    merge_trusted_energy,
)
from edgecraft.utils.trash import move_to_trash


# Inline device catalog
_DEVICE_CATALOG: Dict[str, Dict[str, Any]] = {
    "jetson_orin_agx": {
        "has_gpu": True,
        "supported_formats": ["pt", "onnx", "engine"],
        "energy_sampler": "jetson_ina",
    },
    "jetson_xavier_nx": {
        "has_gpu": True,
        "supported_formats": ["pt", "onnx", "engine"],
        "energy_sampler": "jetson_ina",
    },
    "jetson_xavier": {
        "has_gpu": True,
        "supported_formats": ["pt", "onnx", "engine"],
        "energy_sampler": "jetson_ina",
    },
    "jetson_tx2": {
        "has_gpu": True,
        "supported_formats": ["pt", "onnx", "engine"],
        "energy_sampler": "jetson_ina",
        "remote_base": "~/.edgecraft-edge-runner-inline",
        "inline_runner": True,
    },
    "raspberry_pi_5": {
        "has_gpu": False,
        "supported_formats": ["onnx", "tflite"],
        "energy_sampler": "rpi_pmic",
        "execution_mode": "native",
        "remote_base": "~/.edgecraft-edge-runner-inline",
        "inline_runner": True,
    },
    "x86_cpu_desktop": {
        "has_gpu": False,
        "supported_formats": ["pt", "onnx", "tflite"],
        "remote_base": "~/.edgecraft-edge-runner-inline",
        "inline_runner": True,
    },
}

_FALLBACK_DEVICE: Dict[str, Any] = {
    "has_gpu": False,
    "supported_formats": ["onnx"],
}

_STAGING_IGNORE_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
    "__MACOSX",
}

_STAGING_IGNORE_SUFFIXES = {
    ".cache",
    ".tmp",
    ".log",
}

_EDGE_OUTPUT_CHECKPOINT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
}


def _runtime_tmp_root() -> Path:
    root = Path(getattr(settings, "RUNTIME_TMP_DIR", "") or os.environ.get("TMPDIR") or "tmp")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_trtexec_metrics(stdout: str) -> Dict[str, Any]:
    """Extract lightweight edge evidence from TensorRT trtexec logs."""
    text = stdout or ""
    metrics: Dict[str, Any] = {}
    m = re.search(
        r"GPU Compute Time:\s*min\s*=\s*([0-9.]+)\s*ms.*?mean\s*=\s*([0-9.]+)\s*ms",
        text,
        re.S,
    )
    if m:
        metrics["latency_ms"] = float(m.group(2))
        metrics["trtexec_gpu_compute_mean_ms"] = float(m.group(2))
        metrics["trtexec_gpu_compute_min_ms"] = float(m.group(1))
    m = re.search(r"Throughput:\s*([0-9.]+)\s*qps", text)
    if m:
        metrics["throughput_qps"] = float(m.group(1))
    if "PASSED TensorRT.trtexec" in text:
        metrics["trtexec_build_success"] = 1.0
    cache = re.search(r"TRT_COMPILE_CACHE_HIT=(0|1)", text)
    if cache:
        metrics["trt_compile_cache_hit"] = float(cache.group(1))
    build_time = re.search(r"TRT_COMPILE_SECONDS=([0-9.]+)", text)
    if build_time:
        metrics["trt_compile_seconds"] = float(build_time.group(1))
    timing_hit = re.search(r"TRT_TIMING_CACHE_HIT=(0|1)", text)
    if timing_hit:
        metrics["trt_timing_cache_hit"] = float(timing_hit.group(1))
    timing_id = re.search(r"TRT_TIMING_CACHE_ID=([a-fA-F0-9]+)", text)
    if timing_id:
        metrics["trt_timing_cache_id"] = timing_id.group(1)
    return metrics


_VALIDATION_RESULT_FIELDS = (
    "artifact_loadable",
    "runtime_executed",
    "evaluation_bundle_valid",
    "prediction_parity",
    "prediction_mismatch_count",
    "evaluation_valid",
    "sample_count",
    "prediction_sha256",
    "local_prediction_sha256",
)


def _parse_stdout_metric_payload(stdout: str) -> Dict[str, Any]:
    """Return the last JSON result payload emitted by infer.py."""
    for line in reversed((stdout or "").strip().splitlines()):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            if "latency_avg_ms" in data and "latency_ms" not in metrics:
                metrics["latency_ms"] = data["latency_avg_ms"]
            if "peak_memory_mb" in data and "memory_mb" not in metrics:
                metrics["memory_mb"] = data["peak_memory_mb"]
        metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if not metrics and data.get("status") != "error":
            continue
        result = {
            "status": data.get("status", "success"),
            "metrics": dict(metrics),
            **{
                key: data[key]
                for key in (
                    "error",
                    "failed_stage",
                    "runtime_used",
                    "runtime_provider",
                    "artifact_used",
                    "fallback_reason",
                    "primary_error_tail",
                    *_VALIDATION_RESULT_FIELDS,
                )
                if data.get(key) is not None
            },
        }
        if isinstance(data.get("measurement_protocol"), dict):
            result["measurement_protocol"] = dict(data["measurement_protocol"])
        return merge_trusted_energy(result, stdout)
    return {}


def _sha256_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_cleanup_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        _chmod_tree_user_readable(path)
        move_to_trash(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"EdgeRunner could not cleanup local job dir {path}: {exc}")


def _snapshot_staged_job(job_dir: Path, *, max_entries: int = 80) -> Dict[str, Any]:
    """Return a compact, generic snapshot of a staged edge job bundle."""
    snapshot: Dict[str, Any] = {
        "job_dir": str(job_dir),
        "exists": job_dir.exists(),
        "run_sh_exists": (job_dir / "run.sh").is_file(),
        "meta_env_exists": (job_dir / "meta.env").is_file(),
        "payload_exists": (job_dir / "payload").is_dir(),
        "entries": [],
    }
    if not job_dir.exists():
        return snapshot
    entries: list[Dict[str, Any]] = []
    for path in sorted(job_dir.rglob("*"))[:max_entries]:
        try:
            rel = str(path.relative_to(job_dir))
            stat = path.stat()
            entries.append(
                {
                    "path": rel,
                    "type": "dir" if path.is_dir() else "file",
                    "mode": oct(stat.st_mode & 0o777),
                    "size": stat.st_size if path.is_file() else None,
                }
            )
        except OSError as exc:
            entries.append({"path": str(path), "error": str(exc)})
    snapshot["entries"] = entries
    snapshot["truncated"] = len(entries) >= max_entries
    return snapshot


def _infer_failed_stage(blob: str) -> str:
    text = (blob or "").lower()
    stages = re.findall(r"failed_stage=([a-z_]+)", text)
    if stages:
        return stages[-1]
    if (
        "modulenotfounderror" in text
        or ("traceback" in text and "infer.py" in text)
    ):
        return "inference"
    if "collect_results" in text or "failed to collect" in text:
        return "collect_results"
    if "connection refused" in text or "ssh" in text or "remote_execute" in text:
        return "remote_execute"
    if "tar" in text or "extract" in text or "parse" in text:
        return "parse_results"
    return "bootstrap"


def _copytree_for_edge(src: Path, dst: Path) -> None:
    """Copy a dataset/workspace tree with generic edge-staging exclusions."""

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in _STAGING_IGNORE_NAMES:
                ignored.add(name)
                continue
            if any(name.endswith(suffix) for suffix in _STAGING_IGNORE_SUFFIXES):
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_ignore)


def _copy_dataset_sample_for_edge(
    src: Path,
    dst: Path,
    *,
    max_files: int,
    max_bytes: int,
    preferred_relpaths: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Stage a bounded dataset sample for edge-side infer.py smoke runs.

    Edge benchmarking should measure the exported artifact/runtime, not move an
    entire training corpus through the Jetson /tmp partition.  This sampler
    preserves relative paths while capping payload size; infer.py scripts that
    need one or a few examples can still run, and scripts that use dummy inputs
    avoid unnecessary disk pressure.
    """
    dst.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    copied_bytes = 0
    skipped_files = 0

    def _copy_one(file_src: Path, rel: Path) -> bool:
        nonlocal copied_files, copied_bytes, skipped_files
        if copied_files >= max_files:
            skipped_files += 1
            return False
        try:
            size = file_src.stat().st_size
        except OSError:
            skipped_files += 1
            return False
        if size > max_bytes:
            skipped_files += 1
            logger.debug(f"Skipping oversized edge dataset sample file ({size} bytes): {file_src}")
            return False
        if copied_bytes + size > max_bytes:
            skipped_files += 1
            return False
        file_dst = dst / rel
        file_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_src, file_dst)
        copied_files += 1
        copied_bytes += size
        return True

    preferred = list(preferred_relpaths or ())
    if len(preferred) > max_files:
        if max_files == 1:
            spread_indices = [len(preferred) // 2]
        else:
            spread_indices = [
                index * (len(preferred) - 1) // (max_files - 1)
                for index in range(max_files)
            ]
        spread_set = set(spread_indices)
        preferred = [preferred[index] for index in spread_indices] + [
            value for index, value in enumerate(preferred) if index not in spread_set
        ]

    copied_relpaths: set[Path] = set()
    for raw in preferred:
        rel = Path(str(raw))
        if rel.is_absolute() or ".." in rel.parts:
            continue
        file_src = src / rel
        if file_src.is_file() and _copy_one(file_src, rel):
            copied_relpaths.add(rel)

    if src.is_file():
        _copy_one(src, Path(src.name))
    else:
        for file_src in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = file_src.relative_to(src)
            if rel in copied_relpaths:
                continue
            parts = set(rel.parts)
            if parts.intersection(_STAGING_IGNORE_NAMES):
                continue
            if any(file_src.name.endswith(suffix) for suffix in _STAGING_IGNORE_SUFFIXES):
                continue
            _copy_one(file_src, rel)
            if copied_files >= max_files or copied_bytes >= max_bytes:
                break

    return {
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "skipped_files": skipped_files,
        "max_files": max_files,
        "max_bytes": max_bytes,
    }


def _declared_edge_sample_paths(config_path: Path) -> list[str]:
    """Return file-like test IDs declared by a staged split manifest."""
    if not config_path.is_file():
        return []
    try:
        import yaml as _yaml

        config = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        manifest = config.get("split_manifest") or {}
        manifest_path = Path(str(manifest.get("path") or "")).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = (config_path.parent / manifest_path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(value) for value in (payload.get("splits") or {}).get("test", [])]
    except (OSError, ValueError, TypeError, AttributeError, ImportError):
        return []


def _chmod_tree_user_readable(root: Path) -> None:
    """Make staged payload files readable/writable by the edge user."""
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        try:
            if path.is_dir():
                path.chmod(0o755)
            else:
                path.chmod(0o644)
        except OSError:
            logger.debug(f"Could not chmod staged path: {path}")


def _copy_artifact_group(src: Path, outputs_dir: Path) -> None:
    """Copy an artifact and common sidecar files such as ONNX external data."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        _copytree_for_edge(src, outputs_dir)
        return
    shutil.copy2(src, outputs_dir / src.name)
    stem = src.name
    sidecar_candidates = [
        src.with_name(f"{stem}.data"),
        src.with_suffix(src.suffix + ".data"),
        src.with_suffix(".data"),
    ]
    for sidecar in sidecar_candidates:
        if sidecar.exists() and sidecar.is_file():
            shutil.copy2(sidecar, outputs_dir / sidecar.name)


def _copy_runtime_python_modules(workspace: Path, payload_dir: Path) -> List[str]:
    """Stage the generated infer module closure without guessing import names."""
    copied: List[str] = []
    if not workspace.is_dir():
        return copied
    for source in sorted(workspace.glob("*.py")):
        if not source.is_file():
            continue
        shutil.copy2(source, payload_dir / source.name)
        copied.append(source.name)
    return copied


def _copy_workspace_outputs_for_edge(ws_outputs: Path, dst_outputs: Path) -> Dict[str, Any]:
    """Merge lightweight runtime sidecars without staging bulky checkpoints.

    The primary artifact is copied first through ``_copy_artifact_group``.  Many
    training scripts then leave full HF/PyTorch checkpoints in ``outputs/``;
    copying them all to the edge makes result collection fragile and measures
    filesystem pressure rather than runtime behavior.  Keep small metadata and
    tokenizer files, but skip large checkpoint formats unless they were already
    selected as the primary artifact.
    """
    dst_outputs.mkdir(parents=True, exist_ok=True)
    max_extra_mb = int(edgecraft_env("EDGE_OUTPUT_EXTRA_MAX_MB", "16"))
    max_extra_bytes = max(1, max_extra_mb) * 1024 * 1024
    copied = 0
    skipped = 0

    for f in sorted(ws_outputs.iterdir()):
        target = dst_outputs / f.name
        if f.is_dir():
            files = [path for path in f.rglob("*") if path.is_file()]
            if any(
                path.suffix.lower() in _EDGE_OUTPUT_CHECKPOINT_SUFFIXES
                and path.name != "best.pt"
                for path in files
            ):
                skipped += 1
                logger.debug(f"Skipping edge payload directory with checkpoint sidecars: {f}")
                continue
            try:
                size = sum(path.stat().st_size for path in files)
            except OSError:
                skipped += 1
                continue
            if size > max_extra_bytes:
                skipped += 1
                logger.debug(f"Skipping oversized edge payload directory ({size} bytes): {f}")
                continue
            _copytree_for_edge(f, target)
            copied += len(files)
            continue
        if target.exists():
            continue
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        try:
            size = f.stat().st_size
        except OSError:
            skipped += 1
            continue
        if suffix in _EDGE_OUTPUT_CHECKPOINT_SUFFIXES and f.name != "best.pt":
            skipped += 1
            logger.debug(f"Skipping bulky edge payload checkpoint sidecar: {f}")
            continue
        if size > max_extra_bytes:
            skipped += 1
            logger.debug(f"Skipping oversized edge payload sidecar ({size} bytes): {f}")
            continue
        shutil.copy2(f, target)
        copied += 1
    return {"copied": copied, "skipped": skipped}


def _copy_required_artifacts_for_edge(
    artifact_manifest: Any,
    workspace: Path,
    dst_outputs: Path,
) -> Dict[str, Any]:
    """Stage files that generated training code explicitly requires at runtime."""
    artifacts = artifact_manifest.get("artifacts", {}) if isinstance(artifact_manifest, dict) else {}
    required = [record for record in artifacts.values() if isinstance(record, dict) and record.get("required_on_edge")]
    if not required:
        return {"status": "success", "copied": [], "bytes": 0}

    source_root = (workspace / "outputs").resolve()
    max_bytes = max(1, int(edgecraft_env("EDGE_ARTIFACT_BUNDLE_MAX_MB", "256"))) * 1024 * 1024
    staged: list[tuple[Path, Path, int]] = []
    total_bytes = 0
    for record in required:
        raw_path = str(record.get("path") or "")
        source = Path(raw_path)
        source = source if source.is_absolute() else workspace / source
        try:
            source = source.resolve()
            relative = source.relative_to(source_root)
        except (OSError, ValueError):
            return {"status": "error", "error": f"required artifact is outside workspace outputs: {raw_path}"}
        if not (source.is_file() or source.is_dir()):
            return {"status": "error", "error": f"required artifact is missing: {raw_path}"}
        target = dst_outputs / relative
        if target.exists():
            staged.append((source, relative, 0))
            continue
        size = (
            source.stat().st_size
            if source.is_file()
            else sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
        )
        total_bytes += size
        if total_bytes > max_bytes:
            return {
                "status": "error",
                "error": f"required artifact bundle exceeds {max_bytes} bytes",
                "bytes": total_bytes,
            }
        expected_hash = str(record.get("sha256") or "")
        if source.is_file() and expected_hash and _sha256_file(str(source)) != expected_hash:
            return {"status": "error", "error": f"required artifact hash mismatch: {raw_path}"}
        staged.append((source, relative, size))

    dst_outputs.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source, relative, _ in staged:
        target = dst_outputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if source.is_dir():
                _copytree_for_edge(source, target)
            else:
                shutil.copy2(source, target)
        copied.append(relative.as_posix())
    return {"status": "success", "copied": copied, "bytes": total_bytes}


def _edge_eval_manifest_rel_path(artifact_manifest: Any, workspace: Path) -> str:
    """Return the staged manifest path declared by the artifact contract."""
    artifacts = artifact_manifest.get("artifacts", {}) if isinstance(artifact_manifest, dict) else {}
    outputs = (workspace / "outputs").resolve()
    for record in artifacts.values():
        if not isinstance(record, dict) or not record.get("required_on_edge"):
            continue
        raw_path = str(record.get("path") or "")
        provenance = str(record.get("provenance") or "")
        if not (
            provenance.endswith(":edge_eval_manifest")
            or Path(raw_path).name == "edge_eval_manifest.json"
        ):
            continue
        source = Path(raw_path)
        source = source if source.is_absolute() else workspace / source
        try:
            relative = source.resolve().relative_to(outputs)
        except (OSError, ValueError):
            continue
        staged = (Path("outputs") / relative).as_posix()
        if re.fullmatch(r"outputs/[A-Za-z0-9._/-]+", staged):
            return staged
    return "outputs/edge_eval_manifest.json"


def _select_dataset_stage_source(dataset_path: Path, payload_dir: Path) -> Path:
    """Select the smallest generic dataset root that satisfies config/data.yaml.

    This handles category-structured anomaly datasets without naming MVTec:
    if config/data.yaml declares a category and that child contains train/good,
    stage only that category directory.
    """
    cfg_path = payload_dir / "config" / "data.yaml"
    if not dataset_path.is_dir() or not cfg_path.exists():
        return dataset_path
    try:
        import yaml as _yaml

        cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        category = str(cfg.get("category") or "").strip()
        if category:
            child = dataset_path / category
            if (child / "train" / "good").is_dir():
                return child
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Dataset stage source selection skipped: {exc}")
    return dataset_path


class EdgeRunner(BaseDeployer):
    """Deploy and benchmark models on edge devices using workspace's infer.py."""

    name = "edge_runner"
    description = "Deploy models to edge devices and run benchmarks"
    modality = None

    def __init__(
        self,
        edge_runner_path: str = None,
        ssh_key_path: str = None
    ):
        self.edge_runner_path = edge_runner_path or self._find_edge_runner()
        self.ssh_key_path = ssh_key_path

    def _find_edge_runner(self) -> Optional[str]:
        """Find edge-runner directory."""
        configured = edgecraft_env("EDGE_RUNNER_PATH").strip()
        if configured and Path(configured).is_dir():
            return str(Path(configured).resolve())
        edge_runner = Path(__file__).with_name("runner") / "server"

        if edge_runner.exists():
            return str(edge_runner)

        return None

    def _get_device_config(self, device_id: str) -> Dict[str, Any]:
        """Get configuration for a specific device."""
        try:
            declared = load_declared_devices().get(device_id)
        except (OSError, ValueError) as exc:
            logger.debug(f"Declared device resource lookup failed: {exc}")
            declared = None
        if declared:
            config = dict(_DEVICE_CATALOG.get(device_id) or _FALLBACK_DEVICE)
            config.update({
                key: declared[key]
                for key in (
                    "execution_mode",
                    "docker_image",
                    "native_envs",
                    "native_python",
                    "remote_base",
                    "inline_runner",
                    "runner_min_free_mb",
                    "energy_sampler",
                )
                if key in declared
            })
            config["supported_formats"] = supported_formats(declared)
            return config
        if device_id not in _DEVICE_CATALOG:
            try:
                from edgecraft.models import ensure_registries_initialized
                from edgecraft.models.family_registry import DeviceRegistry
                from edgecraft.models.specs import RuntimeId

                ensure_registries_initialized()
                spec = DeviceRegistry.get(device_id)
                if spec is not None:
                    supported = ["onnx"]
                    if RuntimeId.PYTORCH in spec.supported_runtimes:
                        supported.append("pt")
                    if RuntimeId.TENSORRT in spec.supported_runtimes:
                        supported.append("engine")
                    if RuntimeId.TFLITE in spec.supported_runtimes:
                        supported.append("tflite")
                    return {
                        "has_gpu": bool(spec.has_gpu),
                        "supported_formats": list(dict.fromkeys(supported)),
                    }
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"DeviceRegistry lookup failed for {device_id}: {exc}")
            logger.warning(f"Unknown device '{device_id}'. Using fallback defaults.")
            return _FALLBACK_DEVICE
        return _DEVICE_CATALOG[device_id]

    def device_capabilities(self, device_id: str) -> Dict[str, Any]:
        """Return a copy of the measured execution capabilities for one device."""
        return dict(self._get_device_config(device_id))

    def _execution_mode(self, device_id: str) -> str:
        """Return docker or native from device evidence."""
        return str(self._get_device_config(device_id).get("execution_mode") or "docker").lower()

    def _get_docker_image(self, device_id: str, docker_image: Optional[str] = None) -> str:
        """Resolve Docker image from the caller (CLI / agent state)."""
        if self._execution_mode(device_id) == "native":
            return (docker_image or "").strip()
        image = (docker_image or "").strip()
        if not image or image == "unset":
            raise ValueError(
                "Docker image is required for containerized edge hosts. "
                "Native-runtime devices must declare execution_mode=native in the device mapping."
            )
        return image

    def _get_native_python(
        self,
        device_id: str,
        native_python: Optional[str] = None,
        runtime: Optional[str] = None,
    ) -> str:
        """Resolve Python executable for native-runtime edge hosts."""
        cfg = self._get_device_config(device_id)
        envs = cfg.get("native_envs") or {}
        runtime_key = str(runtime or "").lower().strip()
        runtime_key = {
            "onnx": "onnxruntime",
            "ort": "onnxruntime",
            "pt": "pytorch",
            "torchscript": "pytorch",
            "tflite": "litert",
            "ai_edge_litert": "litert",
        }.get(runtime_key, runtime_key)
        return (
            str(native_python or "").strip()
            or edgecraft_env("EDGE_NATIVE_PYTHON").strip()
            or str(envs.get(runtime_key) or "").strip()
            or str(envs.get("onnxruntime") or envs.get("pytorch") or envs.get("litert") or "").strip()
            or str(cfg.get("native_python") or "").strip()
            or "python3"
        )

    def _get_remote_base(
        self,
        device_id: str,
        device_ip: Optional[str] = None,
        remote_base: Optional[str] = None,
    ) -> str:
        """Resolve the edge-runner base directory on the target host."""
        base = (
            remote_base
            or edgecraft_env("EDGE_REMOTE_BASE")
            or self._get_device_config(device_id).get("remote_base")
            or ""
        )
        base = str(base).strip()
        if base.startswith("~/") and device_ip and "@" in device_ip:
            user = device_ip.split("@", 1)[0].strip()
            if user:
                return f"/home/{user}/{base[2:]}"
        return base

    def _run_edge_script(
        self,
        script_name: str,
        args: list,
        timeout: int = 600,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an edge-runner script."""
        if not self.edge_runner_path:
            return {"status": "error", "error": "edge-runner path not found"}

        script_path = os.path.join(self.edge_runner_path, script_name)

        if not os.path.exists(script_path):
            return {"status": "error", "error": f"Script not found: {script_path}"}

        try:
            result = subprocess.run(
                ["bash", script_path] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or settings.EDGECRAFT_ROOT,
            )

            status = "success" if result.returncode == 0 else "error"
            d: Dict[str, Any] = {
                "status": status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
            if status == "error":
                lines = (result.stderr or result.stdout or "").strip().splitlines()
                tail = lines[-1] if lines else ""
                blob = f"{result.stderr}\n{result.stdout}"
                d["error"] = f"exit code {result.returncode}: {tail}"[:500]
                d["failed_stage"] = _infer_failed_stage(blob)
                d["stderr_tail"] = (result.stderr or "")[-1200:]
                d["stdout_tail"] = (result.stdout or "")[-1200:]
                d["command"] = f"bash {script_name} {' '.join(args)}"
            return d

        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Script timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run(
        self,
        artifact_path: str,
        device_id: str,
        device_ip: str,
        ssh_key: str = None,
        workspace_path: str = None,
        infer_script: str = None,
        dataset_path: str = None,
        timeout: Optional[int] = None,
        docker_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy and run inference benchmark on edge device.

        In code-centric architecture, this method uses workspace's infer.py
        instead of hardcoded benchmark scripts.

        Args:
            artifact_path: Path to model file.
            device_id: Device identifier.
            device_ip: SSH target (user@host).
            ssh_key: Path to SSH private key.
            workspace_path: Path to trial workspace containing infer.py.
            infer_script: Direct path to infer.py (alternative to workspace_path).
            timeout: Job timeout in seconds.
            docker_image: Pre-built image on the edge (from ``edgecraft synth`` / ``deploy``).

        Returns:
            Dict with benchmark results and metrics.
        """
        if not device_ip:
            return {"status": "error", "error": "device_ip is required"}
        from edgecraft.utils.network import validate_ssh_target

        try:
            device_ip = validate_ssh_target(device_ip)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

        ssh_key = ssh_key or self.ssh_key_path
        if not ssh_key:
            return {"status": "error", "error": "SSH key path is required"}

        # Engine build + fused edge eval can legitimately exceed 10 minutes.
        # Keep caller override, otherwise use a conservative default.
        effective_timeout = int(timeout or getattr(settings, "EDGE_RUN_TIMEOUT_S", 2400))
        if effective_timeout <= 0:
            effective_timeout = int(getattr(settings, "EDGE_RUN_TIMEOUT_S", 2400))

        # Get device configuration
        device_config = self._get_device_config(device_id)
        execution_mode = self._execution_mode(device_id)
        docker_image = self._get_docker_image(device_id, docker_image=docker_image)
        native_python = self._get_native_python(device_id, kwargs.get("native_python"))
        has_gpu = device_config.get("has_gpu", False)

        # Determine infer.py path
        infer_py_path = None
        if infer_script and os.path.exists(infer_script):
            infer_py_path = infer_script
        elif workspace_path:
            ws_infer = Path(workspace_path) / "infer.py"
            if ws_infer.exists():
                infer_py_path = str(ws_infer)

        # Generate job ID
        safe_device_id = device_id.replace(" ", "_").replace("/", "-")
        job_id = f"edgecraft_{safe_device_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Create job bundle
        job_dir = tempfile.mkdtemp(
            prefix=f"edgecraft_job_{job_id}_",
            dir=str(_runtime_tmp_root()),
        )

        try:
            # Create payload directory
            payload_dir = os.path.join(job_dir, "payload")
            os.makedirs(payload_dir, exist_ok=True)
            staged_artifact_path = ""
            artifact_bundle_staging: Dict[str, Any] = {
                "status": "success",
                "copied": [],
                "bytes": 0,
            }

            # Copy model file(s)
            if artifact_path and os.path.exists(artifact_path):
                outputs_dir = os.path.join(payload_dir, "outputs")
                if os.path.isdir(artifact_path):
                    _copytree_for_edge(Path(artifact_path), Path(outputs_dir))
                else:
                    # Copy to outputs/ to match infer.py expectations
                    _copy_artifact_group(Path(artifact_path), Path(outputs_dir))
                    staged_artifact_path = f"outputs/{Path(artifact_path).name}"

            if workspace_path:
                artifact_bundle_staging = _copy_required_artifacts_for_edge(
                    kwargs.get("artifact_manifest"),
                    Path(workspace_path),
                    Path(payload_dir) / "outputs",
                )
                if artifact_bundle_staging.get("status") != "success":
                    return {
                        "status": "error",
                        "error": artifact_bundle_staging.get("error", "required artifact staging failed"),
                        "failed_stage": "artifact_bundle",
                        "artifact_bundle_staging": artifact_bundle_staging,
                    }

            # Copy workspace outputs directory if available
            if workspace_path:
                ws_outputs = Path(workspace_path) / "outputs"
                if ws_outputs.exists():
                    dst_outputs = Path(payload_dir) / "outputs"
                    stats = _copy_workspace_outputs_for_edge(ws_outputs, dst_outputs)
                    logger.debug(
                        "Edge workspace outputs merged: "
                        f"copied={stats['copied']} skipped={stats['skipped']} dst={dst_outputs}"
                    )

            # Copy workspace config if available (infer.py may read config/data.yaml).
            if workspace_path:
                ws_cfg = Path(workspace_path) / "config"
                if ws_cfg.exists() and ws_cfg.is_dir():
                    _copytree_for_edge(ws_cfg, Path(payload_dir) / "config")

            # Stage dataset for edge-side eval (when available).
            dataset_bundle = self._stage_edge_dataset_bundle(
                payload_dir=Path(payload_dir),
                dataset_path=dataset_path,
            )

            # P0: infer.py is the contract for synth edge execution.
            # Keep edge-runner runtime framework-agnostic (cv/audio/nlp) and
            # avoid family-specific fallback logic.
            if not infer_py_path:
                return {
                    "status": "error",
                    "error": "infer.py not found; synth edge execution requires infer.py in workspace",
                }
            if workspace_path:
                copied_modules = _copy_runtime_python_modules(Path(workspace_path), Path(payload_dir))
                logger.debug(f"Edge runtime Python modules staged: {copied_modules}")
            shutil.copy2(infer_py_path, os.path.join(payload_dir, "infer.py"))
            _chmod_tree_user_readable(Path(payload_dir))
            deploy_fmt = str(kwargs.get("export_format") or "onnx").lower().strip(".")
            artifact_suffix = Path(artifact_path).suffix.lower().strip(".") if artifact_path else ""
            if artifact_suffix in {"pt", "pth", "ckpt", "safetensors"}:
                deploy_fmt = "pt"
            elif artifact_suffix == "onnx" and deploy_fmt not in {"engine", "tensorrt"}:
                deploy_fmt = "onnx"
            elif artifact_suffix == "tflite":
                deploy_fmt = "tflite"
            quant_mode = str(kwargs.get("quant_mode") or "fp16").lower()
            graph_hash = str(kwargs.get("graph_hash") or "")
            source_artifact_hash = _sha256_file(artifact_path) if artifact_path and os.path.isfile(artifact_path) else ""
            execution_provenance = {
                "requested_graph_hash": graph_hash,
                "source_artifact_hash": source_artifact_hash,
                "require_exact_runtime": bool(kwargs.get("require_exact_runtime", False)),
                "requested_runtime": deploy_fmt,
            }
            input_profile = str(kwargs.get("input_profile") or "default")
            edge_eval_manifest_rel = _edge_eval_manifest_rel_path(
                kwargs.get("artifact_manifest"),
                Path(workspace_path) if workspace_path else Path(payload_dir),
            )
            if execution_mode == "native" and not str(kwargs.get("native_python") or "").strip():
                native_python = self._get_native_python(device_id, runtime=deploy_fmt)
            run_script = self._generate_infer_run_script(
                docker_image,
                has_gpu,
                dataset_dir_rel=dataset_bundle.get("dataset_dir_rel"),
                config_path_rel=dataset_bundle.get("config_path_rel"),
                export_format=deploy_fmt,
                quant_mode=quant_mode,
                execution_mode=execution_mode,
                native_python=native_python,
                graph_hash=graph_hash,
                source_artifact_hash=source_artifact_hash,
                input_profile=input_profile,
                artifact_path_rel=staged_artifact_path,
                edge_eval_manifest_rel=edge_eval_manifest_rel,
                energy_sampler=str(device_config.get("energy_sampler") or ""),
                require_exact_runtime=bool(kwargs.get("require_exact_runtime", False)),
            )

            # Create run.sh
            run_sh_path = os.path.join(job_dir, "run.sh")
            with open(run_sh_path, "w") as f:
                f.write("#!/usr/bin/env bash\n")
                f.write("set -euo pipefail\n\n")
                f.write(run_script)
            os.chmod(run_sh_path, 0o755)

            # Create meta.env
            meta = {
                "JOB_ID": job_id,
                "TIMEOUT_SECS": effective_timeout,
                # Run from job root for compatibility with run.sh scripts that
                # access both meta.env and payload/.
                "WORKDIR": ".",
                "DOCKER_IMAGE": docker_image or "native",
                "EXECUTION_MODE": execution_mode,
                "NATIVE_PYTHON": native_python,
                "HAS_GPU": "1" if has_gpu else "0",
                "STAGED_ARTIFACT_PATH": staged_artifact_path,
            }
            meta_path = os.path.join(job_dir, "meta.env")
            with open(meta_path, "w") as f:
                for key, value in meta.items():
                    f.write(f"{key}={value}\n")
            staging_snapshot = _snapshot_staged_job(Path(job_dir))
            if not staging_snapshot.get("run_sh_exists"):
                return {
                    "status": "error",
                    "device_id": device_id,
                    "job_id": job_id,
                    "error": "edge payload packaging failed before submit: staged job missing top-level run.sh",
                    "failed_stage": "payload_packaging",
                    "staging_snapshot": staging_snapshot,
                }
            logger.debug(f"EdgeRunner staged job snapshot: {json.dumps(staging_snapshot)[:1200]}")

            # Run job and collect results. collect_results can fail transiently
            # due to tar validation races on unstable links; retry once.
            collection_dir = Path(workspace_path or job_dir).resolve() / ".edgecraft" / "edge-results"
            collection_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            collect_cmd = [
                "--edge", device_ip,
                "--key", ssh_key,
                "--job-dir", job_dir,
                "--collect-out", str(collection_dir),
                "--stream",
                "--collect-retries", "4",
                "--collect-retry-delay", "3",
                "--collect-extract",
                "--collect-keep-job",
                "--no-preflight-cleanup",
                "--stream-grace-secs", "5",
                "--wait-secs", str(effective_timeout)
            ]
            remote_base = self._get_remote_base(
                device_id,
                device_ip=device_ip,
                remote_base=kwargs.get("remote_base"),
            )
            if remote_base:
                collect_cmd.extend(["--remote-base", remote_base])
            runner_min_free_mb = (
                kwargs.get("runner_min_free_mb")
                or device_config.get("runner_min_free_mb")
            )
            inline_requested = edgecraft_env("EDGE_RUN_INLINE").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if (
                bool(device_config.get("inline_runner"))
                or bool(kwargs.get("run_inline"))
                or inline_requested
            ):
                collect_cmd.append("--run-inline")
                if runner_min_free_mb is not None:
                    collect_cmd.extend(
                        ["--inline-min-free-mb", str(int(runner_min_free_mb))]
                    )
            result: Dict[str, Any] = {"status": "error", "error": "unknown"}
            for attempt in range(2):
                result = self._run_edge_script(
                    "run_job_and_collect.sh",
                    collect_cmd,
                    timeout=effective_timeout + 120,
                    cwd=job_dir,
                )
                if result.get("status") == "success":
                    break
                blob = "\n".join(
                    str(result.get(k, "") or "")
                    for k in ("error", "stderr", "stdout")
                ).lower()
                if re.search(r"failed to collect .*file|failed to collect valid result tar|collect_results", blob):
                    logger.warning(
                        f"EdgeRunner collect_results failed (attempt {attempt + 1}/2), retrying once."
                    )
                    time.sleep(2)
                    continue
                break
            if result["status"] == "success":
                results_dir = str(collection_dir / job_id)
                parsed = self._parse_results(results_dir)
                if not parsed.get("metrics"):
                    trt_metrics = _parse_trtexec_metrics(result.get("stdout", ""))
                    if trt_metrics:
                        parsed["status"] = "success"
                        parsed["metrics"] = trt_metrics
                        parsed["runtime_used"] = parsed.get("runtime_used") or "engine"
                        parsed["runtime_provider"] = parsed.get("runtime_provider") or "TensorRT.trtexec"
                        parsed["error"] = None

                return {
                    "status": parsed.get("status", "success"),
                    "device_id": device_id,
                    "job_id": job_id,
                    "metrics": parsed.get("metrics", {}),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "failed_stage": parsed.get("failed_stage"),
                    "return_code": parsed.get("return_code", result.get("return_code")),
                    "stderr_tail": parsed.get("stderr_tail", (result.get("stderr", "") or "")[-1200:]),
                    "stdout_tail": parsed.get("stdout_tail", (result.get("stdout", "") or "")[-1200:]),
                    "error": parsed.get("error"),
                    "runtime_used": parsed.get("runtime_used"),
                    "runtime_provider": parsed.get("runtime_provider"),
                    "artifact_used": parsed.get("artifact_used"),
                    "fallback_reason": parsed.get("fallback_reason"),
                    "measurement_protocol": parsed.get("measurement_protocol"),
                    "primary_error_tail": parsed.get("primary_error_tail"),
                    **{
                        key: parsed.get(key)
                        for key in _VALIDATION_RESULT_FIELDS
                        if parsed.get(key) is not None
                    },
                    "artifact_bundle_staging": artifact_bundle_staging,
                    **execution_provenance,
                }
            else:
                err = result.get("error", "Unknown error")
                stderr = result.get("stderr", "")
                stdout = result.get("stdout", "")
                streamed = _parse_stdout_metric_payload(stdout)
                if streamed.get("status") == "success" and streamed.get("metrics"):
                    logger.warning(
                        f"EdgeRunner collected execution metrics for {job_id} from the live stream; "
                        "the result archive was unavailable."
                    )
                    return {
                        **streamed,
                        "device_id": device_id,
                        "job_id": job_id,
                        "stdout": stdout,
                        "stderr": stderr,
                        "stdout_tail": stdout[-1200:],
                        "stderr_tail": stderr[-1200:],
                        "collection_status": "error",
                        "collection_error": err,
                        "artifact_bundle_staging": artifact_bundle_staging,
                        **execution_provenance,
                    }
                if err == "Unknown error" and (stderr or stdout):
                    hint = (stderr or stdout).strip().splitlines()
                    err = hint[-1][:500] if hint else "Unknown error"
                logger.warning(
                    f"EdgeRunner job {job_id} failed: {err}"
                    + (f"\n  stderr(tail): {stderr[-800:]}" if stderr else "")
                    + (f"\n  stdout(tail): {stdout[-800:]}" if stdout else "")
                )
                return {
                    "status": "error",
                    "device_id": device_id,
                    "job_id": job_id,
                    "error": err,
                    "failed_stage": result.get("failed_stage", _infer_failed_stage(f"{stderr}\n{stdout}\n{err}")),
                    "return_code": result.get("return_code"),
                    "stderr_tail": result.get("stderr_tail", stderr[-1200:] if stderr else ""),
                    "stdout_tail": result.get("stdout_tail", stdout[-1200:] if stdout else ""),
                    "command": result.get("command", "run_job_and_collect.sh"),
                    "stderr": stderr,
                    "stdout": stdout,
                    "artifact_bundle_staging": artifact_bundle_staging,
                    **execution_provenance,
                }

        finally:
            if not getattr(settings, "KEEP_EDGE_JOB_DIR", False):
                _safe_cleanup_tree(Path(job_dir))

    def _generate_infer_run_script(
        self,
        docker_image: str,
        has_gpu: bool,
        dataset_dir_rel: Optional[str] = None,
        config_path_rel: Optional[str] = None,
        export_format: str = "onnx",
        quant_mode: str = "fp16",
        deploy_target: Optional[str] = None,
        execution_mode: str = "docker",
        native_python: str = "python3",
        graph_hash: str = "",
        source_artifact_hash: str = "",
        input_profile: str = "default",
        artifact_path_rel: str = "",
        edge_eval_manifest_rel: str = "outputs/edge_eval_manifest.json",
        energy_sampler: str = "",
        require_exact_runtime: bool = False,
    ) -> str:
        """Generate run.sh that executes workspace's infer.py."""
        docker_opts_literal = edge_runner_infer_docker_opts_bash(has_gpu)
        power_sampler_script = edge_power_sampler_bash(energy_sampler, docker_image)
        edge_eval_enabled = "1" if dataset_dir_rel else "0"
        edge_dataset_dir = dataset_dir_rel or ""
        edge_config_path = config_path_rel or "config/data.yaml"
        edge_eval_manifest = str(edge_eval_manifest_rel or "")
        if not re.fullmatch(r"outputs/[A-Za-z0-9._/-]+", edge_eval_manifest):
            edge_eval_manifest = "outputs/edge_eval_manifest.json"
        deploy_fmt = (deploy_target or export_format or "onnx").lower().strip(".")
        quant = (quant_mode or "fp16").lower()
        graph_key = re.sub(r"[^a-fA-F0-9]", "", graph_hash) or "unknown_graph"
        source_key = re.sub(r"[^a-fA-F0-9]", "", source_artifact_hash) or "unknown_weights"
        profile_key = re.sub(r"[^A-Za-z0-9_.-]", "_", input_profile) or "default"
        declared_artifact = str(artifact_path_rel or "")
        if not re.fullmatch(r"outputs/[A-Za-z0-9._-]+", declared_artifact):
            declared_artifact = ""
        declared_format = Path(declared_artifact).suffix.lower().lstrip(".")
        exact_runtime = bool(require_exact_runtime)
        try:
            trt_build_timeout = max(
                1,
                int(edgecraft_env("TRT_BUILD_TIMEOUT_SECS", "900")),
            )
        except ValueError:
            trt_build_timeout = 900
        canonical_deploy_fmt = "engine" if deploy_fmt == "tensorrt" else deploy_fmt
        initial_artifact_path = f"outputs/best.{canonical_deploy_fmt}"
        engine_source_path = (
            declared_artifact
            if declared_artifact and declared_format == "onnx"
            else "outputs/best.onnx"
        )
        if declared_artifact and declared_format == canonical_deploy_fmt:
            initial_artifact_path = declared_artifact
        native_runtime_fallback = ""
        if not exact_runtime:
            native_runtime_fallback = f'''if [[ "{deploy_fmt}" == "tflite" && ! -f "$EDGECRAFT_ARTIFACT_PATH" && -f "outputs/best.onnx" ]]; then
    EDGECRAFT_RUNTIME="onnx"
    EDGECRAFT_ARTIFACT_PATH="outputs/best.onnx"
    FALLBACK_REASON="tflite_unavailable"
elif [[ "{deploy_fmt}" == "onnx" && ! -f "$EDGECRAFT_ARTIFACT_PATH" && -f "outputs/best.tflite" ]]; then
    EDGECRAFT_RUNTIME="tflite"
    EDGECRAFT_ARTIFACT_PATH="outputs/best.tflite"
    FALLBACK_REASON="onnx_unavailable"
fi'''
        if str(execution_mode or "docker").lower() == "native":
            return f'''
echo "[edgecraft] Running infer.py benchmark"
echo "[edgecraft] Execution mode: native"
echo "[edgecraft] Native python: {native_python}"
echo "[edgecraft] Deploy target: {deploy_fmt} (quant={quant})"
echo "[edgecraft] Edge dataset eval enabled: {edge_eval_enabled}"
echo "[edgecraft] Edge dataset dir: {edge_dataset_dir}"

{power_sampler_script}

cd payload

echo "[edgecraft] Payload contents:"
ls -la
ls -la outputs/ 2>/dev/null || echo "No outputs directory"

if [[ ! -f "infer.py" ]]; then
    echo "[edgecraft] ERROR: infer.py not found"
    exit 1
fi

PYTHON_BIN="{native_python}"
EDGECRAFT_RUNTIME="{deploy_fmt}"
EDGECRAFT_ARTIFACT_PATH="{initial_artifact_path}"
FALLBACK_REASON=""
{native_runtime_fallback}
echo "[edgecraft] Runtime selected: $EDGECRAFT_RUNTIME artifact=$EDGECRAFT_ARTIFACT_PATH fallback=$FALLBACK_REASON"

export EDGE_EVAL_DATASET={edge_eval_enabled}
export EDGE_EVAL_MANIFEST='{edge_eval_manifest}'
export EDGE_DATASET_DIR='{edge_dataset_dir}'
export EDGE_DATASET_CONFIG='{edge_config_path}'
export EDGECRAFT_RUNTIME="$EDGECRAFT_RUNTIME"
export EDGECRAFT_ARTIFACT_PATH="$EDGECRAFT_ARTIFACT_PATH"
export EDGECRAFT_FALLBACK_REASON="$FALLBACK_REASON"
edgecraft_start_power_sampler
set +e
"$PYTHON_BIN" infer.py 2>&1 | tee .edgecraft_primary_infer.log
INFER_EXIT_CODE=${{PIPESTATUS[0]}}
set -e
edgecraft_stop_power_sampler
exit $INFER_EXIT_CODE
'''
        if quant == "int8":
            fp16_flag = "--int8 --fp16"
        else:
            fp16_flag = "--fp16" if quant == "fp16" else ""

        engine_build_block = ""
        raw_engine_fallback_block = ""
        runtime_fallback_block = ""
        pt_failure_fallback_block = ""
        if deploy_fmt in ("engine", "tensorrt"):
            engine_build_failure = (
                '''echo "[edgecraft] failed_stage=engine_build"
        exit "$ENGINE_EXIT"'''
                if exact_runtime
                else '''echo "[edgecraft] WARNING: trtexec engine build failed or timed out with code $ENGINE_EXIT; falling back to ONNX/PT runtime"'''
            )
            engine_build_block = f'''
# TensorRT engine build (cloud train exports ONNX; edge builds engine)
if [[ ! -f "outputs/best.engine" && -f "{engine_source_path}" ]]; then
    echo "[edgecraft] deploy_target=engine -> building TRT engine on edge"
    echo "[edgecraft] trtexec --onnx={engine_source_path} --saveEngine=outputs/best.engine {fp16_flag}"
    mkdir -p "$HOME/.cache/edgecraft/tensorrt"
    if docker run $DOCKER_OPTS -v "$HOME/.cache/edgecraft/tensorrt:/edgecraft-trt-cache" {docker_image} bash -c "
        TRT_BUILD_TIMEOUT={trt_build_timeout}
        TRTEXEC=\\$(command -v trtexec 2>/dev/null || true)
        if [[ -z \\\"\\$TRTEXEC\\\" && -x /usr/src/tensorrt/bin/trtexec ]]; then
            TRTEXEC=/usr/src/tensorrt/bin/trtexec
        fi
        if [[ -z \\\"\\$TRTEXEC\\\" ]]; then
            echo '[edgecraft] ERROR: trtexec not found in container'
            exit 1
        fi
        echo \\\"[edgecraft] using \\$TRTEXEC\\\"
        TRT_VERSION=\\$(\\\"\\$TRTEXEC\\\" --version 2>&1 | tail -1)
        TRT_CACHE_MATERIAL='{graph_key}|{source_key}|{quant}|{profile_key}|'\\\"\\$TRT_VERSION\\\"
        TRT_CACHE_KEY=\\$(printf '%s' \\\"\\$TRT_CACHE_MATERIAL\\\" | sha256sum | cut -d' ' -f1)
        TRT_CACHE_PATH=/edgecraft-trt-cache/\\$TRT_CACHE_KEY.engine
        TRT_TIMING_MATERIAL='{graph_key}|{quant}|{profile_key}|'\\\"\\$TRT_VERSION\\\"
        TRT_TIMING_KEY=\\$(printf '%s' \\\"\\$TRT_TIMING_MATERIAL\\\" | sha256sum | cut -d' ' -f1)
        TRT_TIMING_PATH=/edgecraft-trt-cache/\\$TRT_TIMING_KEY.timing
        TRT_TIMING_FLAG=''
        if \\\"\\$TRTEXEC\\\" --help 2>&1 | grep -q -- '--timingCacheFile'; then
            TRT_TIMING_FLAG=\\\"--timingCacheFile=\\$TRT_TIMING_PATH\\\"
            if [[ -s \\\"\\$TRT_TIMING_PATH\\\" ]]; then
                echo '[edgecraft] TRT_TIMING_CACHE_HIT=1'
            else
                echo '[edgecraft] TRT_TIMING_CACHE_HIT=0'
            fi
            echo \\\"[edgecraft] TRT_TIMING_CACHE_ID=\\$TRT_TIMING_KEY\\\"
        fi
        START=\\$(date +%s)
        if [[ -s \\\"\\$TRT_CACHE_PATH\\\" ]]; then
            cp \\\"\\$TRT_CACHE_PATH\\\" outputs/best.engine
            echo '[edgecraft] TRT_COMPILE_CACHE_HIT=1'
        else
            timeout \\\"\\$TRT_BUILD_TIMEOUT\\\" \\\"\\$TRTEXEC\\\" --onnx={engine_source_path} --saveEngine=outputs/best.engine {fp16_flag} \\$TRT_TIMING_FLAG
            BUILD_EXIT=\\$?
            if [[ \\$BUILD_EXIT -eq 0 && -s outputs/best.engine ]]; then
                cp outputs/best.engine \\\"\\$TRT_CACHE_PATH.tmp\\\"
                mv \\\"\\$TRT_CACHE_PATH.tmp\\\" \\\"\\$TRT_CACHE_PATH\\\"
            fi
            echo '[edgecraft] TRT_COMPILE_CACHE_HIT=0'
            [[ \\$BUILD_EXIT -eq 0 ]] || exit \\$BUILD_EXIT
        fi
        END=\\$(date +%s)
        echo \\\"[edgecraft] TRT_COMPILE_SECONDS=\\$((END-START))\\\"
    "; then
        ENGINE_EXIT=0
    else
        ENGINE_EXIT=$?
    fi
    if [[ $ENGINE_EXIT -ne 0 ]]; then
        {engine_build_failure}
    fi
fi
'''
            if not exact_runtime:
                raw_engine_fallback_block = r'''
if [[ -f "infer.py" ]] \
   && ! grep -Eq '(^[[:space:]]*(import[[:space:]]+tensorrt|from[[:space:]]+tensorrt[[:space:]]+import)|edgecraft-runtime-driver:[[:space:]]*engine=trtexec)' infer.py; then
    if [[ -f "outputs/best.onnx" ]]; then
        EDGECRAFT_RUNTIME="onnx"
        EDGECRAFT_ARTIFACT_PATH="outputs/best.onnx"
        FALLBACK_REASON="raw_trtexec_engine_requires_tensorrt_python_loader"
        sed -i 's#outputs/best\.engine#outputs/best.onnx#g' infer.py
    fi
fi
'''

        if not exact_runtime:
            runtime_fallback_block = f'''if [[ "{deploy_fmt}" == "engine" && ! -f "$EDGECRAFT_ARTIFACT_PATH" ]]; then
    if [[ -f "outputs/best.onnx" ]]; then
        EDGECRAFT_RUNTIME="onnx"
        EDGECRAFT_ARTIFACT_PATH="outputs/best.onnx"
        FALLBACK_REASON="engine_unavailable"
    elif [[ -f "outputs/best.pt" ]]; then
        EDGECRAFT_RUNTIME="pt"
        EDGECRAFT_ARTIFACT_PATH="outputs/best.pt"
        FALLBACK_REASON="engine_and_onnx_unavailable"
    fi
elif [[ "{deploy_fmt}" == "onnx" && ! -f "$EDGECRAFT_ARTIFACT_PATH" && -f "outputs/best.pt" ]]; then
    EDGECRAFT_RUNTIME="pt"
    EDGECRAFT_ARTIFACT_PATH="outputs/best.pt"
    FALLBACK_REASON="onnx_unavailable"
elif [[ "{deploy_fmt}" == "pt" && ! -f "$EDGECRAFT_ARTIFACT_PATH" && -f "outputs/best.onnx" ]]; then
    EDGECRAFT_RUNTIME="onnx"
    EDGECRAFT_ARTIFACT_PATH="outputs/best.onnx"
    FALLBACK_REASON="pt_unavailable"
fi'''
            pt_failure_fallback_block = r'''if [[ \$status -ne 0 && -f "outputs/best.pt" ]]; then
        export EDGECRAFT_RUNTIME='pt'
        export EDGECRAFT_ARTIFACT_PATH='outputs/best.pt'
        export EDGECRAFT_FALLBACK_REASON='preferred_runtime_failed_try_generated_pytorch'
        echo '[edgecraft] Preferred runtime failed; trying generated PyTorch inference path'
        \$PYTHON_BIN infer.py 2>&1 | tee .edgecraft_pytorch_infer.log
        pt_status=\$?
        if [[ \$pt_status -eq 0 ]]; then
            exit 0
        fi
        export EDGECRAFT_FALLBACK_REASON='generated_pytorch_inference_failed_artifact_present'
        run_pt_fallback
        exit \$?
    fi'''

        return f'''
echo "[edgecraft] Running infer.py benchmark"
echo "[edgecraft] Docker image: {docker_image}"
echo "[edgecraft] Has GPU: {'1' if has_gpu else '0'}"
echo "[edgecraft] Deploy target: {deploy_fmt} (quant={quant})"
echo "[edgecraft] Edge dataset eval enabled: {edge_eval_enabled}"
echo "[edgecraft] Edge dataset dir: {edge_dataset_dir}"

{power_sampler_script}

# Change to payload directory
cd payload

# List contents
echo "[edgecraft] Payload contents:"
ls -la
ls -la outputs/ 2>/dev/null || echo "No outputs directory"

# Build Docker command (shared with docker preflight — see docker_opts.py)
DOCKER_OPTS="{docker_opts_literal}"

# Check if infer.py exists
if [[ ! -f "infer.py" ]]; then
    echo "[edgecraft] ERROR: infer.py not found"
    exit 1
fi
{engine_build_block}

# Runtime/artifact negotiation. infer.py should prefer these env vars, while
# legacy scripts may still use outputs/best.<format> directly.
EDGECRAFT_RUNTIME="{deploy_fmt}"
EDGECRAFT_ARTIFACT_PATH="{initial_artifact_path}"
FALLBACK_REASON=""
{runtime_fallback_block}
{raw_engine_fallback_block}
echo "[edgecraft] Runtime selected: $EDGECRAFT_RUNTIME artifact=$EDGECRAFT_ARTIFACT_PATH fallback=$FALLBACK_REASON"

# Prebuilt-image mode: do not mutate runtime deps at job time.
echo "[edgecraft] Running infer.py benchmark"
edgecraft_start_power_sampler
set +e
docker run $DOCKER_OPTS {docker_image} bash -c "
    set -o pipefail
    mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch /workspace/.tmp
    export TMPDIR=/workspace/.tmp
    PYTHON_BIN=\\$(command -v python3 || command -v python)
    export EDGE_EVAL_DATASET={edge_eval_enabled}
    export EDGE_EVAL_MANIFEST='{edge_eval_manifest}'
    export EDGE_DATASET_DIR='{edge_dataset_dir}'
    export EDGE_DATASET_CONFIG='{edge_config_path}'
    export EDGECRAFT_RUNTIME='$EDGECRAFT_RUNTIME'
    export EDGECRAFT_ARTIFACT_PATH='$EDGECRAFT_ARTIFACT_PATH'
    export EDGECRAFT_FALLBACK_REASON='$FALLBACK_REASON'
    run_pt_fallback() {{
        \\$PYTHON_BIN - <<'PY'
import json, os, resource, time
from pathlib import Path
artifact = Path(os.environ.get('EDGECRAFT_ARTIFACT_PATH', 'outputs/best.pt'))
primary_log = Path('.edgecraft_primary_infer.log')
t0 = time.perf_counter()
loaded = False
reason = os.environ.get('EDGECRAFT_FALLBACK_REASON') or 'pt_runtime_generic_artifact_loadability'
error = ''
primary_error_tail = ''
try:
    import torch
    try:
        torch.load(str(artifact), map_location='cpu', weights_only=False)
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        torch.load(str(artifact), map_location='cpu')
    loaded = True
except Exception as exc:  # keep artifact evidence even if deserialization is runtime-specific
    error = str(exc)[:200]
    reason = reason + ':torch_load_failed'
if primary_log.exists():
    primary_text = primary_log.read_text(errors='ignore')
    primary_error_tail = (
        primary_text
        if len(primary_text) <= 2400
        else primary_text[:1200] + '\n...\n' + primary_text[-1200:]
    )
latency_ms = (time.perf_counter() - t0) * 1000.0
payload = {{
    'status': 'error',
    'artifact_used': str(artifact),
    'runtime_used': 'torch',
    'runtime_provider': 'torch_load_fallback',
    'fallback_reason': reason,
    'primary_error_tail': primary_error_tail,
    'metrics': {{
        'Memory_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        'artifact_loaded': float(loaded),
        'artifact_present': float(artifact.exists()),
        'artifact_load_latency_ms': latency_ms,
        'load_error_present': float(bool(error)),
    }},
    'load_error': error,
}}
if loaded:
    payload['error'] = 'infer.py failed; PyTorch artifact was loadable but no inference completed'
    payload['failed_stage'] = 'inference'
else:
    payload['error'] = error or 'PyTorch artifact could not be loaded'
    payload['failed_stage'] = 'artifact_load'
print(json.dumps(payload))
raise SystemExit(1)
PY
    }}
    \\$PYTHON_BIN infer.py 2>&1 | tee .edgecraft_primary_infer.log
    status=\\$?
    {pt_failure_fallback_block}
    exit \\$status
"

INFER_EXIT_CODE=$?
set -e
edgecraft_stop_power_sampler

# Capture the last JSON line from infer.py output as metrics
echo "[edgecraft] Benchmark complete with exit code $INFER_EXIT_CODE"
exit $INFER_EXIT_CODE
'''

    @staticmethod
    def _stage_edge_dataset_bundle(payload_dir: Path, dataset_path: Optional[str]) -> Dict[str, str]:
        """Bundle a bounded dataset sample and rewrite config/data.yaml.

        Returns relative payload paths used by run.sh env vars.
        """
        out: Dict[str, str] = {}
        if not dataset_path:
            return out
        ds_src = Path(dataset_path).expanduser().resolve()
        if not ds_src.exists():
            logger.warning(f"Edge dataset staging skipped: dataset path not found: {ds_src}")
            return out
        ds_src = _select_dataset_stage_source(ds_src, payload_dir)

        ds_dst = payload_dir / "dataset"
        if ds_dst.exists():
            move_to_trash(ds_dst)
        ds_dst.mkdir(parents=True, exist_ok=True)
        stage_full = edgecraft_env("EDGE_STAGE_FULL_DATASET").lower() in {"1", "true", "yes"}
        if stage_full and ds_src.is_dir():
            _copytree_for_edge(ds_src, ds_dst)
            dataset_rel = "dataset"
            logger.debug(f"Edge dataset full staging enabled: {ds_src} -> {ds_dst}")
        elif stage_full:
            shutil.copy2(ds_src, ds_dst / ds_src.name)
            dataset_rel = f"dataset/{ds_src.name}"
            logger.debug(f"Edge dataset full staging enabled: {ds_src} -> {ds_dst / ds_src.name}")
        else:
            max_files = int(edgecraft_env("EDGE_DATASET_MAX_FILES", "64"))
            max_mb = int(edgecraft_env("EDGE_DATASET_MAX_MB", "64"))
            cfg_path = payload_dir / "config" / "data.yaml"
            stats = _copy_dataset_sample_for_edge(
                ds_src,
                ds_dst,
                max_files=max(1, max_files),
                max_bytes=max(1, max_mb) * 1024 * 1024,
                preferred_relpaths=_declared_edge_sample_paths(cfg_path),
            )
            dataset_rel = "dataset" if ds_src.is_dir() else f"dataset/{ds_src.name}"
            logger.debug(
                "Edge dataset sample staged: "
                f"src={ds_src} files={stats['copied_files']}/{stats['max_files']} "
                f"bytes={stats['copied_bytes']}/{stats['max_bytes']}"
            )
        _chmod_tree_user_readable(ds_dst)

        cfg_path = payload_dir / "config" / "data.yaml"
        if cfg_path.exists():
            try:
                import yaml as _yaml

                cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                if isinstance(cfg, dict):
                    dataset_root_rel = "dataset"
                    for key in ("dataset_path",):
                        if key in cfg:
                            cfg[key] = dataset_rel
                    for key in ("audio_root", "path", "root", "dataset_root", "data_root"):
                        if key in cfg:
                            cfg[key] = dataset_root_rel
                    split_manifest = cfg.get("split_manifest")
                    if isinstance(split_manifest, dict) and split_manifest.get("path"):
                        manifest_src = Path(str(split_manifest["path"])).expanduser()
                        if not manifest_src.is_absolute():
                            manifest_src = (cfg_path.parent / manifest_src).resolve()
                        if manifest_src.is_file():
                            manifest_dst = cfg_path.parent / "split_manifest.json"
                            if manifest_src.resolve() != manifest_dst.resolve():
                                shutil.copy2(manifest_src, manifest_dst)
                            split_manifest = dict(split_manifest)
                            split_manifest["path"] = "split_manifest.json"
                            cfg["split_manifest"] = split_manifest
                        else:
                            logger.warning(
                                "Declared split manifest was not staged because it does not exist: "
                                f"{manifest_src}"
                            )
                    cfg_path.write_text(
                        _yaml.dump(cfg, default_flow_style=False, allow_unicode=True),
                        encoding="utf-8",
                    )
                    logger.debug(f"Edge dataset config rewritten for local payload: {cfg_path}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to rewrite edge data config for payload dataset: {exc}")

        out["dataset_dir_rel"] = "dataset"
        if cfg_path.exists():
            out["config_path_rel"] = "config/data.yaml"
        return out

    def _parse_results(self, results_dir: str) -> Dict[str, Any]:
        """Parse benchmark results from collected output."""
        result = {
            "status": "success",
            "metrics": {},
            "failed_stage": None,
            "return_code": 0,
            "stderr_tail": "",
            "stdout_tail": "",
        }

        if not os.path.exists(results_dir):
            result["status"] = "error"
            result["error"] = f"Results directory not found: {results_dir}"
            result["failed_stage"] = "collect_results"
            return result

        # Start from edge-runner result metadata if present.
        result_json_path = os.path.join(results_dir, "result.json")
        if os.path.exists(result_json_path):
            try:
                with open(result_json_path) as f:
                    edge_meta = json.load(f)
                exit_code = int(edge_meta.get("exit_code", 0))
                result["return_code"] = exit_code
                if exit_code != 0:
                    result["status"] = "error"
                    result["error"] = f"Job exited with code {exit_code}"
            except Exception:
                pass

        # Try to find metrics.json
        metrics_json = os.path.join(results_dir, "metrics.json")
        payload_metrics = os.path.join(results_dir, "payload", "metrics.json")

        metrics_path = None
        if os.path.exists(metrics_json):
            metrics_path = metrics_json
        elif os.path.exists(payload_metrics):
            metrics_path = payload_metrics

        if metrics_path:
            try:
                with open(metrics_path) as f:
                    data = json.load(f)
                result["status"] = data.get("status", "success")
                if isinstance(data.get("metrics"), dict):
                    result["metrics"] = data.get("metrics", {})
                else:
                    # Accept flat JSON metric payloads.
                    result["metrics"] = {
                        k: v for k, v in data.items()
                        if isinstance(v, (int, float))
                    }
                if data.get("error"):
                    result["error"] = data["error"]
                for key in (
                    "runtime_used",
                    "runtime_provider",
                    "artifact_used",
                    "fallback_reason",
                    "measurement_protocol",
                    "primary_error_tail",
                    *_VALIDATION_RESULT_FIELDS,
                ):
                    if data.get(key) is not None:
                        result[key] = data.get(key)
            except json.JSONDecodeError as e:
                result["status"] = "error"
                result["error"] = f"Failed to parse metrics.json: {e}"
        # The wrapper may emit metrics.json while infer.py writes stricter
        # validation metadata to stdout. Keep file metrics authoritative, but
        # always merge missing execution and parity fields from infer.py.
        stdout_log = os.path.join(results_dir, "stdout.log")
        stdout_text = ""
        if os.path.exists(stdout_log):
            with open(stdout_log) as f:
                stdout_text = f.read()
            result["stdout_tail"] = stdout_text[-1200:]

            stdout_payload = _parse_stdout_metric_payload(stdout_text)
            if not result.get("metrics"):
                result["metrics"] = stdout_payload.get("metrics", {})
            for key in (
                "runtime_used",
                "runtime_provider",
                "artifact_used",
                "fallback_reason",
                "measurement_protocol",
                "primary_error_tail",
                *_VALIDATION_RESULT_FIELDS,
            ):
                if result.get(key) is None and stdout_payload.get(key) is not None:
                    result[key] = stdout_payload[key]
        stderr_log = os.path.join(results_dir, "stderr.log")
        if os.path.exists(stderr_log):
            with open(stderr_log) as f:
                stderr = f.read()
            result["stderr_tail"] = stderr[-1200:]
        primary_infer_log = os.path.join(
            results_dir, "payload", ".edgecraft_primary_infer.log"
        )
        if os.path.exists(primary_infer_log):
            with open(primary_infer_log, errors="replace") as f:
                primary_error = f.read()
            if primary_error.strip() and not result.get("primary_error_tail"):
                result["primary_error_tail"] = (
                    primary_error
                    if len(primary_error) <= 2400
                    else primary_error[:1200] + "\n...\n" + primary_error[-1200:]
                )
        trt_metrics = _parse_trtexec_metrics(stdout_text or result.get("stdout_tail", ""))
        if trt_metrics:
            merged_metrics = dict(result.get("metrics") or {})
            for key, value in trt_metrics.items():
                merged_metrics.setdefault(key, value)
            result["metrics"] = merged_metrics
            if not result.get("runtime_used"):
                result["runtime_used"] = result.get("runtime_used") or "engine"
            result["runtime_provider"] = result.get("runtime_provider") or "TensorRT.trtexec"
            if result.get("status") == "error" and result.get("failed_stage") in {None, "collect_results"}:
                result["status"] = "success"
                result["error"] = None

        numeric_metrics = {
            key: value
            for key, value in (result.get("metrics") or {}).items()
            if isinstance(value, (int, float))
        }
        result["metrics"] = numeric_metrics
        result = merge_trusted_energy(result, stdout_text)
        if result.get("status") == "success" and not result.get("metrics"):
            result["status"] = "error"
            result["error"] = "edge benchmark produced no numeric metrics"
            result["failed_stage"] = "collect_results"

        # Check exit code
        exit_code_file = os.path.join(results_dir, "exit_code")
        if os.path.exists(exit_code_file):
            with open(exit_code_file) as f:
                exit_code = int(f.read().strip())
                result["return_code"] = exit_code
                if exit_code != 0 and result["status"] != "error":
                    result["status"] = "error"
                    result["error"] = f"Job exited with code {exit_code}"
        if result.get("status") == "error":
            blob = "\n".join(
                [
                    str(result.get("error", "")),
                    str(result.get("primary_error_tail", "")),
                    str(result.get("stderr_tail", "")),
                    stdout_text,
                ]
            )
            if not result.get("failed_stage"):
                result["failed_stage"] = _infer_failed_stage(blob)
        return result

    # Legacy methods for backward compatibility
    def deploy(
        self,
        artifact_path: str,
        device_id: str,
        device_ip: str = None,
        ssh_key: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy artifact to edge device (legacy interface)."""
        return self.run(
            artifact_path=artifact_path,
            device_id=device_id,
            device_ip=device_ip,
            ssh_key=ssh_key,
            **kwargs
        )

    def run_benchmark(
        self,
        device_id: str,
        device_ip: str,
        model_path: str,
        ssh_key: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference benchmark on edge device (legacy interface)."""
        return self.run(
            artifact_path=model_path,
            device_id=device_id,
            device_ip=device_ip,
            ssh_key=ssh_key,
            **kwargs
        )


# Register the tool
ToolRegistry.register(EdgeRunner())
