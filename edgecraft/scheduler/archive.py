"""Durable snapshots for trained jobs waiting on an edge device.

This module stays at the orchestration boundary.  It serializes the existing
JobSpec/JobResult contracts and never inspects datasets, models, or search-tree
semantics.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.scheduler.edge_pool import EdgeDevicePool
from edgecraft.scheduler.types import JobResult, JobSpec, JobStatus, WorkloadEstimate
from edgecraft.scheduler.worker import _edge_artifact_manifest, run_edge_stage


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_stats(path: Path) -> tuple[int, str]:
    """Return stable byte count and content digest for one archived directory."""
    digest = hashlib.sha256()
    size = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    return size, digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "job"


def job_spec_to_dict(spec: JobSpec) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for item in fields(JobSpec):
        value = getattr(spec, item.name)
        if item.name == "variant":
            payload[item.name] = value.model_dump(mode="json") if value is not None else None
        elif item.name == "estimate":
            payload[item.name] = asdict(value)
        else:
            payload[item.name] = value
    return payload


def job_spec_from_dict(payload: Dict[str, Any]) -> JobSpec:
    known = {item.name for item in fields(JobSpec)}
    values = {key: value for key, value in payload.items() if key in known}
    variant = values.get("variant")
    if isinstance(variant, dict):
        values["variant"] = SolutionVariant.model_validate(variant)
    estimate = values.get("estimate")
    if isinstance(estimate, dict):
        values["estimate"] = WorkloadEstimate(**estimate)
    return JobSpec(**values)


def job_result_to_dict(result: JobResult) -> Dict[str, Any]:
    payload = asdict(result)
    payload["status"] = result.status.value
    return payload


def job_result_from_dict(payload: Dict[str, Any]) -> JobResult:
    known = {item.name for item in fields(JobResult)}
    values = {key: value for key, value in payload.items() if key in known}
    values["status"] = JobStatus(values["status"])
    return JobResult(**values)


def _copy_with_record(source: Path, destination: Path, root: Path) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
        size_bytes, sha256 = _directory_stats(destination)
        path_type = "directory"
    else:
        shutil.copy2(source, destination)
        size_bytes, sha256 = destination.stat().st_size, _sha256(destination)
        path_type = "file"
    return {
        "source_path": str(source),
        "archive_path": str(destination.relative_to(root)),
        "path_type": path_type,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def archive_pending_edge_job(
    spec: JobSpec,
    train_result: JobResult,
    archive_root: Path,
    *,
    artifact_manifest: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a trained artifact and its existing orchestration contracts."""
    workspace = Path(spec.workspace_path).resolve() if spec.workspace_path else None

    def declared_file(raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute() and workspace is not None:
            path = workspace / path
        return path.resolve()

    artifact_value = train_result.artifact_paths.get("artifact") or train_result.artifact_paths.get("train")
    if not artifact_value:
        raise ValueError(f"job {spec.job_id} has no trained artifact to archive")
    artifact = declared_file(artifact_value)
    if not artifact.is_file():
        raise FileNotFoundError(f"trained artifact does not exist: {artifact}")

    snapshot_id = f"{_safe_name(spec.job_id)}_{int(datetime.now(timezone.utc).timestamp() * 1_000_000)}"
    snapshot_dir = archive_root.resolve() / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    artifact_record = _copy_with_record(
        artifact,
        snapshot_dir / "artifact" / artifact.name,
        snapshot_dir,
    )
    manifest_requirements: Dict[Path, bool] = {}
    raw_artifacts = (
        artifact_manifest.get("artifacts", {})
        if isinstance(artifact_manifest, dict)
        else {}
    )
    for raw_record in raw_artifacts.values():
        if not isinstance(raw_record, dict) or not raw_record.get("path"):
            continue
        try:
            source = declared_file(str(raw_record["path"]))
        except OSError:
            continue
        manifest_requirements[source] = bool(raw_record.get("required_on_edge"))

    artifact_files: Dict[str, Dict[str, Any]] = {}
    workspace_records: Dict[Path, Dict[str, Any]] = {}
    for key, raw in train_result.artifact_paths.items():
        if not raw:
            continue
        source = declared_file(str(raw))
        if not source.exists() or not (source.is_file() or source.is_dir()):
            continue
        record = workspace_records.get(source)
        if record is None:
            record = _copy_with_record(
                source,
                snapshot_dir / "workspace" / "outputs" / source.name,
                snapshot_dir,
            )
            workspace_records[source] = record
        key_record = dict(record)
        if artifact_manifest is not None:
            key_record["required_on_edge"] = bool(
                source == artifact or manifest_requirements.get(source, False)
            )
        artifact_files[str(key)] = key_record
    primary_record = dict(workspace_records[artifact])
    if artifact_manifest is not None:
        primary_record["required_on_edge"] = True
    artifact_files.setdefault("artifact", primary_record)
    artifact_files.setdefault("train", primary_record)

    contract_files: List[Dict[str, Any]] = []
    candidates: List[Path] = []
    for raw in (spec.train_script, spec.infer_script):
        if raw:
            candidates.append(Path(raw).resolve())
    if workspace and workspace.is_dir():
        candidates.extend(workspace / name for name in ("loader.py", "train.py", "infer.py"))
    seen: set[Path] = set()
    for source in candidates:
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        contract_files.append(
            _copy_with_record(source, snapshot_dir / "workspace" / source.name, snapshot_dir)
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": _utc_now(),
        "state": "pending_edge",
        "job_spec": job_spec_to_dict(spec),
        "train_result": job_result_to_dict(train_result),
        "artifact": artifact_record,
        "artifact_files": artifact_files,
        "contract_files": contract_files,
    }
    manifest_path = snapshot_dir / "pending_edge_job.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_pending_edge_snapshot(manifest_path: Path) -> List[str]:
    """Return integrity errors without mutating the snapshot."""
    manifest_path = manifest_path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"manifest unreadable: {exc}"]
    errors: List[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={payload.get('schema_version')!r}")
    records = [
        payload.get("artifact") or {},
        *(payload.get("artifact_files") or {}).values(),
        *(payload.get("contract_files") or []),
    ]
    checked: set[str] = set()
    for record in records:
        archive_path = str(record.get("archive_path") or "")
        if archive_path in checked:
            continue
        checked.add(archive_path)
        archived = manifest_path.parent / str(record.get("archive_path") or "")
        path_type = str(record.get("path_type") or "file")
        if path_type == "directory":
            if not archived.is_dir():
                errors.append(f"missing archived directory: {archived}")
                continue
            size_bytes, sha256 = _directory_stats(archived)
        else:
            if not archived.is_file():
                errors.append(f"missing archived file: {archived}")
                continue
            size_bytes, sha256 = archived.stat().st_size, _sha256(archived)
        if size_bytes != int(record.get("size_bytes", -1)):
            errors.append(f"size mismatch: {archived}")
        if sha256 != str(record.get("sha256") or ""):
            errors.append(f"sha256 mismatch: {archived}")
    return errors


def mark_edge_snapshot_completed(
    manifest_path: Path,
    result: JobResult,
    *,
    source: str,
) -> Path:
    marker = manifest_path.resolve().parent / "edge_completed.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "snapshot_id": manifest_path.parent.name,
                "completed_at": _utc_now(),
                "source": source,
                "result": job_result_to_dict(result),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def discover_unfinished_snapshots(archive_root: Path) -> List[Path]:
    root = archive_root.resolve()
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.glob("*/pending_edge_job.json")
        if not (path.parent / "edge_completed.json").exists()
    )


async def resume_pending_edge_snapshots(
    manifests: Iterable[Path],
    *,
    edge_pool: Optional[EdgeDevicePool] = None,
    mark_completed: bool = True,
) -> List[JobResult]:
    """Validate and execute archived edge stages without retraining."""
    pool = edge_pool or EdgeDevicePool()

    async def run_one(manifest_path: Path) -> JobResult:
        errors = verify_pending_edge_snapshot(manifest_path)
        if errors:
            raise ValueError(f"invalid pending edge snapshot {manifest_path}: {'; '.join(errors)}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        spec = job_spec_from_dict(payload["job_spec"])
        train_result = job_result_from_dict(payload["train_result"])
        artifact_files = payload.get("artifact_files") or {}
        if artifact_files:
            train_result.artifact_paths = {
                str(key): str(manifest_path.parent / record["archive_path"])
                for key, record in artifact_files.items()
            }
        else:
            artifact = manifest_path.parent / payload["artifact"]["archive_path"]
            train_result.artifact_paths["artifact"] = str(artifact)
            train_result.artifact_paths["train"] = str(artifact)

        archived_contracts = {
            Path(item["archive_path"]).name: manifest_path.parent / item["archive_path"]
            for item in payload.get("contract_files") or []
        }
        archived_infer = archived_contracts.get("infer.py")
        if archived_infer and archived_infer.is_file():
            spec.infer_script = str(archived_infer)
            spec.workspace_path = str(manifest_path.parent / "workspace")

        has_requirement_metadata = any(
            "required_on_edge" in record for record in artifact_files.values()
        )
        required_keys = (
            {
                str(key)
                for key, record in artifact_files.items()
                if record.get("required_on_edge")
            }
            if has_requirement_metadata
            else None
        )
        restored_manifest = _edge_artifact_manifest(
            train_result.artifact_paths,
            spec.workspace_path,
            required_keys=required_keys,
        )

        async with pool.acquire(spec.edge_device_id, spec.job_id):
            result = await asyncio.to_thread(
                run_edge_stage,
                spec,
                train_result,
                restored_manifest,
            )
        if mark_completed:
            mark_edge_snapshot_completed(manifest_path, result, source="resume")
        return result

    return await asyncio.gather(*(run_one(Path(path).resolve()) for path in manifests))
