"""In-process asynchronous task manager for serve-mode benchmark workflows."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
import uuid

from rich.console import Console

from edgecraft.agent.workspace.manager import trial_bank_path
from edgecraft.api.schemas import CreateTaskRequest, TaskResponse
from edgecraft.config.settings import settings
from edgecraft.config.profiles import profile_default_iterations, profile_settings


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class _TaskRecord:
    task_id: str
    request: Dict[str, Any]
    execution_request: Dict[str, Any] = field(default_factory=dict, repr=False)
    tenant_id: str = "default"
    run_id: Optional[str] = None
    trial_bank_path: Optional[str] = None
    status: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    future: Optional[Future] = None


class TaskManager:
    """Manage async synthesis tasks in a local thread pool."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        # Serve mode has one process-wide mechanism profile. Resolve it once
        # before worker threads start so concurrent tenant requests cannot race
        # through mutable per-request setting overrides.
        active_profile = str(settings.PROFILE or "paper").strip().lower()
        for name, value in profile_settings(active_profile).items():
            setattr(settings, name, value)
        if max_workers is None:
            max_workers = int(settings.TASK_MAX_WORKERS)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="edgecraft-task")
        self._lock = Lock()
        self._tasks: Dict[str, _TaskRecord] = {}

    def create_task(self, req: CreateTaskRequest) -> _TaskRecord:
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        payload = req.model_dump(mode="json")
        run_id = payload.get("run_id") or f"run_{uuid.uuid4().hex[:12]}"
        tenant_id = payload.get("tenant_id") or "default"
        payload["run_id"] = run_id
        public_payload = dict(payload)
        public_payload.pop("ssh_key_path", None)
        if payload.get("credential_handle"):
            public_payload.pop("dataset_path", None)
        rec = _TaskRecord(
            task_id=task_id,
            request=public_payload,
            execution_request=payload,
            tenant_id=tenant_id,
            run_id=run_id,
            trial_bank_path=str(trial_bank_path(run_id, tenant_id=tenant_id)),
        )
        with self._lock:
            self._tasks[task_id] = rec
            rec.future = self._pool.submit(self._run_task, task_id)
        return rec

    def list_tasks(self) -> List[_TaskRecord]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda x: x.created_at)

    def get_task(self, task_id: str) -> Optional[_TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tenants(self) -> List[Dict[str, Any]]:
        with self._lock:
            tenants: Dict[str, Dict[str, Any]] = {}
            for rec in self._tasks.values():
                item = tenants.setdefault(
                    rec.tenant_id,
                    {"tenant_id": rec.tenant_id, "task_count": 0, "active_tasks": 0, "run_ids": []},
                )
                item["task_count"] += 1
                if rec.status in {"queued", "preflight", "running"}:
                    item["active_tasks"] += 1
                if rec.run_id:
                    item["run_ids"].append(rec.run_id)
            return sorted(tenants.values(), key=lambda x: x["tenant_id"])

    def list_tasks_for_tenant(self, tenant_id: str) -> List[_TaskRecord]:
        with self._lock:
            return sorted(
                [rec for rec in self._tasks.values() if rec.tenant_id == tenant_id],
                key=lambda x: x.created_at,
            )

    def tree_lifecycle(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            result = dict(rec.result or {})
            all_trials = result.get("all_trials") or []
            return {
                "task_id": rec.task_id,
                "tenant_id": rec.tenant_id,
                "run_id": rec.run_id,
                "status": rec.status,
                "trial_bank_path": rec.trial_bank_path or result.get("trial_bank_path"),
                "iterations": result.get("iterations", rec.metrics.get("iterations")),
                "trial_count": len(all_trials),
                "all_trials": all_trials,
                "best_trial_id": (result.get("best_trial") or {}).get("trial_id"),
                "error": rec.error,
            }

    def artifact_manifest(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            manifest = self._artifact_manifest_for_record(rec)
            manifest["task_id"] = rec.task_id
            manifest["tenant_id"] = rec.tenant_id
            manifest["run_id"] = rec.run_id
            manifest["status"] = rec.status
            manifest["best_trial_id"] = ((rec.result or {}).get("best_trial") or {}).get("trial_id")
            return manifest

    def artifact_path(self, task_id: str, artifact_key: str) -> Optional[Path]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            manifest = self._artifact_manifest_for_record(rec)
            item = (manifest.get("artifacts") or {}).get(artifact_key)
            if not item:
                return None
            path = Path(item["path"]).expanduser().resolve()
            if not self._path_belongs_to_record(rec, path):
                return None
            if not path.exists() or not path.is_file():
                return None
            return path

    def cancel_task(self, task_id: str) -> Tuple[bool, str]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return False, "task not found"
            if rec.status in {"completed", "completed_infeasible", "failed", "cancelled"}:
                return False, f"task already {rec.status}"
            fut = rec.future
            if fut and fut.cancel():
                rec.status = "cancelled"
                rec.updated_at = _now_iso()
                return True, "task cancelled"
            return False, "task is already running and cannot be cancelled"

    def to_response(self, rec: _TaskRecord) -> TaskResponse:
        return TaskResponse(
            task_id=rec.task_id,
            status=rec.status,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
            tenant_id=rec.tenant_id,
            run_id=rec.run_id,
            trial_bank_path=rec.trial_bank_path,
            request=rec.request,
            result=rec.result,
            artifact_manifest=self._artifact_manifest_for_record(rec),
            error=rec.error,
            metrics=rec.metrics,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run_task(self, task_id: str) -> None:
        # Keep heavyweight graph/model imports out of control-plane startup and
        # reviewer-only authorization tests.
        from edgecraft.agent.graph import run_agent
        from edgecraft.utils.synth_checks import run_synth_preflight

        with self._lock:
            rec = self._tasks[task_id]
            rec.status = "preflight"
            rec.updated_at = _now_iso()
            payload = dict(rec.execution_request or rec.request)

        try:
            console = Console(record=True, width=120)
            preflight = run_synth_preflight(
                payload["intent"],
                payload["dataset_path"],
                payload["device_ip"],
                payload.get("ssh_key_path"),
                console,
                docker_image=payload.get("docker_image"),
                sleep_after_steps=False,
            )
            if not preflight.success:
                raise RuntimeError(preflight.message or "synth preflight failed")
            requested_device = str(payload.get("device_id") or "").strip()
            if requested_device and requested_device != str(preflight.device or ""):
                raise RuntimeError(
                    "probed target device does not match the requested device_id"
                )
            if payload.get("constraints") and preflight.intent is not None:
                from edgecraft.core.task import Constraint

                structured = [
                    Constraint.model_validate(item)
                    for item in payload.get("constraints") or []
                ]
                user_spec = preflight.intent.user_spec.model_copy(
                    update={"constraints": structured}
                )
                preflight.intent = preflight.intent.model_copy(
                    update={"user_spec": user_spec}
                )

            with self._lock:
                rec = self._tasks[task_id]
                rec.status = "running"
                rec.updated_at = _now_iso()

            result = run_agent(
                preflight=preflight,
                ssh_key=payload.get("ssh_key_path"),
                max_iterations=int(
                    payload.get("iterations")
                    or profile_default_iterations(payload.get("profile") or settings.PROFILE)
                ),
                branching_factor=payload.get("branching_factor"),
                run_id=payload.get("run_id"),
                tenant_id=payload.get("tenant_id") or "default",
                force_real_edgebench=bool(payload.get("force_real_edgebench", False)),
            )
            result_status = str(result.get("status") or "")
            status = (
                "completed_infeasible"
                if result_status == "completed_infeasible"
                else "completed" if result_status not in {"error", "failed"}
                else "failed"
            )
            with self._lock:
                rec = self._tasks[task_id]
                rec.result = result
                rec.status = status
                rec.metrics = {
                    "iterations": result.get("iterations"),
                    "run_id": result.get("run_id"),
                    "trial_count": len(result.get("all_trials") or []),
                }
                rec.run_id = result.get("run_id") or rec.run_id
                rec.trial_bank_path = result.get("trial_bank_path") or rec.trial_bank_path
                rec.error = result.get("error")
                rec.updated_at = _now_iso()
        except Exception as exc:
            with self._lock:
                rec = self._tasks[task_id]
                rec.status = "failed"
                rec.error = str(exc)
                rec.updated_at = _now_iso()

    def _artifact_manifest_for_record(self, rec: _TaskRecord) -> Dict[str, Any]:
        result = rec.result or {}
        best = result.get("best_trial") or {}
        raw_paths = best.get("artifact_paths") or {}
        artifacts: Dict[str, Dict[str, Any]] = {}
        for key, raw_path in raw_paths.items():
            if not raw_path:
                continue
            path = Path(str(raw_path)).expanduser().resolve()
            if not self._path_belongs_to_record(rec, path):
                continue
            exists = path.exists() and path.is_file()
            artifacts[str(key)] = {
                "key": str(key),
                "path": str(path),
                "filename": path.name,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
            }
        return {"artifacts": artifacts}

    @staticmethod
    def _path_belongs_to_record(rec: _TaskRecord, path: Path) -> bool:
        """Allow API artifact access only inside the owning run workspace."""
        if not rec.trial_bank_path:
            return False
        root = Path(rec.trial_bank_path).expanduser().resolve().parent
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True


_TASK_MANAGER: Optional[TaskManager] = None
_TASK_MANAGER_LOCK = Lock()


def get_task_manager() -> TaskManager:
    """Return singleton task manager instance."""
    global _TASK_MANAGER
    with _TASK_MANAGER_LOCK:
        if _TASK_MANAGER is None:
            _TASK_MANAGER = TaskManager()
        return _TASK_MANAGER
