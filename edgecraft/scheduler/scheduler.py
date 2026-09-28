"""Persistent stage-ready scheduler for EdgeCraft jobs.

Every submitted candidate becomes a train/export stage followed by a dependent
edge stage (or an edge-only P1 stage). Stages from all concurrent submissions
remain in process-wide ready queues. Whenever a resource becomes available,
the scheduler applies tenant round-robin first, then shortest ready stage within
that tenant, with the tree prior used only as a tie-break.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from loguru import logger

from edgecraft.config.settings import settings
from edgecraft.scheduler.edge_pool import EdgeDevicePool
from edgecraft.scheduler.gpu_pool import GpuPool
from edgecraft.scheduler.types import JobResult, JobSpec, JobStatus
from edgecraft.scheduler.worker import run_edge_stage, run_train_stage


@dataclass
class _JobState:
    spec: JobSpec
    future: asyncio.Future
    sequence: int
    job_start: float
    job_start_wall: float
    train_result: Optional[JobResult] = None
    pending_edge_snapshot: Optional[Path] = None


@dataclass
class _StageTicket:
    state: _JobState
    resource: str
    ready_at: float

    @property
    def tenant_id(self) -> str:
        return self.state.spec.tenant_id or "default"

    @property
    def estimated_seconds(self) -> float:
        estimate = self.state.spec.estimate
        if self.resource == "gpu":
            return max(0.0, float(estimate.gpu_wall_time_s))
        return max(0.0, float(estimate.edge_wall_time_s))


class _StageReadyQueue:
    """Per-resource tenant queues with persistent round-robin cursors."""

    def __init__(self, policy: str) -> None:
        self.policy = policy
        self._queues: Dict[str, Dict[str, List[_StageTicket]]] = {
            "gpu": defaultdict(list),
            "edge": defaultdict(list),
        }
        self._tenants: Dict[str, Deque[str]] = {
            "gpu": deque(),
            "edge": deque(),
        }

    def put(self, ticket: _StageTicket) -> None:
        resource_queues = self._queues[ticket.resource]
        tenant = ticket.tenant_id
        if not resource_queues[tenant]:
            self._tenants[ticket.resource].append(tenant)
        resource_queues[tenant].append(ticket)

    def pop(
        self,
        resource: str,
        *,
        eligible: Optional[Callable[[_StageTicket], bool]] = None,
    ) -> Optional[_StageTicket]:
        tenant_cycle = self._tenants[resource]
        resource_queues = self._queues[resource]
        for _ in range(len(tenant_cycle)):
            tenant = tenant_cycle.popleft()
            queue = resource_queues.get(tenant, [])
            candidates = [item for item in queue if eligible is None or eligible(item)]
            if not candidates:
                if queue:
                    tenant_cycle.append(tenant)
                else:
                    resource_queues.pop(tenant, None)
                continue
            selected = min(candidates, key=self._rank)
            queue.remove(selected)
            if queue:
                tenant_cycle.append(tenant)
            else:
                resource_queues.pop(tenant, None)
            return selected
        return None

    def _rank(self, ticket: _StageTicket) -> tuple:
        spec = ticket.state.spec
        if self.policy == "fifo":
            return (ticket.state.sequence,)
        if self.policy == "tenant_round_robin":
            return (-float(spec.prior_score), ticket.state.sequence)
        return (
            ticket.estimated_seconds,
            -float(spec.prior_score),
            ticket.state.sequence,
        )


class Scheduler:
    """Online scheduler over persistent GPU and edge stage-ready queues."""

    def __init__(
        self,
        policy: Optional[str] = None,
        *,
        gpu_pool: Optional[GpuPool] = None,
        edge_pool: Optional[EdgeDevicePool] = None,
    ) -> None:
        self.gpu_pool = gpu_pool or GpuPool()
        self.edge_pool = edge_pool or EdgeDevicePool()
        self.policy = policy or settings.SCHEDULER_POLICY
        self._ready = _StageReadyQueue(self.policy)
        self._lock: Optional[asyncio.Lock] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sequence = 0
        self._active_gpu = 0
        self._active_edge_devices: set[str] = set()
        self._stage_tasks: set[asyncio.Task] = set()
        self._futures: set[asyncio.Future] = set()

    async def submit_batch(self, specs: List[JobSpec]) -> List[JobResult]:
        """Enqueue jobs atomically and return results in submission order.

        Multiple invocations may overlap. They feed the same stage-ready queues,
        so arrivals after an earlier call's enqueue window still participate in
        every subsequent resource-allocation decision.
        """
        if not specs:
            return []
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._lock = asyncio.Lock()
        elif self._loop is not loop:
            raise RuntimeError("Scheduler must remain on its owning event loop")
        assert self._lock is not None

        states: List[_JobState] = []
        async with self._lock:
            for spec in specs:
                now = time.monotonic()
                state = _JobState(
                    spec=spec,
                    future=loop.create_future(),
                    sequence=self._sequence,
                    job_start=now,
                    job_start_wall=time.time(),
                )
                self._sequence += 1
                self._futures.add(state.future)
                states.append(state)
                if spec.edge_only:
                    state.train_result = self._edge_only_seed(state)
                    self._ready.put(_StageTicket(state, "edge", now))
                else:
                    self._ready.put(_StageTicket(state, "gpu", now))
            self._pump_locked()

        return list(await asyncio.gather(*(state.future for state in states)))

    def _edge_only_seed(self, state: _JobState) -> JobResult:
        spec = state.spec
        result = JobResult(
            job_id=spec.job_id,
            status=JobStatus.SUCCESS,
            stage_reached="artifact_ready",
            artifact_paths=dict(spec.prebuilt_artifact_paths or {}),
            edge_device_id=spec.edge_device_id or "",
            diagnostics={
                "scheduler_stage": spec.stage_label,
                "train_stage": {"requested": False, "executed": False},
            },
        )
        result.stage_times.update(
            {"job_start_wall_s": state.job_start_wall, "job_start_s": 0.0}
        )
        return result

    def _pump_locked(self) -> None:
        gpu_capacity = self.gpu_pool.size * int(self.gpu_pool.max_jobs_per_gpu)
        while self._active_gpu < gpu_capacity:
            ticket = self._ready.pop("gpu")
            if ticket is None:
                break
            self._active_gpu += 1
            self._track(asyncio.create_task(self._execute_gpu(ticket)))

        while True:
            ticket = self._ready.pop(
                "edge",
                eligible=lambda item: (
                    not item.state.spec.edge_device_id
                    or item.state.spec.edge_device_id not in self._active_edge_devices
                ),
            )
            if ticket is None:
                break
            device_id = ticket.state.spec.edge_device_id or ""
            if device_id:
                self._active_edge_devices.add(device_id)
            self._track(asyncio.create_task(self._execute_edge(ticket)))

    def _track(self, task: asyncio.Task) -> None:
        self._stage_tasks.add(task)
        task.add_done_callback(self._stage_tasks.discard)

    async def _execute_gpu(self, ticket: _StageTicket) -> None:
        state = ticket.state
        spec = state.spec
        wait_s = time.monotonic() - ticket.ready_at
        try:
            async with self.gpu_pool.acquire(
                spec.job_id, spec.estimate.gpu_memory_gb
            ) as gpu_id:
                result = await asyncio.to_thread(run_train_stage, spec, gpu_id)
        except Exception as exc:  # noqa: BLE001
            result = JobResult(
                job_id=spec.job_id,
                status=JobStatus.TRAIN_FAILED,
                error_stage="train",
                error=str(exc),
            )
        result.train_wait_s = wait_s
        result.queue_wait_s = wait_s
        result.stage_times.update(
            {
                "job_start_wall_s": state.job_start_wall,
                "job_start_s": 0.0,
                "train_start_s": wait_s,
                "train_end_s": time.monotonic() - state.job_start,
            }
        )
        if result.status == JobStatus.SUCCESS and not spec.skip_edge:
            state.pending_edge_snapshot = self._archive_pending_edge(spec, result)

        assert self._lock is not None
        async with self._lock:
            self._active_gpu -= 1
            state.train_result = result
            if result.status != JobStatus.SUCCESS or spec.skip_edge:
                if spec.skip_edge:
                    result.diagnostics["edge_stage"] = {
                        "requested": False,
                        "executed": False,
                        "reason": "caller_requested_train_only",
                    }
                    if spec.surrogate_edge_metrics:
                        result.diagnostics["surrogate_edge_metrics"] = dict(
                            spec.surrogate_edge_metrics
                        )
                self._finish_locked(state, result)
            else:
                self._ready.put(_StageTicket(state, "edge", time.monotonic()))
            self._pump_locked()

    async def _execute_edge(self, ticket: _StageTicket) -> None:
        state = ticket.state
        spec = state.spec
        result = state.train_result or self._edge_only_seed(state)
        edge_wait_s = time.monotonic() - ticket.ready_at
        try:
            async with self.edge_pool.acquire(spec.edge_device_id, spec.job_id):
                result.stage_times["edge_start_s"] = time.monotonic() - state.job_start
                result.stage_times["edge_start_wall_s"] = time.time()
                result = await asyncio.to_thread(
                    run_edge_stage,
                    spec,
                    result,
                    spec.artifact_manifest,
                )
                result.stage_times["edge_end_s"] = time.monotonic() - state.job_start
                result.stage_times["edge_end_wall_s"] = time.time()
        except Exception as exc:  # noqa: BLE001
            result.status = JobStatus.EDGE_FAILED
            result.error_stage = "edge"
            result.error = str(exc)
        result.edge_wait_s = edge_wait_s
        result.queue_wait_s = float(result.train_wait_s or 0.0) + edge_wait_s
        self._complete_edge_snapshot(state, result)

        assert self._lock is not None
        async with self._lock:
            device_id = spec.edge_device_id or ""
            if device_id:
                self._active_edge_devices.discard(device_id)
            self._finish_locked(state, result)
            self._pump_locked()

    def _finish_locked(self, state: _JobState, result: JobResult) -> None:
        result.total_wall_s = time.monotonic() - state.job_start
        result.stage_times["job_end_s"] = result.total_wall_s
        result.stage_times["job_end_wall_s"] = time.time()
        if not state.future.done():
            state.future.set_result(result)
        self._futures.discard(state.future)

    @staticmethod
    def _archive_pending_edge(spec: JobSpec, result: JobResult) -> Optional[Path]:
        archive_root = str(settings.PENDING_EDGE_ARCHIVE_DIR or "").strip()
        if not archive_root or spec.edge_only:
            return None
        try:
            from edgecraft.scheduler.archive import archive_pending_edge_job

            snapshot = archive_pending_edge_job(
                spec,
                result,
                archive_root=Path(archive_root),
            )
            logger.info(f"Archived pending edge job {spec.job_id}: {snapshot}")
            return snapshot
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not archive pending edge job {spec.job_id}: {exc}")
            return None

    @staticmethod
    def _complete_edge_snapshot(state: _JobState, result: JobResult) -> None:
        if state.pending_edge_snapshot is None:
            return
        try:
            from edgecraft.scheduler.archive import mark_edge_snapshot_completed

            mark_edge_snapshot_completed(
                state.pending_edge_snapshot,
                result,
                source="scheduler",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not complete edge snapshot {state.pending_edge_snapshot}: {exc}"
            )

    def _order_specs(self, specs: List[JobSpec]) -> List[JobSpec]:
        """Project one batch through the policy for compact reviewer tests.

        Runtime execution uses the persistent stage queues above; this helper is
        side-effect free and does not participate in dispatch.
        """
        if self.policy == "fifo":
            return list(specs)
        queues: Dict[str, Deque[JobSpec]] = defaultdict(deque)
        tenants: Deque[str] = deque()
        for spec in specs:
            tenant = spec.tenant_id or "default"
            if not queues[tenant]:
                tenants.append(tenant)
            queues[tenant].append(spec)
        for tenant in list(tenants):
            if self.policy == "cost_aware":
                key = lambda item: (
                    float(item.estimate.gpu_wall_time_s),
                    -float(item.prior_score),
                )
            else:
                key = lambda item: (-float(item.prior_score),)
            queues[tenant] = deque(sorted(queues[tenant], key=key))
        ordered: List[JobSpec] = []
        while tenants:
            tenant = tenants.popleft()
            ordered.append(queues[tenant].popleft())
            if queues[tenant]:
                tenants.append(tenant)
        return ordered

    def close(self) -> None:
        """Cancel queued work during process shutdown."""
        for task in list(self._stage_tasks):
            task.cancel()
        for future in list(self._futures):
            if not future.done():
                future.cancel()
        self._stage_tasks.clear()
        self._futures.clear()
