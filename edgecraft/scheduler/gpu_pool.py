"""GpuPool: integer-slot GPU resource pool.

Phase 1 policy: one Job per GPU (max_jobs_per_gpu = 1). The data structure
is intentionally List[JobId] per GPU rather than a single occupant, so that
future co-location experiments require only a policy change, not a
structural one.

Concurrency model: a single asyncio.Condition coordinates waiters. Each
``acquire()`` blocks until a GPU has a free slot whose remaining vRAM
admits the requested upper bound, then atomically reserves it. ``release()``
notifies all waiters. This is sufficient for the single-process scheduler;
distributed deployments will replace this module.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from edgecraft.config.settings import edgecraft_env


@dataclass
class _GpuState:
    gpu_id: int
    vram_total_gb: float
    jobs: List[str] = field(default_factory=list)
    used_vram_gb: float = 0.0


class GpuPool:
    """Asyncio-aware GPU slot pool.

    Physical GPU IDs can be supplied through ``EDGECRAFT_GPU_IDS``. Otherwise the
    pool uses ``range(EDGECRAFT_GPU_COUNT)`` (default one GPU). vRAM per GPU is
    read from ``EDGECRAFT_GPU_VRAM_GB`` (default 24). These are advisory -- Phase 1
    admits any job whose estimate fits under vram_total_gb.
    """

    def __init__(
        self,
        gpu_count: Optional[int] = None,
        vram_gb_per_gpu: Optional[float] = None,
        max_jobs_per_gpu: int = 1,
        gpu_ids: Optional[List[int]] = None,
    ):
        if gpu_ids is None:
            raw_gpu_ids = edgecraft_env("GPU_IDS").strip()
            gpu_ids = [int(value.strip()) for value in raw_gpu_ids.split(",") if value.strip()]
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(f"GpuPool requires unique GPU IDs, got {gpu_ids}")
        if gpu_ids:
            if gpu_count is not None and gpu_count != len(gpu_ids):
                raise ValueError(
                    f"gpu_count={gpu_count} does not match explicit GPU IDs {gpu_ids}"
                )
            gpu_count = len(gpu_ids)
        if gpu_count is None:
            gpu_count = int(edgecraft_env("GPU_COUNT", "1"))
        if vram_gb_per_gpu is None:
            vram_gb_per_gpu = float(edgecraft_env("GPU_VRAM_GB", "24"))
        if gpu_count <= 0:
            raise ValueError(f"GpuPool requires gpu_count>=1, got {gpu_count}")

        self.max_jobs_per_gpu = max_jobs_per_gpu
        selected_ids = gpu_ids or list(range(gpu_count))
        self._gpus: Dict[int, _GpuState] = {
            gpu_id: _GpuState(gpu_id=gpu_id, vram_total_gb=vram_gb_per_gpu)
            for gpu_id in selected_ids
        }
        self._cond = asyncio.Condition()

    @property
    def size(self) -> int:
        return len(self._gpus)

    def _try_pick(self, mem_gb: float) -> Optional[int]:
        """Best-fit selection: GPU with smallest free vram that still admits."""
        candidates = []
        for gpu in self._gpus.values():
            if len(gpu.jobs) >= self.max_jobs_per_gpu:
                continue
            free = gpu.vram_total_gb - gpu.used_vram_gb
            if free + 1e-6 < mem_gb:
                # Soft violation: still allow if GPU is empty (estimate may
                # be wrong; let it run and surface OOM as a real signal).
                if len(gpu.jobs) > 0:
                    continue
            candidates.append((free, gpu.gpu_id))
        if not candidates:
            return None
        # Smallest free that fits → tightest pack
        candidates.sort()
        return candidates[0][1]

    @asynccontextmanager
    async def acquire(self, job_id: str, mem_gb: float):
        """Wait for a GPU slot, then yield ``gpu_id``. Releases on context exit."""
        async with self._cond:
            while True:
                gpu_id = self._try_pick(mem_gb)
                if gpu_id is not None:
                    g = self._gpus[gpu_id]
                    g.jobs.append(job_id)
                    g.used_vram_gb += mem_gb
                    logger.debug(
                        f"GpuPool: assigned job={job_id} → gpu={gpu_id} "
                        f"(used={g.used_vram_gb:.1f}/{g.vram_total_gb:.1f} GB, "
                        f"jobs={len(g.jobs)}/{self.max_jobs_per_gpu})"
                    )
                    break
                await self._cond.wait()
        try:
            yield gpu_id
        finally:
            async with self._cond:
                g = self._gpus[gpu_id]
                if job_id in g.jobs:
                    g.jobs.remove(job_id)
                g.used_vram_gb = max(0.0, g.used_vram_gb - mem_gb)
                logger.debug(
                    f"GpuPool: released job={job_id} from gpu={gpu_id} "
                    f"(used={g.used_vram_gb:.1f}/{g.vram_total_gb:.1f} GB)"
                )
                self._cond.notify_all()
