"""Process-wide shared scheduler runtime.

Why this exists
---------------
`Scheduler` owns asyncio primitives (`Condition`, `Lock`) and should run on a
single event loop. When multiple synth requests run concurrently in different
threads, calling `asyncio.run(scheduler.submit_batch(...))` per thread can
cross event-loop boundaries and cause race/loop-affinity errors.

This module hosts one `Scheduler` inside a dedicated background thread + event
loop, and exposes a thread-safe synchronous submit API.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import List, Optional

from loguru import logger

from edgecraft.config.settings import edgecraft_env
from edgecraft.scheduler.scheduler import Scheduler
from edgecraft.scheduler.types import JobResult, JobSpec


@dataclass
class _PendingRequest:
    specs: List[JobSpec]
    future: Future
    enqueue_time: float
    seq: int
    result_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.result_count = len(self.specs)


class SharedSchedulerRuntime:
    """Thread-safe gateway to a process-wide scheduler instance."""

    def __init__(self):
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._scheduler: Optional[Scheduler] = None
        self._queue: Optional[asyncio.Queue[_PendingRequest]] = None
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._batch_tasks: set[asyncio.Task] = set()
        self._thread = threading.Thread(
            target=self._loop_worker,
            name="edgecraft-shared-scheduler",
            daemon=True,
        )
        self._closed = False
        self._thread.start()
        self._ready.wait(timeout=10.0)
        if self._scheduler is None or self._loop is None:
            raise RuntimeError("SharedSchedulerRuntime failed to initialize.")

    @property
    def scheduler(self) -> Scheduler:
        if self._scheduler is None:
            raise RuntimeError("Shared scheduler not initialized.")
        return self._scheduler

    def submit_batch(self, specs: List[JobSpec], timeout_s: Optional[float] = None) -> List[JobResult]:
        """Submit a job batch from any thread and wait for completion."""
        if self._closed:
            raise RuntimeError("SharedSchedulerRuntime is closed.")
        if not specs:
            return []
        if self._loop is None:
            raise RuntimeError("SharedSchedulerRuntime loop unavailable.")

        min_timeout_s = float(edgecraft_env("SHARED_SCHEDULER_MIN_TIMEOUT_S", "1800"))
        effective_timeout: Optional[float] = timeout_s
        if effective_timeout is not None and min_timeout_s > 0 and effective_timeout < min_timeout_s:
            logger.warning(
                "Shared scheduler timeout is too small for train+edge workloads; "
                f"raise {effective_timeout}s -> {min_timeout_s}s."
            )
            effective_timeout = min_timeout_s

        if self._queue is None:
            raise RuntimeError("SharedSchedulerRuntime queue unavailable.")
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        fut: Future = Future()
        req = _PendingRequest(
            specs=list(specs),
            future=fut,
            enqueue_time=time.monotonic(),
            seq=seq,
        )
        self._loop.call_soon_threadsafe(self._queue.put_nowait, req)
        try:
            return fut.result(timeout=effective_timeout)
        except FutureTimeoutError as exc:
            # Keep a short grace window for long-running single-trial jobs
            # (common in dev micro-bench with real edge bench).
            grace_s = float(edgecraft_env("SHARED_SCHEDULER_TIMEOUT_GRACE_S", "600"))
            if grace_s > 0:
                logger.warning(
                    "Shared scheduler submit timed out after "
                    f"{effective_timeout}s; waiting extra {grace_s}s grace window."
                )
                try:
                    return fut.result(timeout=grace_s)
                except FutureTimeoutError:
                    pass
            fut.cancel()
            raise TimeoutError("submit_batch timed out in shared scheduler runtime") from exc

    def close(self) -> None:
        """Stop the runtime and release executor resources."""
        if self._closed:
            return
        self._closed = True
        if self._loop is None:
            return

        async def _shutdown():
            self.scheduler.close()
            for task in list(self._batch_tasks):
                task.cancel()
            if self._batch_tasks:
                await asyncio.gather(*self._batch_tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SharedSchedulerRuntime.close: graceful shutdown failed: {exc}")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)

    def _loop_worker(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._scheduler = Scheduler()
        self._queue = asyncio.Queue()
        dispatcher = loop.create_task(self._dispatch_loop())
        logger.info(
            "SharedSchedulerRuntime initialized "
            f"(gpu_pool_size={self._scheduler.gpu_pool.size})"
        )
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _dispatch_loop(self) -> None:
        """Forward every arrival immediately to the persistent online scheduler."""
        if self._queue is None:
            raise RuntimeError("SharedSchedulerRuntime queue unavailable.")
        while True:
            request = await self._queue.get()
            # Scheduler.submit_batch enqueues into a process-lifetime ready
            # queue, so each request can be forwarded without an arrival-time
            # microbatch. Staggered requests still compete at the next free
            # GPU/device stage boundary.
            task = asyncio.create_task(self._run_pending_requests([request]))
            self._batch_tasks.add(task)
            task.add_done_callback(self._batch_tasks.discard)

    async def _run_pending_requests(self, requests: List[_PendingRequest]) -> None:
        requests.sort(key=lambda req: req.seq)
        specs: List[JobSpec] = []
        for req in requests:
            specs.extend(req.specs)
        oldest_wait_s = time.monotonic() - min(req.enqueue_time for req in requests)
        logger.debug(
            "SharedSchedulerRuntime dispatching "
            f"{len(specs)} jobs from {len(requests)} request(s) "
            f"(oldest_wait_s={oldest_wait_s:.3f})."
        )
        try:
            results = await self.scheduler.submit_batch(specs)
        except Exception as exc:  # noqa: BLE001
            for req in requests:
                if not req.future.cancelled():
                    req.future.set_exception(exc)
            return

        cursor = 0
        for req in requests:
            chunk = results[cursor : cursor + req.result_count]
            cursor += req.result_count
            if not req.future.cancelled():
                req.future.set_result(chunk)


_shared_runtime_lock = threading.Lock()
_shared_runtime: Optional[SharedSchedulerRuntime] = None


def get_shared_scheduler_runtime() -> SharedSchedulerRuntime:
    """Get or create process-wide shared scheduler runtime."""
    global _shared_runtime
    if _shared_runtime is not None:
        return _shared_runtime
    with _shared_runtime_lock:
        if _shared_runtime is None:
            _shared_runtime = SharedSchedulerRuntime()
    return _shared_runtime


def close_shared_scheduler_runtime() -> None:
    """Close and clear process-wide scheduler runtime."""
    global _shared_runtime
    with _shared_runtime_lock:
        if _shared_runtime is not None:
            _shared_runtime.close()
            _shared_runtime = None
