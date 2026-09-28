"""EdgeCraft job scheduler.

Decouples pipeline execution from the iterative tree search controller (LangGraph).
Phase 1 scope: single-process asyncio scheduler with a GPU slot pool and a
per-edge-device mutex. Train and EdgeBench stages of a Job execute on
disjoint resources, enabling intra-request pipelining (Train of trial T+1
can overlap with EdgeBench of trial T).

Public API:
    Scheduler.submit_batch(specs) -> List[JobResult]   (async)
    JobSpec, JobResult, JobStatus                      (dataclasses)
    WorkloadEstimator                                  (estimates per spec)
"""

from edgecraft.scheduler.types import (
    JobSpec,
    JobResult,
    JobStatus,
    WorkloadEstimate,
)
from edgecraft.scheduler.scheduler import Scheduler
from edgecraft.scheduler.shared_runtime import (
    SharedSchedulerRuntime,
    get_shared_scheduler_runtime,
    close_shared_scheduler_runtime,
)

__all__ = [
    "Scheduler",
    "SharedSchedulerRuntime",
    "get_shared_scheduler_runtime",
    "close_shared_scheduler_runtime",
    "JobSpec",
    "JobResult",
    "JobStatus",
    "WorkloadEstimate",
]
