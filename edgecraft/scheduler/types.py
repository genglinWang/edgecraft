"""Job contracts between Search Controller and Scheduler.

These dataclasses are the *only* interface across the two layers.
The Search Controller produces JobSpec; the Scheduler returns JobResult.
Field changes here are breaking changes — keep stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from edgecraft.agent.search.solution_variant import SolutionVariant


class JobStatus(str, Enum):
    """Coarse-grained outcome of a Job. Repair semantics live in the search layer."""
    SUCCESS = "success"
    TRAIN_FAILED = "train_failed"
    EDGE_FAILED = "edge_failed"
    OOM = "oom"
    TIMEOUT = "timeout"
    DATA_PREP_FAILED = "data_prep_failed"


@dataclass
class WorkloadEstimate:
    """Best-effort workload estimate attached to every JobSpec.

    These numbers drive admission control, queue ordering, and timeout
    setting. They do not need to be tight; over-estimation only delays a
    requeue.
    """
    gpu_memory_gb: float = 8.0          # upper bound for GPU memory needed by train
    gpu_wall_time_s: float = 1800.0     # expected train wall time
    edge_wall_time_s: float = 60.0      # expected edge benchmark wall time
    confidence: float = 0.20            # 0.85 (history) | 0.50 (heuristic) | 0.20 (fallback)
    source: str = "fallback"            # "trial_bank" | "family_heuristic" | "fallback"


@dataclass
class JobSpec:
    """Self-contained execution unit submitted to the scheduler.

    The scheduler does not interpret model semantics; it only manages
    resources and forwards artifacts between stages.
    """
    job_id: str
    tenant_id: str = "default"
    request_id: str = ""

    # Variant + workspace (already materialized by WorkspaceManager)
    variant: Optional[SolutionVariant] = None
    workspace_path: str = ""
    train_script: str = ""
    infer_script: str = ""

    # Edge target
    edge_device_id: str = ""
    edge_device_ip: str = ""
    edge_ssh_key: str = ""
    edge_docker_image: Optional[str] = None
    dataset_path: str = ""
    export_format: Optional[str] = None
    quant_mode: Optional[str] = None
    require_exact_runtime: bool = True

    # A P1 probe materializes its initialized artifact before scheduling and
    # requests only the exclusive target-device stage.  It still enters the
    # same fair, cost-aware ready queue as P2 jobs.
    edge_only: bool = False
    prebuilt_artifact_paths: Dict[str, str] = field(default_factory=dict)
    artifact_manifest: Optional[Dict[str, Any]] = None
    stage_label: str = "full_pipeline"

    # Legacy execution fields retained for callers that intentionally request a
    # train-only job. Predictions carried here are diagnostics and never become
    # canonical edge measurements or feasibility evidence.
    skip_edge: bool = False
    surrogate_edge_metrics: Optional[Dict[str, float]] = None

    # Estimates and timeouts
    estimate: WorkloadEstimate = field(default_factory=WorkloadEstimate)
    train_timeout_s: int = 7200
    edge_timeout_s: int = 600

    # Repair budget (consumed inside workers; scheduler is unaware of semantics)
    debug_max_retries: int = 0

    # Priority (higher = run sooner; semantic use remains outside the scheduler).
    priority: int = 0

    # LLM-supplied feasibility prior in [0, 1].  Used purely as a SCHEDULING
    # hint inside the JobPool: among siblings of the same tenant, jobs with
    # higher prior_score are pulled first.  Never affects search-tree selection.
    prior_score: float = 0.5

    # Pruning hint from search controller (best feasible local metric to beat)
    prune_check: Optional[Dict[str, Any]] = None


@dataclass
class JobResult:
    """Aggregated outcome of one Job's full lifecycle.

    Field schema is aligned with TrialResult so the scorer can consume it
    without any adapter logic — see pipeline_executor for the mapping.
    """
    job_id: str
    status: JobStatus

    # Stage flags (StageReached enum value strings)
    stage_reached: str = "pending"

    # Metrics
    local_metrics: Dict[str, float] = field(default_factory=dict)
    edge_metrics: Dict[str, float] = field(default_factory=dict)

    # Artifact paths (e.g. "train" -> ws/outputs/best.onnx)
    artifact_paths: Dict[str, str] = field(default_factory=dict)

    # Failure diagnostics
    error: Optional[str] = None
    error_stage: Optional[str] = None
    stdout: str = ""
    stderr: str = ""

    # Repair lineage (list of dicts mirroring TrialResult.DebugAttempt fields)
    debug_attempts: int = 0
    debug_history: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    queue_wait_s: float = 0.0
    train_wait_s: float = 0.0
    edge_wait_s: float = 0.0
    train_wall_s: float = 0.0
    edge_wall_s: float = 0.0
    total_wall_s: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)

    # Placement record (for logging / future analytics)
    gpu_id: Optional[int] = None
    edge_device_id: str = ""

    # Additive evidence-only extension kept last to preserve positional callers.
    diagnostics: Dict[str, Any] = field(default_factory=dict)
