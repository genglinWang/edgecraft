"""TrialResult: the outcome of running one SolutionVariant through the pipeline."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from edgecraft.agent.contracts import ArtifactContract, CraftingProgress, RuntimeReport
from edgecraft.agent.search.branch_judgment import BranchJudgment
from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.agent.search.verification import GapSlackBrief, VerificationReport


class StageReached(str, Enum):
    """The furthest pipeline stage completed before the trial stopped."""
    PENDING = "pending"
    DATA_PREP = "data_prep"
    TRAIN = "train"
    EDGE_BENCHMARK = "edge_benchmark"


class LocalMetrics(BaseModel):
    """Metrics collected from train.py execution.

    In the code-centric architecture, all metrics are stored in all_metrics.
    The Scorer determines which metric is primary from UserSpec.
    """
    all_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="All metrics from train.py output JSON.",
    )


class TrainingTrace(BaseModel):
    """Compact factual record of one completed training process."""

    points: List[Dict[str, Any]] = Field(
        default_factory=list,
        max_length=32,
        description="Representative epoch/step observations emitted by train.py.",
    )
    num_train: Optional[int] = None
    num_val: Optional[int] = None
    best_step: Optional[float] = None
    stopped_reason: str = ""


class EdgeMetrics(BaseModel):
    """Metrics collected from infer.py execution on edge device."""
    latency_ms: Optional[float] = Field(
        None,
        description="Decision latency in ms; p95 when the paper measurement contract is present.",
    )
    latency_mean_ms: Optional[float] = Field(
        None,
        description="Arithmetic-mean latency retained for diagnostics only.",
    )
    memory_mb: Optional[float] = Field(None, description="Peak memory usage in MB.")
    runtime_used: Optional[str] = None
    runtime_provider: Optional[str] = None
    artifact_used: Optional[str] = None
    throughput_fps: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    all_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Full metrics from infer.py output JSON.",
    )


class DebugAttempt(BaseModel):
    """One debugger repair attempt for train.py or infer.py."""

    stage: str = Field(..., description="train or edge_benchmark")
    attempt: int = Field(..., description="1-based repair attempt index")
    error_signature: str = Field("", description="Classified error signature")
    reason: str = Field("", description="Why debugger was triggered")
    before_hash: str = Field("", description="SHA256 of script before patch")
    after_hash: str = Field("", description="SHA256 of script after patch")
    patch_summary: str = Field("", description="One-line summary of applied patch")
    patch_applied: bool = Field(False, description="Whether the patch was written to the script")
    success: bool = Field(False, description="Whether the next retry after this patch passed the failed stage")


class TrialResult(BaseModel):
    """Complete record of one iterative-search trial (one variant evaluation)."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    trial_id: str
    variant: SolutionVariant
    training_seed: Optional[int] = Field(
        None,
        description="Training seed confirmed by generated code for paired lineage comparisons.",
    )

    # ------------------------------------------------------------------
    # Execution state
    # ------------------------------------------------------------------
    stage_reached: StageReached = StageReached.PENDING
    crafting_progress: CraftingProgress = CraftingProgress.PENDING

    # ------------------------------------------------------------------
    # Metrics (filled stage-by-stage)
    # ------------------------------------------------------------------
    local_metrics: Optional[LocalMetrics] = None
    edge_metrics: Optional[EdgeMetrics] = None
    training_trace: Optional[TrainingTrace] = Field(
        None,
        description=(
            "Measured training history used as prompt evidence only. It never "
            "changes verifier authority or deterministic scoring."
        ),
    )

    # ------------------------------------------------------------------
    # Scoring (filled by Scorer node)
    # ------------------------------------------------------------------
    score: float = 0.0
    is_feasible: bool = False
    branch_judgment: Optional[BranchJudgment] = Field(
        None,
        description=(
            "LLM/fallback semantic judgment about whether this code-space branch "
            "deserves further expansion. Metrics are evidence; this records branch potential."
        ),
    )
    constraint_violations: List[str] = Field(
        default_factory=list,
        description="Human-readable list of violated constraints.",
    )

    # ------------------------------------------------------------------
    # Error tracking
    # ------------------------------------------------------------------
    error: Optional[str] = None
    error_stage: Optional[str] = None
    stdout: str = Field("", description="Captured stdout from script execution.")
    stderr: str = Field("", description="Captured stderr from script execution.")
    debug_attempts: int = Field(
        0,
        description="Total number of debugger attempts for this trial.",
    )
    debug_history: List[DebugAttempt] = Field(
        default_factory=list,
        description="Per-attempt debugger lineage for reproducibility.",
    )
    failure_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured failure diagnostics (failed_stage, tails, return codes).",
    )
    failure_taxonomy: Optional[str] = Field(
        None,
        description="Framework-level failure category used for debugger/search decisions.",
    )
    next_search_hint: Optional[str] = Field(
        None,
        description="Actionable hint for the next proposal/debug step.",
    )
    observations: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Read-only factual evidence produced while running this trial, such as "
            "loader smoke summaries, sample observations, artifact manifests, and "
            "runtime observations. This is evidence, not control flow."
        ),
    )
    verification_report: Optional[VerificationReport] = Field(
        None,
        description="Multi-fidelity verification evidence for this trial.",
    )
    gap_slack: List[GapSlackBrief] = Field(
        default_factory=list,
        description="Constraint gap/slack evidence derived from user requirements.",
    )

    # ------------------------------------------------------------------
    # Crafting contracts
    # ------------------------------------------------------------------
    artifact_contract: ArtifactContract = Field(default_factory=ArtifactContract)
    runtime_report: RuntimeReport = Field(default_factory=RuntimeReport)

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------
    artifact_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="stage name → absolute path of the produced artifact.",
    )
    workspace_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    duration_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_metric(self, name: str) -> Optional[float]:
        """Get a metric value by name from local or edge metrics."""
        # Check local metrics first
        if self.local_metrics and name in self.local_metrics.all_metrics:
            return self.local_metrics.all_metrics[name]
        # Then check edge metrics
        if self.edge_metrics:
            if name in self.edge_metrics.all_metrics:
                return self.edge_metrics.all_metrics[name]
            # Shorthand for common edge metrics
            if name == "Latency":
                return self.edge_metrics.latency_ms
            if name == "Memory_mb":
                return self.edge_metrics.memory_mb
        return None

    def to_prompt_text(self, include_variant: bool = True) -> str:
        """Compact representation for inclusion in LLM prompts."""
        lines = [f"[{self.trial_id}]"]
        if include_variant:
            lines.append(self.variant.short_description())

        if self.local_metrics and self.local_metrics.all_metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" for k, v in list(self.local_metrics.all_metrics.items())[:4]
            )
            lines.append(f"  local: {metrics_str}")
        if self.edge_metrics:
            decision_latency = (
                self.edge_metrics.latency_p95_ms
                if self.edge_metrics.latency_p95_ms is not None
                else self.edge_metrics.latency_ms
            )
            latency = (
                f"{decision_latency:.1f}ms"
                if decision_latency is not None
                else "not measured"
            )
            memory = (
                f"{self.edge_metrics.memory_mb:.0f}MB"
                if self.edge_metrics.memory_mb is not None
                else "not measured"
            )
            lines.append(
                f"  edge: latency_p95={latency}, "
                f"mem={memory}"
            )
            runtime_used = self.edge_metrics.runtime_used or self.runtime_report.runtime_used
            runtime_provider = self.edge_metrics.runtime_provider or self.runtime_report.runtime_provider
            artifact_used = self.edge_metrics.artifact_used or self.runtime_report.artifact_used
            if runtime_used or runtime_provider:
                lines.append(
                    f"  runtime: {runtime_used or '?'} "
                    f"provider={runtime_provider or '?'}"
                )
            if artifact_used:
                lines.append(f"  artifact_used={artifact_used}")
        status = "FEASIBLE" if self.is_feasible else "INFEASIBLE"
        lines.append(
            f"  progress={self.crafting_progress.value}  "
            f"score={self.score:.4f}  status={status}"
        )
        if self.error:
            lines.append(f"  error='{self.error}' at stage={self.error_stage}")
        if self.failure_taxonomy:
            lines.append(f"  failure_taxonomy={self.failure_taxonomy}")
        if self.next_search_hint:
            lines.append(f"  next_hint={self.next_search_hint}")
        if self.verification_report:
            lines.append(
                "  verification="
                f"{self.verification_report.level}/{self.verification_report.status}"
            )
        if self.gap_slack:
            compact = []
            for item in self.gap_slack[:4]:
                if item.known and item.gap is not None and item.slack is not None:
                    compact.append(f"{item.metric}:gap={item.gap:.4g},slack={item.slack:.4g}")
                else:
                    compact.append(f"{item.metric}:unknown")
            lines.append("  gap_slack=" + "; ".join(compact))
        if self.branch_judgment:
            lines.append(f"  {self.branch_judgment.to_prompt_text()}")
        if self.constraint_violations:
            lines.append(f"  violations={self.constraint_violations}")
        return "\n".join(lines)
