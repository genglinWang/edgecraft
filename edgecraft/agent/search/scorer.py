"""Scorer: deterministic, LLM-free quality scoring for EdgeCraft trials.

Scoring philosophy
------------------
* Hard constraints (latency, memory, *any* custom metric in UserSpec.constraints)
  must ALL be satisfied for a trial to be declared **feasible**.
* Primary optimisation target is taken from UserSpec.preferences[0].
  Its direction (maximize / minimize) is respected.
* Infeasible trials receive a *soft-penalised* score so the refinement tree retains
  gradient information about how far a node is from the feasible region.

Data-driven metrics
-------------------
Metric names are normalized through ``MetricRegistry`` (alias -> canonical),
so scorer logic stays stable while generated scripts remain flexible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from edgecraft.agent.metrics import MetricRegistry, lookup_trial_metric
from edgecraft.config.settings import settings
from edgecraft.core.task import UserSpec
from edgecraft.agent.search.trial_result import EdgeMetrics, LocalMetrics, TrialResult
from edgecraft.agent.search.verification import (
    GapSlackBrief,
    VerifierPolicy,
    build_l2_verification_report,
)


@dataclass
class ConstraintCheckResult:
    all_satisfied: bool
    violations: List[str] = field(default_factory=list)
    violation_factor: float = 1.0


def _get_metric_value(
    metric_name: str,
    local_metrics: Optional[LocalMetrics],
    edge_metrics: Optional[EdgeMetrics],
) -> Optional[float]:
    """Backward-compatible wrapper around the shared metric evidence lookup."""
    return lookup_trial_metric(metric_name, local_metrics, edge_metrics)


class Scorer:
    """Computes a scalar score for a TrialResult given the UserSpec constraints."""

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def check_constraints(
        self,
        result: TrialResult,
        user_spec: UserSpec,
    ) -> ConstraintCheckResult:
        """Verify every UserSpec.constraint against the trial result metrics."""
        violations: List[str] = []
        max_ratio = 1.0

        execution_contract = (getattr(result, "observations", {}) or {}).get(
            "edge_execution_contract"
        )
        if (
            isinstance(execution_contract, dict)
            and execution_contract.get("artifact_executed") is False
        ):
            reasons = ", ".join(execution_contract.get("reasons") or [])
            suffix = f" ({reasons})" if reasons else ""
            violations.append(f"Artifact execution: not verified{suffix}")
            max_ratio = 2.0

        if bool(getattr(settings, "REQUIRE_EDGE_QUALITY_PARITY", False)):
            validity = (getattr(result, "observations", {}) or {}).get(
                "evaluation_validity"
            )
            if not isinstance(validity, dict) or validity.get("evaluation_valid") is not True:
                reasons = ", ".join(
                    validity.get("reasons") or []
                    if isinstance(validity, dict)
                    else ["evaluation validity evidence is missing"]
                )
                suffix = f" ({reasons})" if reasons else ""
                violations.append(f"Evaluation validity: not verified{suffix}")
                max_ratio = max(max_ratio, 2.0)

        for constraint in user_spec.constraints:
            actual = _get_metric_value(
                constraint.metric,
                result.local_metrics,
                result.edge_metrics,
            )
            if actual is None:
                violations.append(
                    f"{constraint.metric}: not measured "
                    f"(trial stopped at stage={result.stage_reached.value})"
                )
                max_ratio = max(max_ratio, 2.0)
                continue

            target = constraint.target
            satisfied = False
            if constraint.comparison == "lte":
                satisfied = actual <= target
                if not satisfied:
                    ratio = actual / target if target else float("inf")
                    max_ratio = max(max_ratio, ratio)
                    violations.append(
                        f"{constraint.metric}: {actual:.4g} > {target:.4g}"
                        f" (≤{target:.4g} required, {ratio:.2f}x over)"
                    )
            elif constraint.comparison == "gte":
                satisfied = actual >= target
                if not satisfied:
                    ratio = target / actual if actual else float("inf")
                    max_ratio = max(max_ratio, ratio)
                    violations.append(
                        f"{constraint.metric}: {actual:.4g} < {target:.4g}"
                        f" (≥{target:.4g} required, {ratio:.2f}x under)"
                    )
            elif constraint.comparison == "eq":
                tol = abs(target * 0.01)
                satisfied = abs(actual - target) <= tol
                if not satisfied:
                    ratio = max(actual, target) / (min(actual, target) or 1e-9)
                    max_ratio = max(max_ratio, ratio)
                    violations.append(
                        f"{constraint.metric}: {actual:.4g} ≠ {target:.4g}"
                        f" (={target:.4g} required)"
                    )

        return ConstraintCheckResult(
            all_satisfied=len(violations) == 0,
            violations=violations,
            violation_factor=max_ratio,
        )

    def build_gap_slack(
        self,
        result: TrialResult,
        user_spec: UserSpec,
    ) -> List[GapSlackBrief]:
        """Return constraint alignment evidence without changing scoring."""
        briefs: List[GapSlackBrief] = []
        for constraint in user_spec.constraints or []:
            actual = _get_metric_value(
                constraint.metric,
                result.local_metrics,
                result.edge_metrics,
            )
            source = "unknown"
            evidence_id = ""
            fidelity = ""
            sigma = None
            calibration_error = 0.0
            report = getattr(result, "verification_report", None)
            if report is not None:
                canonical = MetricRegistry.canonicalize_name(constraint.metric)
                matches = [
                    item
                    for item in list(getattr(report, "evidence", []) or [])
                    if MetricRegistry.canonicalize_name(getattr(item, "quantity", "")) == canonical
                    and getattr(item, "value", None) is not None
                ]
                rank = {"proxy": 0, "static": 1, "measured_congruent": 2}
                if matches:
                    item = max(
                        enumerate(matches),
                        key=lambda pair: (
                            rank.get(getattr(pair[1], "fidelity", ""), -1),
                            1 if getattr(pair[1], "probe_id", "") == "full" else 0,
                            pair[0],
                        ),
                    )[1]
                    actual = float(item.value)
                    evidence_id = str(item.id)
                    fidelity = str(item.fidelity)
                    sigma = item.sigma
                    source = str(item.probe_id)
                    calibration_error = float(item.protocol.get("calibration_error") or 0.0)
            if actual is not None:
                if source == "unknown":
                    source = "edge" if MetricRegistry.is_edge_metric(constraint.metric) else "local"
            briefs.append(
                GapSlackBrief.from_constraint(
                    metric=constraint.metric,
                    comparison=constraint.comparison,
                    target=constraint.target,
                    value=actual,
                    sigma=sigma,
                    source=source,
                    evidence_id=evidence_id,
                    fidelity=fidelity,
                    calibration_error=calibration_error,
                )
            )
        return briefs

    # ------------------------------------------------------------------
    # Primary metric extraction (data-driven from UserSpec)
    # ------------------------------------------------------------------

    def get_primary_metric_value(
        self,
        result: TrialResult,
        user_spec: UserSpec,
    ) -> Tuple[float, str]:
        """Return (value, direction) for the primary optimisation metric.

        Primary metric is determined from UserSpec.preferences[0].
        No ModalityHandler dependency.
        """
        # Determine primary metric from UserSpec
        if user_spec.preferences:
            metric_name = user_spec.preferences[0].metric
            direction = user_spec.preferences[0].direction
        else:
            # Fallback: use first metric from all_metrics
            if result.local_metrics and result.local_metrics.all_metrics:
                metric_name = next(iter(result.local_metrics.all_metrics))
                direction = "maximize"
            else:
                return 0.0, "maximize"

        value = _get_metric_value(metric_name, result.local_metrics, result.edge_metrics)
        if value is None:
            value = 0.0 if direction == "maximize" else float("inf")

        return value, direction

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def compute_score(
        self,
        result: TrialResult,
        user_spec: UserSpec,
    ) -> float:
        """Return a scalar quality score in the range [0, 1] (approximately).

        * If all constraints are satisfied → score = primary_metric_value
          (normalised so higher is always better).
        * If any constraint is violated → score = primary_metric_value /
          violation_factor (soft penalty).
        """
        constraint_check = self.check_constraints(result, user_spec)
        if result.verification_report is None:
            result.verification_report = build_l2_verification_report(result)
        result.gap_slack = self.build_gap_slack(result, user_spec)
        primary_value, direction = self.get_primary_metric_value(result, user_spec)

        # Normalise so that "higher score = better" regardless of metric direction
        if direction == "minimize":
            primary_score = 1.0 / (1.0 + primary_value) if primary_value >= 0 else 0.0
        else:
            primary_score = float(primary_value)

        if constraint_check.all_satisfied:
            score = primary_score
        else:
            score = primary_score / constraint_check.violation_factor

        logger.debug(
            f"Scorer: {result.trial_id}  primary={primary_value:.4f}  "
            f"feasible={constraint_check.all_satisfied}  "
            f"violation_factor={constraint_check.violation_factor:.2f}  "
            f"score={score:.4f}"
        )
        return score

    def is_feasible(self, result: TrialResult, user_spec: UserSpec) -> bool:
        """Return true only after the verifier has accepted full P2 evidence.

        Constraint checks still determine the numerical score and signed gaps,
        but they are not an independent acceptance path.  This keeps proposal
        quality/ranking separate from the verifier's decision authority.
        """
        report = result.verification_report
        recomputed = (
            VerifierPolicy(kappa=float(settings.VERIFIER_KAPPA)).decide(
                list(report.evidence or []),
                user_spec,
                next_probe="",
            )
            if report is not None
            else None
        )
        return bool(
            report is not None
            and report.admission_status == "admitted"
            and report.decision == "accept"
            and recomputed is not None
            and recomputed.action == "accept"
            and self.check_constraints(result, user_spec).all_satisfied
        )

    # ------------------------------------------------------------------
    # Improvement comparison
    # ------------------------------------------------------------------

    def is_better(
        self,
        candidate: TrialResult,
        current_best: TrialResult,
        user_spec: UserSpec,
    ) -> bool:
        """True if candidate improves on current_best w.r.t. the primary metric."""
        c_val, direction = self.get_primary_metric_value(candidate, user_spec)
        b_val, _ = self.get_primary_metric_value(current_best, user_spec)
        if direction == "maximize":
            return c_val > b_val
        return c_val < b_val


# Module-level singleton for convenience
_scorer = Scorer()


def compute_score(result: TrialResult, user_spec: UserSpec) -> float:
    return _scorer.compute_score(result, user_spec)


def is_feasible(result: TrialResult, user_spec: UserSpec) -> bool:
    return _scorer.is_feasible(result, user_spec)
