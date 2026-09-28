"""TrialBank: persistent, cross-iteration store of all trial results.

Unlike the old `tool_results` list (which was cleared on every re-plan),
TrialBank is never cleared during a search run.  The ProposalGenerator and
PromptSampler read from it to inform LLM proposals.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

from edgecraft.agent.search.trial_result import StageReached, TrialResult
from edgecraft.agent.search.verification import VerifierPolicy
from edgecraft.agent.metrics import MetricRegistry


class TrialBank:
    """In-memory store of TrialResult objects, optionally persisted to JSON."""

    def __init__(
        self,
        run_id: str,
        persist_path: Optional[Path] = None,
    ) -> None:
        self.run_id = run_id
        self._trials: Dict[str, TrialResult] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, trial: TrialResult) -> None:
        self._trials[trial.trial_id] = trial
        logger.debug(
            f"TrialBank.add {trial.trial_id}  score={trial.score:.4f}  "
            f"feasible={trial.is_feasible}  stage={trial.stage_reached.value}"
        )
        if self._persist_path:
            self._save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, trial_id: str) -> Optional[TrialResult]:
        return self._trials.get(trial_id)

    def get_all(self) -> List[TrialResult]:
        return list(self._trials.values())

    def get_top_k(self, k: int = 5, feasible_only: bool = False) -> List[TrialResult]:
        """Return the *k* highest-scoring trials."""
        trials = list(self._trials.values())
        if feasible_only:
            trials = [t for t in trials if t.is_feasible]
        return sorted(trials, key=lambda t: t.score, reverse=True)[:k]

    def get_best_feasible(
        self,
        *,
        require_evaluation_valid: bool = False,
    ) -> Optional[TrialResult]:
        feasible = [t for t in self._trials.values() if t.is_feasible]
        if require_evaluation_valid:
            feasible = [
                t for t in feasible
                if ((t.observations or {}).get("evaluation_validity") or {}).get(
                    "evaluation_valid"
                ) is True
            ]
        if not feasible:
            return None
        return max(feasible, key=lambda t: t.score)

    @staticmethod
    def _has_authoritative_p2(
        trial: TrialResult,
        user_spec: Any = None,
    ) -> bool:
        """Return whether a trial contains an executed, congruent P2 result."""
        report = trial.verification_report
        if report is None or str(report.level).upper() not in {"L2", "P2"}:
            return False
        if report.admission_status != "admitted":
            return False
        try:
            artifact_exists = any(
                Path(str(raw)).expanduser().is_file()
                for raw in trial.artifact_paths.values()
                if raw
            )
        except OSError:
            artifact_exists = False
        if not artifact_exists:
            return False
        authorized = [
            item for item in report.evidence
            if VerifierPolicy.authorized_p2(item)
        ]
        if not authorized:
            return False
        constraints = list(getattr(user_spec, "constraints", []) or [])
        if not constraints:
            return True
        measured = {
            MetricRegistry.canonicalize_name(item.quantity)
            for item in authorized
            if item.value is not None
        }
        return all(
            MetricRegistry.canonicalize_name(constraint.metric) in measured
            for constraint in constraints
        )

    def get_best_verified(self, user_spec: Any = None) -> Optional[TrialResult]:
        """Return the highest-scoring P2-verified candidate, feasible or not."""
        verified = [
            trial for trial in self._trials.values()
            if self._has_authoritative_p2(trial, user_spec)
        ]
        if not verified:
            return None

        def rank(trial: TrialResult) -> tuple[float, float]:
            known_gaps = [
                float(item.normalized_gap_high)
                for item in trial.gap_slack
                if item.known and item.normalized_gap_high is not None
            ]
            total_gap = sum(known_gaps) if known_gaps else float("inf")
            return float(trial.score), -total_gap

        return max(verified, key=rank)

    def get_failures(self, k: int = 3) -> List[TrialResult]:
        """Return a diverse sample of failed / infeasible trials for LLM context."""
        failures = [
            t for t in self._trials.values()
            if not t.is_feasible or t.stage_reached != StageReached.EDGE_BENCHMARK
        ]
        if not failures:
            return []
        # Prefer diversity: latency failures vs accuracy failures vs errors
        lat_failures = [
            t for t in failures
            if any("latency" in v.lower() for v in t.constraint_violations)
        ]
        acc_failures = [
            t for t in failures
            if any("map" in v.lower() or "accuracy" in v.lower() for v in t.constraint_violations)
        ]
        errors = [t for t in failures if t.error is not None]

        result: List[TrialResult] = []
        for pool in (lat_failures, acc_failures, errors, failures):
            for t in pool:
                if t not in result:
                    result.append(t)
                if len(result) >= k:
                    return result
        return result[:k]

    def count(self) -> int:
        return len(self._trials)

    def count_feasible(self) -> int:
        return sum(1 for t in self._trials.values() if t.is_feasible)

    def recent_scores(self, n: int = 5) -> List[float]:
        """Scores of the last *n* trials in insertion order."""
        all_trials = list(self._trials.values())
        return [t.score for t in all_trials[-n:]]

    def has_converged(self, patience: int = 3, min_improvement: float = 0.002) -> bool:
        """True if the best feasible score has not improved over the last *patience* trials."""
        if self.count_feasible() < patience:
            return False
        top = self.get_top_k(k=patience + 1, feasible_only=True)
        if len(top) < 2:
            return False
        best = top[0].score
        older = top[-1].score
        return (best - older) < min_improvement

    # ------------------------------------------------------------------
    # LLM context formatting
    # ------------------------------------------------------------------

    def to_prompt_text(self, max_trials: int = 10) -> str:
        """Format the top-*max_trials* trials as a compact block for LLM prompts."""
        top = self.get_top_k(k=max_trials)
        if not top:
            return "No trials completed yet."
        lines = [f"=== Past Trials (top {len(top)} of {self.count()} total) ==="]
        for t in top:
            lines.append(t.to_prompt_text())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.model_dump(mode="json") for t in self._trials.values()]
        pending = self._persist_path.with_name(
            f".{self._persist_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with pending.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pending, self._persist_path)
        finally:
            pending.unlink(missing_ok=True)

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())
            for item in data:
                trial = TrialResult.model_validate(item)
                self._trials[trial.trial_id] = trial
            logger.info(
                f"TrialBank loaded {len(self._trials)} trials from {self._persist_path}"
            )
        except Exception as exc:
            raise ValueError(
                f"TrialBank at {self._persist_path} is unreadable; refusing to discard history"
            ) from exc
