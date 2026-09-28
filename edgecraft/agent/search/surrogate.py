"""SurrogatePredictor: fast latency/memory estimator to gate EdgeBenchmark calls.

Before spending minutes on an SSH+Docker benchmark run, the PipelineExecutor
calls this module to get a quick estimate.  If the surrogate predicts that
the variant is clearly infeasible (e.g. predicted latency > 1.5× budget),
the EdgeBenchmark stage is skipped.

Estimation priority (highest confidence first):
1. ProfileStore: Offline profiling results (confidence: 0.95)
2. Past trials: Similar trials from this session (confidence: 0.85)
3. FamilyRegistry heuristics: Static fallback estimates (confidence: 0.35)

IMPORTANT DISTINCTION:
- Offline profiling (ProfileStore): Device-side metrics (latency, memory, throughput)
  These are task-independent and stable across different datasets.
- Online trial evaluation: Task accuracy (mAP, accuracy, F1)
  These depend on the specific dataset and training configuration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from loguru import logger

from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.agent.search.trial_result import TrialResult

if TYPE_CHECKING:
    from edgecraft.core.task import UserSpec


class SurrogatePredictor:
    """Quick feasibility estimator using ProfileStore, past trials, and FamilyRegistry.

    Estimation sources (in priority order):
    1. ProfileStore: Offline profiling results for (model, device, runtime, quant)
    2. Past trials: Similar trials from this session with real edge metrics
    3. FamilyRegistry: Static estimation tables with heuristics
    """

    def __init__(self, prune_latency_factor: float = 1.5) -> None:
        """
        Args:
            prune_latency_factor: Skip EdgeBenchmark if predicted latency
                exceeds (lat_max × prune_latency_factor).
        """
        self.prune_latency_factor = prune_latency_factor
        self._profile_store = None

    @property
    def profile_store(self):
        """Lazy-load ProfileStore to avoid circular imports."""
        if self._profile_store is None:
            from edgecraft.models.profile_store import get_profile_store
            self._profile_store = get_profile_store()
        return self._profile_store

    def predict(
        self,
        variant: SolutionVariant,
        device_id: str,
        past_trials: Optional[List[TrialResult]] = None,
    ) -> Tuple[float, float, float]:
        """Return (estimated_latency_ms, estimated_memory_mb, confidence).

        confidence is in [0, 1]; values < 0.5 indicate a rough guess.

        Estimation priority:
        1. ProfileStore (offline profiles) - confidence 0.95
        2. Past trials from this session - confidence 0.85
        3. FamilyRegistry static tables - confidence 0.35
        """
        from edgecraft.models import FamilyRegistry, ensure_registries_initialized
        from edgecraft.models.specs import QuantMode
        ensure_registries_initialized()

        # Get imgsz from train_code if available
        imgsz = variant.get_imgsz()

        # Construct model_id for ProfileStore lookup
        model_id = f"{variant.model_family}:{variant.model_name}"

        # Try QuantMode enum
        try:
            quant_mode = QuantMode(variant.quant_mode)
        except (ValueError, KeyError):
            quant_mode = QuantMode.FP16

        # 1. Try ProfileStore first (highest confidence)
        latency, lat_conf = self.profile_store.estimate_latency(
            model_id=model_id,
            device_id=device_id,
            quant_mode=quant_mode,
            input_signature={"imgsz": imgsz, "batch_size": 1},
        )

        memory, mem_conf = self.profile_store.estimate_memory(
            model_id=model_id,
            quant_mode=quant_mode,
        )

        if lat_conf > 0.5:
            logger.debug(
                f"Surrogate (ProfileStore) {variant.trial_id}: "
                f"lat={latency:.1f}ms mem={memory:.0f}MB conf={lat_conf:.2f}"
            )
            return latency, memory, lat_conf

        # 2. Try past trials (second highest confidence)
        if past_trials:
            similar = [
                t for t in past_trials
                if t.variant.model_name == variant.model_name
                and t.edge_metrics is not None
                and t.edge_metrics.latency_ms is not None
                and t.edge_metrics.memory_mb is not None
            ]
            if similar:
                ref = min(
                    similar,
                    key=lambda t: abs(t.variant.get_imgsz() - imgsz),
                )
                ref_imgsz = ref.variant.get_imgsz()
                scale = (imgsz / ref_imgsz) ** 2 if ref_imgsz else 1.0
                latency = ref.edge_metrics.latency_ms * scale
                memory = ref.edge_metrics.memory_mb
                lat_conf = 0.85
                logger.debug(
                    f"Surrogate (past trials) {variant.trial_id}: "
                    f"lat={latency:.1f}ms mem={memory:.0f}MB conf={lat_conf:.2f}"
                )
                return latency, memory, lat_conf

        # 3. Fall back to FamilyRegistry heuristics (lowest confidence)
        latency, lat_conf = FamilyRegistry.estimate_latency(
            variant.model_name,
            variant.model_family,
            device_id,
            variant.quant_mode,
            imgsz,
        )
        memory = FamilyRegistry.estimate_memory(
            variant.model_name,
            variant.model_family,
            variant.quant_mode,
        )

        logger.debug(
            f"Surrogate (FamilyRegistry) {variant.trial_id}: "
            f"lat={latency:.1f}ms mem={memory:.0f}MB conf={lat_conf:.2f}"
        )
        return latency, memory, lat_conf

    def should_skip_edge_benchmark(
        self,
        variant: SolutionVariant,
        device_id: str,
        lat_max: Optional[float],
        mem_max: Optional[float],
        past_trials: Optional[List[TrialResult]] = None,
    ) -> Tuple[bool, str]:
        """Return (skip, reason).

        Skips if the surrogate predicts latency > lat_max × prune_latency_factor
        OR memory > mem_max × 1.2 (a tighter gate on memory since OOM crashes).
        """
        lat, mem, confidence = self.predict(variant, device_id, past_trials)

        if lat_max and confidence >= 0.4:
            threshold = lat_max * self.prune_latency_factor
            if lat > threshold:
                return (
                    True,
                    f"Surrogate latency {lat:.1f}ms > {threshold:.1f}ms "
                    f"(= {lat_max:.1f}ms × {self.prune_latency_factor})",
                )

        if mem_max and confidence >= 0.4:
            threshold = mem_max * 1.2
            if mem > threshold:
                return (
                    True,
                    f"Surrogate memory {mem:.0f}MB > {threshold:.0f}MB "
                    f"(= {mem_max:.0f}MB × 1.2)",
                )

        return False, ""

    # ------------------------------------------------------------------
    # PUCT prior computation
    # ------------------------------------------------------------------

    def compute_prior(
        self,
        variant: SolutionVariant,
        device_id: str,
        user_spec: Optional[UserSpec],
        past_trials: Optional[List[TrialResult]] = None,
    ) -> float:
        """Compute a PUCT prior in [0.1, 1.0] from surrogate feasibility.

        High prior  → surrogate predicts variant is likely feasible → PUCT
                      explores it sooner (AlphaGo-style policy prior).
        Low prior   → surrogate predicts constraint violations → deprioritised
                      but not excluded (exploration term still nonzero).

        When surrogate confidence is low the prior regresses toward 0.5
        (neutral), avoiding overconfident pruning from noisy estimates.
        """
        if not user_spec:
            return 1.0
        constraints = getattr(user_spec, "constraints", None)
        if not constraints:
            return 1.0

        lat, mem, confidence = self.predict(variant, device_id, past_trials)

        if confidence < 0.2:
            return 0.5  # too uncertain to guide search

        lat_max = next(
            (c.target for c in constraints
             if "latency" in c.metric.lower() and c.comparison == "lte"),
            None,
        )
        mem_max = next(
            (c.target for c in constraints
             if "memory" in c.metric.lower() and c.comparison == "lte"),
            None,
        )

        score = 1.0

        if lat_max and lat_max > 0:
            ratio = lat / lat_max
            if ratio <= 0.8:
                pass  # well within budget
            elif ratio <= 1.0:
                score *= 0.8
            elif ratio <= 1.5:
                score *= max(0.2, 1.0 / ratio)
            else:
                score *= 0.1

        if mem_max and mem_max > 0:
            ratio = mem / mem_max
            if ratio <= 0.8:
                pass
            elif ratio <= 1.0:
                score *= 0.9
            elif ratio <= 1.5:
                score *= max(0.2, 1.0 / ratio)
            else:
                score *= 0.1

        # Blend with confidence: low confidence → regress toward 0.5 (neutral)
        prior = score * confidence + 0.5 * (1.0 - confidence)
        return max(0.1, min(1.0, prior))
