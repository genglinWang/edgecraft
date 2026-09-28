"""WorkloadEstimator: best-effort per-Job resource estimates.

Three tiers, decreasing confidence:
  Tier 1  trial_bank        — wall_time / peak_memory averaged over prior
                              trials matching (model_family, imgsz, batch).
  Tier 2  family_heuristic  — params_m × 16 B (AdamW master + grads + Adam
                              moments) + framework overhead, plus family
                              default step-time × steps/epoch × epochs.
  Tier 3  fallback          — conservative constants.

Edge wall-time is dominated by ProfileStore latency × iterations + Docker
startup overhead.
"""
from __future__ import annotations

from typing import Iterable, Optional

from loguru import logger

from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.agent.search.trial_result import TrialResult
from edgecraft.scheduler.types import WorkloadEstimate


# ---------------------------------------------------------------------------
# Family-level defaults (per-step seconds at imgsz=640, batch=16, single GPU)
# ---------------------------------------------------------------------------
_FAMILY_STEP_SECONDS = {
    "ultralytics": 0.08,
    "timm": 0.05,
    "transformers": 0.10,
    "huggingface": 0.10,
    "anomaly_detection": 0.04,
    "crowd_counting": 0.06,
    "speechbrain": 0.08,
    "whisper": 0.15,
}

_FALLBACK_GPU_MEM_GB = 8.0
_FALLBACK_TRAIN_S = 1800.0
_FALLBACK_EDGE_S = 60.0
_FRAMEWORK_OVERHEAD_GB = 1.2


class WorkloadEstimator:
    """Stateless estimator; reads from TrialBank + FamilyRegistry + ProfileStore."""

    def __init__(self, profile_store=None):
        # ProfileStore is optional; resolved lazily so unit tests can skip it.
        self._profile_store = profile_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(
        self,
        variant: SolutionVariant,
        device_id: str,
        past_trials: Optional[Iterable[TrialResult]] = None,
    ) -> WorkloadEstimate:
        past_trials = list(past_trials or [])

        gpu_mem = self._estimate_gpu_memory(variant, past_trials)
        gpu_wall, source_train = self._estimate_train_wall(variant, past_trials)
        edge_wall, source_edge = self._estimate_edge_wall(variant, device_id)

        # Source / confidence aggregation: take the lower-confidence side.
        if source_train == "trial_bank" and source_edge == "trial_bank":
            confidence, source = 0.85, "trial_bank"
        elif source_train == "fallback" or source_edge == "fallback":
            confidence, source = 0.20, "fallback"
        else:
            confidence, source = 0.50, "family_heuristic"

        return WorkloadEstimate(
            gpu_memory_gb=gpu_mem,
            gpu_wall_time_s=gpu_wall,
            edge_wall_time_s=edge_wall,
            confidence=confidence,
            source=source,
        )

    # ------------------------------------------------------------------
    # GPU memory (upper bound)
    # ------------------------------------------------------------------
    def _estimate_gpu_memory(
        self,
        variant: SolutionVariant,
        past_trials: list,
    ) -> float:
        # Tier 1: trial_bank match
        peak = self._past_peak_memory_gb(variant, past_trials)
        if peak is not None:
            return max(2.0, peak * 1.15) + _FRAMEWORK_OVERHEAD_GB

        # Tier 2: heuristic (params × 16 B + activations rough term)
        params_m = self._lookup_params_m(variant)
        imgsz = variant.get_imgsz()
        batch = self._infer_batch(variant)
        if params_m is not None:
            param_gb = params_m * 16.0 / 1024.0
            act_gb = batch * (imgsz / 640.0) ** 2 * 0.5
            return max(1.5, param_gb + act_gb) + _FRAMEWORK_OVERHEAD_GB

        return _FALLBACK_GPU_MEM_GB

    def _past_peak_memory_gb(
        self,
        variant: SolutionVariant,
        past_trials: list,
    ) -> Optional[float]:
        matches = [
            t for t in past_trials
            if t.variant.model_family == variant.model_family
            and t.variant.model_name == variant.model_name
            and t.local_metrics
            and "gpu_peak_mem_gb" in t.local_metrics.all_metrics
        ]
        if not matches:
            return None
        vals = [t.local_metrics.all_metrics["gpu_peak_mem_gb"] for t in matches[-5:]]
        return sum(vals) / len(vals)

    def _lookup_params_m(self, variant: SolutionVariant) -> Optional[float]:
        try:
            from edgecraft.models import FamilyRegistry, ensure_registries_initialized
            ensure_registries_initialized()
            spec = FamilyRegistry.get_model_by_name(
                variant.model_name, variant.model_family
            )
            if spec is not None and getattr(spec, "params_m", 0) > 0:
                return float(spec.params_m)
        except Exception as exc:
            logger.debug(f"WorkloadEstimator: params_m lookup failed: {exc}")
        return None

    def _infer_batch(self, variant: SolutionVariant) -> int:
        import re
        m = re.search(r"batch[=\s]*(\d+)", variant.train_code or "")
        if m:
            return int(m.group(1))
        return 16

    # ------------------------------------------------------------------
    # Train wall time
    # ------------------------------------------------------------------
    def _estimate_train_wall(
        self,
        variant: SolutionVariant,
        past_trials: list,
    ) -> tuple[float, str]:
        # Tier 1: trial_bank match
        matches = [
            t for t in past_trials
            if t.variant.model_family == variant.model_family
            and t.variant.model_name == variant.model_name
            and t.duration_seconds > 0
        ]
        if matches:
            recent = matches[-5:]
            avg = sum(t.duration_seconds for t in recent) / len(recent)
            return max(60.0, avg), "trial_bank"

        # Tier 2: family heuristic
        step_s = _FAMILY_STEP_SECONDS.get(variant.model_family)
        epochs = self._infer_epochs(variant)
        steps_per_epoch = self._infer_steps_per_epoch(variant)
        if step_s is not None and epochs and steps_per_epoch:
            wall = step_s * epochs * steps_per_epoch * 1.5
            return max(60.0, wall), "family_heuristic"

        return _FALLBACK_TRAIN_S, "fallback"

    def _infer_epochs(self, variant: SolutionVariant) -> int:
        import re
        m = re.search(r"epochs[=\s]*(\d+)", variant.train_code or "")
        if m:
            return int(m.group(1))
        return 50

    def _infer_steps_per_epoch(self, variant: SolutionVariant) -> int:
        # Without dataset size, use a conservative default.
        return 200

    # ------------------------------------------------------------------
    # Edge wall time
    # ------------------------------------------------------------------
    def _estimate_edge_wall(
        self,
        variant: SolutionVariant,
        device_id: str,
    ) -> tuple[float, str]:
        store = self._get_profile_store()
        if store is not None and device_id:
            try:
                from edgecraft.models.specs import QuantMode
                quant = QuantMode(variant.quant_mode)
                lat_ms, conf = store.estimate_latency(
                    model_id=f"{variant.model_family}:{variant.model_name}",
                    device_id=device_id,
                    quant_mode=quant,
                    input_signature={"imgsz": variant.get_imgsz(), "batch_size": 1},
                )
                if conf > 0.5:
                    # 100 iterations + Docker startup (15s) + transfer (5s)
                    wall = (lat_ms / 1000.0) * 110 + 20.0
                    return max(20.0, wall), "trial_bank"
            except Exception as exc:
                logger.debug(f"WorkloadEstimator: ProfileStore edge lookup failed: {exc}")

        # Heuristic: depends on model size + format
        params_m = self._lookup_params_m(variant) or 5.0
        if variant.export_format == "engine":
            base = 0.04 * params_m + 30.0   # TRT export adds startup cost
        elif variant.export_format == "onnx":
            base = 0.02 * params_m + 25.0
        else:
            base = 0.03 * params_m + 25.0
        return max(20.0, base), "family_heuristic"

    def _get_profile_store(self):
        if self._profile_store is not None:
            return self._profile_store
        try:
            from edgecraft.models.profile_store import get_profile_store
            self._profile_store = get_profile_store()
        except Exception as exc:
            logger.debug(f"WorkloadEstimator: profile_store unavailable: {exc}")
            self._profile_store = None
        return self._profile_store
