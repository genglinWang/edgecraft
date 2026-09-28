"""ProfileStore: Persistent storage for offline profiling results.

This module provides storage and retrieval of model profiling data across
devices, runtimes, and configurations. Key design principles:

1. OFFLINE PROFILING FOCUS: Store latency, memory, throughput - NOT accuracy.
   Accuracy depends on dataset/task and belongs to online trials.

2. UNIFIED SCHEMA: All profile results use the same ProfileResult structure
   regardless of family, runtime, or device.

3. SURROGATE INTEGRATION: ProfileStore feeds into SurrogatePredictor for
   fast feasibility estimation before expensive edge benchmarks.

Profile key dimensions:
- model_id: Family-namespaced model identifier
- device_id: Target device
- runtime_id: Inference runtime (onnxruntime, tensorrt, etc.)
- quant_mode: Quantization (fp32, fp16, int8)
- export_format: Model format (onnx, engine, etc.)
- input_signature: Input dimensions (imgsz, batch_size, etc.)
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from loguru import logger

from edgecraft.models.specs import (
    DeviceClass,
    ExportFormat,
    QuantMode,
    RuntimeId,
)


# ---------------------------------------------------------------------------
# ProfileRequest: Input for offline profiling
# ---------------------------------------------------------------------------

class ProfileRequest(BaseModel):
    """Request specification for offline profiling.

    This defines what to profile - the model, device, runtime, and configuration.
    """

    # Model identity
    model_id: str = Field(..., description="Family-namespaced ID, e.g., 'ultralytics:yolo11n'")
    family_id: str = Field(..., description="Family identifier")
    task_type: str = Field(..., description="Task type string")

    # Artifact
    artifact_path: Optional[str] = Field(
        None,
        description="Path to model artifact (if pre-exported)"
    )
    checkpoint_path: Optional[str] = Field(
        None,
        description="Path to training checkpoint"
    )

    # Export and runtime
    export_format: ExportFormat = ExportFormat.ONNX
    runtime_id: RuntimeId = RuntimeId.ONNXRUNTIME
    quant_mode: QuantMode = QuantMode.FP16

    # Device
    device_id: str = Field(..., description="Target device identifier")
    device_class: DeviceClass = DeviceClass.JETSON

    # Input signature
    input_signature: Dict[str, Any] = Field(
        default_factory=lambda: {"imgsz": 640, "batch_size": 1}
    )

    # Benchmark configuration
    warmup_iterations: int = Field(10, ge=1)
    benchmark_iterations: int = Field(100, ge=10)
    repeat_count: int = Field(1, ge=1, description="Number of benchmark runs")

    # Environment
    docker_image: Optional[str] = None
    environment_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Runtime versions: cuda, tensorrt, pytorch, etc."
    )
    environment_fingerprint: Optional[str] = Field(
        None,
        description="Optional environment signature to avoid cross-env collisions.",
    )

    def get_profile_key(self) -> str:
        """Generate a unique key for this profile configuration."""
        env_fp = self.environment_fingerprint
        if not env_fp and self.environment_versions:
            env_fp = hashlib.sha256(
                json.dumps(self.environment_versions, sort_keys=True).encode()
            ).hexdigest()[:12]
        key_parts = [
            self.model_id,
            self.device_id,
            self.runtime_id.value,
            self.quant_mode.value,
            self.export_format.value,
            json.dumps(self.input_signature, sort_keys=True),
            env_fp or "env:any",
        ]
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ProfileResult: Output from offline profiling
# ---------------------------------------------------------------------------

class ProfileResult(BaseModel):
    """Result of offline profiling.

    Contains execution metrics (latency, memory, etc.) but NOT task accuracy.
    Accuracy depends on dataset and belongs to online trial results.
    """

    # Identity (matches request)
    profile_key: str = Field(..., description="Unique key from ProfileRequest")
    model_id: str
    device_id: str
    runtime_id: RuntimeId
    quant_mode: QuantMode
    export_format: ExportFormat
    input_signature: Dict[str, Any]

    # Status
    status: str = Field("success", description="success | error | timeout")
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Latency metrics (ms)
    latency_avg_ms: float = Field(0.0, ge=0)
    latency_p50_ms: float = Field(0.0, ge=0)
    latency_p95_ms: float = Field(0.0, ge=0)
    latency_p99_ms: float = Field(0.0, ge=0)
    latency_min_ms: float = Field(0.0, ge=0)
    latency_max_ms: float = Field(0.0, ge=0)
    latency_std_ms: float = Field(0.0, ge=0)

    # Throughput
    throughput_fps: float = Field(0.0, ge=0)

    # Memory metrics (MB)
    peak_memory_mb: float = Field(0.0, ge=0)
    model_size_mb: float = Field(0.0, ge=0)
    gpu_memory_mb: Optional[float] = None

    # Timing metrics (ms)
    load_time_ms: float = Field(0.0, ge=0)
    compile_time_ms: float = Field(0.0, ge=0)
    cold_start_ms: float = Field(0.0, ge=0)

    # Power metrics (optional, device-dependent)
    power_w: Optional[float] = None
    energy_mj_per_inference: Optional[float] = None

    # Provenance
    artifact_hash: Optional[str] = None
    logs_path: Optional[str] = None
    benchmark_config: Dict[str, Any] = Field(default_factory=dict)
    environment_versions: Dict[str, str] = Field(default_factory=dict)
    environment_fingerprint: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Raw data for detailed analysis
    raw_latencies: List[float] = Field(
        default_factory=list,
        description="All individual latency measurements (ms)"
    )


# ---------------------------------------------------------------------------
# ProfileStore: Persistent storage for profile results
# ---------------------------------------------------------------------------

class ProfileStore:
    """Storage and retrieval of offline profiling results.

    Provides:
    - Persistent JSON-based storage
    - Query by model, device, runtime, etc.
    - Integration with SurrogatePredictor
    """

    def __init__(self, store_path: Optional[Path] = None):
        """Initialize ProfileStore.

        Args:
            store_path: Path to JSON store file. Defaults to workspaces/profile_store.json
        """
        if store_path is None:
            from edgecraft.config.settings import settings
            base = Path(settings.EDGECRAFT_ROOT) / "workspaces"
            store_path = base / "profile_store.json"

        self.store_path = Path(store_path)
        self._cache: Dict[str, ProfileResult] = {}
        self._load()

    def _load(self) -> None:
        """Load profiles from disk."""
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                for key, value in data.items():
                    self._cache[key] = ProfileResult.model_validate(value)
                logger.debug(f"ProfileStore: loaded {len(self._cache)} profiles from {self.store_path}")
            except Exception as e:
                logger.warning(f"ProfileStore: failed to load from {self.store_path}: {e}")

    def _save(self) -> None:
        """Persist profiles to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump(mode="json") for k, v in self._cache.items()}
        self.store_path.write_text(json.dumps(data, indent=2, default=str))
        logger.debug(f"ProfileStore: saved {len(self._cache)} profiles to {self.store_path}")

    def store(self, result: ProfileResult) -> None:
        """Store a profile result (process-safe with file locking)."""
        lock_path = self.store_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                # Reload from disk to pick up writes from other processes
                self._load()
                self._cache[result.profile_key] = result
                self._save()
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        logger.debug(f"ProfileStore: stored profile {result.profile_key} for {result.model_id}")

    def get(self, profile_key: str) -> Optional[ProfileResult]:
        """Get a profile result by key."""
        return self._cache.get(profile_key)

    def get_by_request(self, request: ProfileRequest) -> Optional[ProfileResult]:
        """Get a profile result matching the request."""
        return self.get(request.get_profile_key())

    def query(
        self,
        model_id: Optional[str] = None,
        device_id: Optional[str] = None,
        runtime_id: Optional[RuntimeId] = None,
        quant_mode: Optional[QuantMode] = None,
        export_format: Optional[ExportFormat] = None,
    ) -> List[ProfileResult]:
        """Query profiles matching the given criteria."""
        results = []
        for profile in self._cache.values():
            if model_id and profile.model_id != model_id:
                continue
            if device_id and profile.device_id != device_id:
                continue
            if runtime_id and profile.runtime_id != runtime_id:
                continue
            if quant_mode and profile.quant_mode != quant_mode:
                continue
            if export_format and profile.export_format != export_format:
                continue
            results.append(profile)
        return sorted(results, key=lambda p: p.timestamp, reverse=True)

    def get_latest(
        self,
        model_id: str,
        device_id: str,
        runtime_id: Optional[RuntimeId] = None,
    ) -> Optional[ProfileResult]:
        """Get the most recent profile for a model on a device."""
        results = self.query(
            model_id=model_id,
            device_id=device_id,
            runtime_id=runtime_id,
        )
        return results[0] if results else None

    def estimate_latency(
        self,
        model_id: str,
        device_id: str,
        quant_mode: QuantMode,
        input_signature: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Estimate latency from stored profiles.

        Returns:
            (latency_ms, confidence) where confidence is in [0, 1].
        """
        # Try exact match first
        results = self.query(
            model_id=model_id,
            device_id=device_id,
            quant_mode=quant_mode,
        )

        # Filter by input signature match
        target_imgsz = input_signature.get("imgsz", 640)
        exact_matches = [
            r for r in results
            if r.input_signature.get("imgsz") == target_imgsz
        ]

        if exact_matches:
            best = exact_matches[0]
            return best.latency_avg_ms, 0.95

        # Try to interpolate from nearby input sizes
        if results:
            # Find closest imgsz
            closest = min(
                results,
                key=lambda r: abs(r.input_signature.get("imgsz", 640) - target_imgsz)
            )
            ref_imgsz = closest.input_signature.get("imgsz", 640)
            scale = (target_imgsz / ref_imgsz) ** 2
            estimated = closest.latency_avg_ms * scale
            return estimated, 0.7

        # No data
        return 0.0, 0.0

    def estimate_memory(
        self,
        model_id: str,
        quant_mode: QuantMode,
    ) -> Tuple[float, float]:
        """Estimate memory from stored profiles.

        Returns:
            (memory_mb, confidence) where confidence is in [0, 1].
        """
        results = self.query(model_id=model_id, quant_mode=quant_mode)
        if results:
            return results[0].peak_memory_mb, 0.9

        # Try any quant mode for this model
        results = self.query(model_id=model_id)
        if results:
            base = results[0].peak_memory_mb
            # Rough scaling by quant mode
            scale = {
                QuantMode.FP32: 2.0,
                QuantMode.FP16: 1.0,
                QuantMode.INT8: 0.55,
                QuantMode.INT4: 0.3,
            }.get(quant_mode, 1.0)
            ref_scale = {
                QuantMode.FP32: 2.0,
                QuantMode.FP16: 1.0,
                QuantMode.INT8: 0.55,
                QuantMode.INT4: 0.3,
            }.get(results[0].quant_mode, 1.0)
            estimated = base * (scale / ref_scale)
            return estimated, 0.6

        return 0.0, 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored profiles."""
        if not self._cache:
            return {"total": 0}

        models = set()
        devices = set()
        runtimes = set()

        for p in self._cache.values():
            models.add(p.model_id)
            devices.add(p.device_id)
            runtimes.add(p.runtime_id.value)

        return {
            "total": len(self._cache),
            "models": len(models),
            "devices": len(devices),
            "runtimes": len(runtimes),
            "model_list": sorted(models),
            "device_list": sorted(devices),
        }


# Module-level singleton
_profile_store: Optional[ProfileStore] = None


def get_profile_store() -> ProfileStore:
    """Get the global ProfileStore instance."""
    global _profile_store
    if _profile_store is None:
        _profile_store = ProfileStore()
    return _profile_store
