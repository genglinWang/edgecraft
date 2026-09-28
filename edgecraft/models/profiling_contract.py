"""Profiling Contract: Formal definition of offline vs online metrics.

This module defines the boundary between offline profiling (device metrics)
and online evaluation (task accuracy). This distinction is critical for
the EdgeCraft architecture.

==========================================================================
OFFLINE PROFILING vs ONLINE EVALUATION
==========================================================================

┌────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PROFILING                                │
│                    (ProfileStore / OfflineProfiler)                     │
├────────────────────────────────────────────────────────────────────────┤
│ PURPOSE: Device-side execution characteristics                          │
│ WHEN: Pre-computed, independent of user task/dataset                    │
│ CONFIDENCE: High (stable across tasks)                                  │
│                                                                         │
│ INPUTS:                                                                 │
│   - model_id: Family-namespaced model (e.g., 'ultralytics:yolo11n')    │
│   - device_id: Target device (e.g., 'jetson_orin_agx')                │
│   - runtime_id: Inference runtime (e.g., 'tensorrt')                   │
│   - quant_mode: Quantization (fp32, fp16, int8)                        │
│   - export_format: Model format (onnx, engine, pt)                     │
│   - input_signature: Input dimensions (imgsz, batch_size)              │
│                                                                         │
│ OUTPUTS (always device metrics, NOT task accuracy):                     │
│   - latency_avg_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms     │
│   - throughput_fps                                                      │
│   - peak_memory_mb, model_size_mb, gpu_memory_mb                       │
│   - load_time_ms, compile_time_ms, cold_start_ms                       │
│   - power_w, energy_mj_per_inference (if device supports)              │
│                                                                         │
│ NOT INCLUDED (these belong to online evaluation):                       │
│   ❌ accuracy, mAP, precision, recall, F1                              │
│   ❌ Training time or convergence metrics                               │
│   ❌ Any task-specific quality metric                                   │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                        ONLINE EVALUATION                                │
│                    (TrialResult / PipelineExecutor)                     │
├────────────────────────────────────────────────────────────────────────┤
│ PURPOSE: Task-specific quality metrics on user's dataset                │
│ WHEN: During agent search, after training on user's data                │
│ CONFIDENCE: Variable (depends on data, training budget)                 │
│                                                                         │
│ INPUTS:                                                                 │
│   - User's dataset                                                      │
│   - Training configuration (epochs, lr, augmentations)                  │
│   - Model variant + hyperparameters                                     │
│                                                                         │
│ OUTPUTS (task-specific accuracy metrics):                               │
│   - Object Detection: mAP, mAP50, precision, recall                    │
│   - Classification: accuracy_top1, accuracy_top5                       │
│   - Segmentation: IoU, dice coefficient                                │
│   - ASR: WER, CER                                                      │
│   - Text: F1, accuracy, perplexity                                     │
│                                                                         │
│ ALSO CAPTURES (complementary to offline profile):                       │
│   - Actual edge metrics (latency_ms, memory_mb) from real run          │
│   - These refine the surrogate for future predictions                   │
└────────────────────────────────────────────────────────────────────────┘

==========================================================================
WHY THIS DISTINCTION MATTERS
==========================================================================

1. OFFLINE PROFILES ARE STABLE:
   - Latency for 'yolo11n' at imgsz=640, fp16 is measured per target device profile
   - This is true regardless of whether the task is COCO, VOC, or custom data
   - We can pre-compute and cache these values

2. ONLINE ACCURACY IS VARIABLE:
   - mAP for 'yolo11n' depends on: dataset difficulty, class imbalance,
     training epochs, learning rate, data augmentation, etc.
   - We CANNOT pre-compute these - they must be measured per-task

3. SEARCH EFFICIENCY:
   - Offline profiles provide a cheap feasibility prior (SurrogatePredictor)
   - Proxy estimates guide launch order and LLM proposals
   - Only calibrated measured evidence may skip a full evaluation

==========================================================================
INTEGRATION POINTS
==========================================================================

1. SurrogatePredictor uses (in priority order):
   a. ProfileStore offline results (confidence 0.95)
   b. Past TrialResult edge metrics (confidence 0.85)
   c. ModelRegistry static tables (confidence 0.35-0.75)

2. PipelineExecutor:
   - Records SurrogatePredictor output as proxy Evidence
   - Runs train.py → online accuracy metrics (mAP, etc.)
   - Runs infer.py → captures actual edge metrics
   - Stores edge metrics in TrialResult for future surrogate refinement

3. Scorer combines both:
   - Offline estimates remain navigation evidence
   - Online measurements for accuracy scoring
   - Multi-objective optimization across all metrics
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Set

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Metric Categories
# ---------------------------------------------------------------------------

class MetricCategory(str, Enum):
    """Category of a metric - determines which system owns it."""
    OFFLINE_LATENCY = "offline_latency"
    OFFLINE_MEMORY = "offline_memory"
    OFFLINE_POWER = "offline_power"
    OFFLINE_THROUGHPUT = "offline_throughput"
    ONLINE_ACCURACY = "online_accuracy"
    ONLINE_CONVERGENCE = "online_convergence"


class MetricDefinition(BaseModel):
    """Definition of a profiling/evaluation metric."""
    name: str
    category: MetricCategory
    unit: str
    description: str
    higher_is_better: bool
    typical_range: tuple = (0.0, 100.0)


# ---------------------------------------------------------------------------
# Offline Profiling Metrics (ProfileStore owns these)
# ---------------------------------------------------------------------------

OFFLINE_METRICS: Dict[str, MetricDefinition] = {
    # Latency metrics
    "latency_avg_ms": MetricDefinition(
        name="latency_avg_ms",
        category=MetricCategory.OFFLINE_LATENCY,
        unit="ms",
        description="Average inference latency",
        higher_is_better=False,
        typical_range=(1.0, 1000.0),
    ),
    "latency_p50_ms": MetricDefinition(
        name="latency_p50_ms",
        category=MetricCategory.OFFLINE_LATENCY,
        unit="ms",
        description="Median (P50) inference latency",
        higher_is_better=False,
        typical_range=(1.0, 1000.0),
    ),
    "latency_p95_ms": MetricDefinition(
        name="latency_p95_ms",
        category=MetricCategory.OFFLINE_LATENCY,
        unit="ms",
        description="95th percentile inference latency",
        higher_is_better=False,
        typical_range=(1.0, 2000.0),
    ),
    "latency_p99_ms": MetricDefinition(
        name="latency_p99_ms",
        category=MetricCategory.OFFLINE_LATENCY,
        unit="ms",
        description="99th percentile inference latency",
        higher_is_better=False,
        typical_range=(1.0, 3000.0),
    ),
    # Throughput metrics
    "throughput_fps": MetricDefinition(
        name="throughput_fps",
        category=MetricCategory.OFFLINE_THROUGHPUT,
        unit="fps",
        description="Inference throughput in frames per second",
        higher_is_better=True,
        typical_range=(0.1, 1000.0),
    ),
    # Memory metrics
    "peak_memory_mb": MetricDefinition(
        name="peak_memory_mb",
        category=MetricCategory.OFFLINE_MEMORY,
        unit="MB",
        description="Peak memory usage during inference",
        higher_is_better=False,
        typical_range=(50.0, 16000.0),
    ),
    "model_size_mb": MetricDefinition(
        name="model_size_mb",
        category=MetricCategory.OFFLINE_MEMORY,
        unit="MB",
        description="Model file size on disk",
        higher_is_better=False,
        typical_range=(1.0, 2000.0),
    ),
    "gpu_memory_mb": MetricDefinition(
        name="gpu_memory_mb",
        category=MetricCategory.OFFLINE_MEMORY,
        unit="MB",
        description="GPU memory allocated during inference",
        higher_is_better=False,
        typical_range=(50.0, 32000.0),
    ),
    # Timing metrics
    "load_time_ms": MetricDefinition(
        name="load_time_ms",
        category=MetricCategory.OFFLINE_LATENCY,
        unit="ms",
        description="Model loading time",
        higher_is_better=False,
        typical_range=(10.0, 30000.0),
    ),
    "compile_time_ms": MetricDefinition(
        name="compile_time_ms",
        category=MetricCategory.OFFLINE_LATENCY,
        unit="ms",
        description="Model compilation time (e.g., TensorRT)",
        higher_is_better=False,
        typical_range=(100.0, 300000.0),
    ),
    "cold_start_ms": MetricDefinition(
        name="cold_start_ms",
        category=MetricCategory.OFFLINE_LATENCY,
        unit="ms",
        description="First inference latency including initialization",
        higher_is_better=False,
        typical_range=(100.0, 60000.0),
    ),
    # Power metrics
    "power_w": MetricDefinition(
        name="power_w",
        category=MetricCategory.OFFLINE_POWER,
        unit="W",
        description="Average power consumption during inference",
        higher_is_better=False,
        typical_range=(1.0, 300.0),
    ),
    "energy_mj_per_inference": MetricDefinition(
        name="energy_mj_per_inference",
        category=MetricCategory.OFFLINE_POWER,
        unit="mJ",
        description="Energy per inference",
        higher_is_better=False,
        typical_range=(1.0, 10000.0),
    ),
}


# ---------------------------------------------------------------------------
# Online Evaluation Metrics (TrialResult owns these)
# ---------------------------------------------------------------------------

ONLINE_METRICS: Dict[str, MetricDefinition] = {
    # Object Detection
    "mAP": MetricDefinition(
        name="mAP",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Mean Average Precision (IoU 0.5:0.95)",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    "mAP50": MetricDefinition(
        name="mAP50",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Mean Average Precision at IoU 0.5",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    "precision": MetricDefinition(
        name="precision",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Detection precision",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    "recall": MetricDefinition(
        name="recall",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Detection recall",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    # Classification
    "accuracy": MetricDefinition(
        name="accuracy",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Classification accuracy",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    "accuracy_top1": MetricDefinition(
        name="accuracy_top1",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Top-1 classification accuracy",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    "accuracy_top5": MetricDefinition(
        name="accuracy_top5",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Top-5 classification accuracy",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    # Segmentation
    "iou": MetricDefinition(
        name="iou",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Intersection over Union",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    "dice": MetricDefinition(
        name="dice",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Dice coefficient",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    # ASR
    "wer": MetricDefinition(
        name="wer",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Word Error Rate",
        higher_is_better=False,
        typical_range=(0.0, 1.0),
    ),
    "cer": MetricDefinition(
        name="cer",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Character Error Rate",
        higher_is_better=False,
        typical_range=(0.0, 1.0),
    ),
    # Text
    "f1": MetricDefinition(
        name="f1",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="F1 score",
        higher_is_better=True,
        typical_range=(0.0, 1.0),
    ),
    "perplexity": MetricDefinition(
        name="perplexity",
        category=MetricCategory.ONLINE_ACCURACY,
        unit="",
        description="Language model perplexity",
        higher_is_better=False,
        typical_range=(1.0, 1000.0),
    ),
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_offline_metric_names() -> Set[str]:
    """Get all offline metric names."""
    return set(OFFLINE_METRICS.keys())


def get_online_metric_names() -> Set[str]:
    """Get all online metric names."""
    return set(ONLINE_METRICS.keys())


def is_offline_metric(name: str) -> bool:
    """Check if a metric belongs to offline profiling."""
    return name in OFFLINE_METRICS


def is_online_metric(name: str) -> bool:
    """Check if a metric belongs to online evaluation."""
    return name in ONLINE_METRICS


def get_metric_definition(name: str) -> MetricDefinition:
    """Get metric definition by name."""
    if name in OFFLINE_METRICS:
        return OFFLINE_METRICS[name]
    if name in ONLINE_METRICS:
        return ONLINE_METRICS[name]
    raise KeyError(f"Unknown metric: {name}")


def get_metrics_by_category(category: MetricCategory) -> List[MetricDefinition]:
    """Get all metrics in a category."""
    all_metrics = {**OFFLINE_METRICS, **ONLINE_METRICS}
    return [m for m in all_metrics.values() if m.category == category]
