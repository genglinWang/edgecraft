"""Failure taxonomy and search hints for EdgeCraft trials.

The taxonomy separates candidate-code issues from infrastructure/runtime
failures so the LLM debugger only edits scripts when that is actually useful.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Tuple


class FailureCategory(str, Enum):
    TRAIN_CODE_ERROR = "train_code_error"
    DATASET_ADAPTER_ERROR = "dataset_adapter_error"
    LOADER_SMOKE_FAILED = "loader_smoke_failed"
    EXPORT_OPERATOR_UNSUPPORTED = "export_operator_unsupported"
    EXPORT_VERSION_MISMATCH = "export_version_mismatch"
    EDGE_RUNTIME_DEPENDENCY = "edge_runtime_dependency"
    EDGE_PROVIDER_MISSING = "edge_provider_missing"
    EDGE_STAGING_PERMISSION = "edge_staging_permission"
    EDGE_PAYLOAD_PACKAGING = "edge_payload_packaging"
    EDGE_DISK_FULL = "edge_disk_full"
    EDGE_COLLECT_RESULTS = "edge_collect_results"
    METRIC_CONTRACT_ERROR = "metric_contract_error"
    DATASET_LABEL_MISSING = "dataset_label_missing"
    QUALITY_BELOW_TARGET = "quality_below_target"
    LATENCY_ABOVE_TARGET = "latency_above_target"
    UNKNOWN = "unknown"


_PATTERNS: list[tuple[FailureCategory, str]] = [
    (FailureCategory.EDGE_DISK_FULL, r"no space left on device|disk full"),
    (
        FailureCategory.EDGE_RUNTIME_DEPENDENCY,
        r"onnxruntime-gpu|no matching distribution found|could not find a version that satisfies",
    ),
    (
        FailureCategory.EDGE_PROVIDER_MISSING,
        r"executionprovider.*not available|provider.*not available|cudaexecutionprovider.*missing",
    ),
    (
        FailureCategory.EDGE_PAYLOAD_PACKAGING,
        r"generated tarball does not contain top-level run\.sh|missing top-level run\.sh|payload packaging failed|staged job missing top-level run\.sh",
    ),
    (
        FailureCategory.EDGE_STAGING_PERMISSION,
        r"permission denied|cannot remove .*bundle|failed to cleanup|chown|chmod",
    ),
    (
        FailureCategory.EDGE_COLLECT_RESULTS,
        r"collect_results|failed to collect|valid result tar|tar validation failed|outbox|collected_results",
    ),
    (
        FailureCategory.EXPORT_OPERATOR_UNSUPPORTED,
        r"unsupported operator|operator.*not supported|no adapter from version|onnx.*opset",
    ),
    (
        FailureCategory.EXPORT_VERSION_MISMATCH,
        r"opset|version converter|onnx.*version|tensorrt.*version",
    ),
    (
        FailureCategory.DATASET_LABEL_MISSING,
        r"supervised .* needs .*label_column.*label_files|label_column.*external label_files|blockid-level label file|supervised labels.*missing|label files.*missing",
    ),
    (
        FailureCategory.LOADER_SMOKE_FAILED,
        r"loader smoke failed|loader_smoke_failed|loader\.py --smoke",
    ),
    (
        FailureCategory.DATASET_ADAPTER_ERROR,
        r"could not load dataset|dataset_contract|metadata\.csv|audiofolder|audio_dataset_empty|parquet|schema",
    ),
    (
        FailureCategory.TRAIN_CODE_ERROR,
        r"traceback|attributeerror|importerror|typeerror|filenotfounderror|unexpected keyword argument|not a valid .* argument",
    ),
    (
        FailureCategory.METRIC_CONTRACT_ERROR,
        r"json contract violation|last stdout line is not json|metric.*missing|keyerror.*metric",
    ),
]


_NON_REPAIRABLE = {
    FailureCategory.EDGE_RUNTIME_DEPENDENCY,
    FailureCategory.EDGE_PROVIDER_MISSING,
    FailureCategory.EDGE_STAGING_PERMISSION,
    FailureCategory.EDGE_PAYLOAD_PACKAGING,
    FailureCategory.EDGE_DISK_FULL,
    FailureCategory.EDGE_COLLECT_RESULTS,
    FailureCategory.DATASET_LABEL_MISSING,
}


_HINTS = {
    FailureCategory.EDGE_RUNTIME_DEPENDENCY: "Use a prebuilt edge runtime, inspect provider availability, or fall back to another artifact/runtime.",
    FailureCategory.EDGE_PROVIDER_MISSING: "Avoid unavailable providers for this device/image and benchmark with the next available runtime.",
    FailureCategory.EDGE_STAGING_PERMISSION: "Fix EdgeRunner payload ownership/cleanup; do not patch candidate infer.py.",
    FailureCategory.EDGE_PAYLOAD_PACKAGING: "Inspect EdgeRunner job bundle creation/tar packaging; do not patch candidate train.py or infer.py.",
    FailureCategory.EDGE_DISK_FULL: "Clean edge runner work/outbox/tmp or reduce staged dataset size.",
    FailureCategory.EDGE_COLLECT_RESULTS: "Retry/repair result collection and classify separately from model code.",
    FailureCategory.EXPORT_OPERATOR_UNSUPPORTED: "Try another export target, lower opset, static shape, or a model family with simpler operators.",
    FailureCategory.EXPORT_VERSION_MISMATCH: "Align ONNX/TensorRT/opset versions or fall back to PyTorch/TorchScript.",
    FailureCategory.DATASET_ADAPTER_ERROR: "Improve the generic dataset adapter/schema inference before changing model code.",
    FailureCategory.LOADER_SMOKE_FAILED: "Repair loader.py so a small batch can be read before training.",
    FailureCategory.METRIC_CONTRACT_ERROR: "Normalize the script JSON contract and canonical metric keys.",
    FailureCategory.DATASET_LABEL_MISSING: "Provide a supervised label file such as anomaly_label.csv or anomaly_label.txt with BlockId and Label columns; do not mutate model code.",
    FailureCategory.TRAIN_CODE_ERROR: "Repair generated train/infer code using stable public APIs and retrieved knowledge.",
    FailureCategory.QUALITY_BELOW_TARGET: "Explore better model capacity, training duration, augmentation, or pretrained weights.",
    FailureCategory.LATENCY_ABOVE_TARGET: "Explore smaller models, lower input size, quantization, or optimized runtime.",
    FailureCategory.UNKNOWN: "Inspect logs and add a taxonomy rule if this failure recurs.",
}


def classify_failure(error: str = "", stderr: str = "", stdout: str = "") -> FailureCategory:
    """Classify a failure blob into a stable category."""
    blob = f"{error}\n{stderr}\n{stdout}".lower()
    for category, pattern in _PATTERNS:
        if re.search(pattern, blob):
            return category
    return FailureCategory.UNKNOWN


def is_repairable(category: FailureCategory | str) -> bool:
    """Return whether the LLM debugger should attempt script repair."""
    try:
        cat = FailureCategory(category)
    except Exception:
        cat = FailureCategory.UNKNOWN
    if cat in _NON_REPAIRABLE:
        return False
    return cat in {
        FailureCategory.TRAIN_CODE_ERROR,
        FailureCategory.DATASET_ADAPTER_ERROR,
        FailureCategory.LOADER_SMOKE_FAILED,
        FailureCategory.METRIC_CONTRACT_ERROR,
        FailureCategory.EXPORT_OPERATOR_UNSUPPORTED,
        FailureCategory.EXPORT_VERSION_MISMATCH,
    }


def next_search_hint(category: FailureCategory | str) -> str:
    """Human-readable hint for ProposalGenerator/CBR summaries."""
    try:
        cat = FailureCategory(category)
    except Exception:
        cat = FailureCategory.UNKNOWN
    return _HINTS.get(cat, _HINTS[FailureCategory.UNKNOWN])


def classify_repairability(error: str = "", stderr: str = "", stdout: str = "") -> Tuple[bool, str, str]:
    """Return (repairable, category, hint)."""
    category = classify_failure(error, stderr, stdout)
    return is_repairable(category), category.value, next_search_hint(category)
