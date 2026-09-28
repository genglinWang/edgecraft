"""Model Registry: single source of truth for model metadata in EdgeCraft.

This module provides:
- ModelCard: Pydantic dataclass for model metadata
- ModelRegistry: Central registry for all supported models
- Latency/memory estimation tables (consolidated from modality_handler and surrogate)
- Code templates per (family, task_type) for LLM few-shot examples
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from loguru import logger

from edgecraft.core.modality import Modality, TaskType


class ModelCard(BaseModel):
    """Metadata for a single model variant."""

    name: str = Field(..., description="Model identifier, e.g. 'yolo11n', 'efficientnet_b0'")
    family: str = Field(..., description="Model family: 'ultralytics', 'timm', 'huggingface'")
    modality: Modality
    supported_tasks: List[TaskType] = Field(default_factory=list)
    params_m: float = Field(0.0, description="Millions of parameters")
    default_input_shape: List[int] = Field(default_factory=lambda: [3, 640, 640])
    default_hyperparams: Dict[str, Any] = Field(default_factory=dict)
    pip_packages: List[str] = Field(default_factory=list)
    supported_export_formats: List[str] = Field(default_factory=lambda: ["onnx"])
    edge_compatible: bool = True


# ---------------------------------------------------------------------------
# Latency/Memory Estimation Tables (consolidated from modality_handler.py and surrogate.py)
# ---------------------------------------------------------------------------

# (model_name, imgsz, quant_mode) -> latency_ms on the reference Jetson profile
_LATENCY_TABLE: Dict[Tuple[str, int, str], float] = {
    ("yolo11n", 320, "fp32"): 12.0,
    ("yolo11n", 416, "fp32"): 18.0,
    ("yolo11n", 640, "fp32"): 38.0,
    ("yolo11n", 320, "fp16"): 8.0,
    ("yolo11n", 416, "fp16"): 12.0,
    ("yolo11n", 640, "fp16"): 22.0,
    ("yolo11n", 320, "int8"): 6.0,
    ("yolo11n", 416, "int8"): 9.0,
    ("yolo11n", 640, "int8"): 16.0,
    ("yolo11s", 320, "fp32"): 18.0,
    ("yolo11s", 416, "fp32"): 30.0,
    ("yolo11s", 640, "fp32"): 62.0,
    ("yolo11s", 320, "fp16"): 12.0,
    ("yolo11s", 416, "fp16"): 20.0,
    ("yolo11s", 640, "fp16"): 38.0,
    ("yolo11s", 320, "int8"): 9.0,
    ("yolo11s", 416, "int8"): 14.0,
    ("yolo11s", 640, "int8"): 27.0,
    ("yolo11m", 320, "fp16"): 22.0,
    ("yolo11m", 416, "fp16"): 36.0,
    ("yolo11m", 640, "fp16"): 72.0,
    ("yolo11m", 320, "int8"): 16.0,
    ("yolo11m", 416, "int8"): 25.0,
    ("yolo11m", 640, "int8"): 48.0,
    ("yolo11l", 416, "fp16"): 55.0,
    ("yolo11l", 640, "fp16"): 110.0,
    ("yolo11l", 416, "int8"): 38.0,
    ("yolo11l", 640, "int8"): 72.0,
    ("mobilenetv3_small_100", 224, "fp16"): 5.0,
    ("efficientnet_b0", 224, "fp16"): 8.0,
    ("efficientnet_b0", 224, "int8"): 5.0,
    ("resnet18", 224, "fp16"): 10.0,
}

# Device scaling factors relative to the reference Jetson profile
_DEVICE_SCALE: Dict[str, float] = {
    "jetson_orin_agx": 0.5,
    "jetson_xavier_nx": 1.6,
    "jetson_xavier": 1.3,
    "jetson_tx2": 3.0,
    "raspberry_pi_5": 8.0,
}

# Memory usage (MB) at fp16 per model
_MEMORY_TABLE: Dict[str, float] = {
    "yolo11n": 350,
    "yolo11s": 500,
    "yolo11m": 850,
    "yolo11l": 1400,
    "yolo11x": 2200,
    "mobilenetv3_small_100": 120,
    "efficientnet_b0": 180,
    "resnet18": 250,
    "shufflenet_v2_x0_5": 100,
}

# Quantization memory multipliers
_QUANT_MEMORY_FACTOR: Dict[str, float] = {
    "fp32": 2.0,
    "fp16": 1.0,
    "int8": 0.55,
}

# Quantization latency multipliers (for fallback estimation)
_QUANT_LATENCY_FACTOR: Dict[str, float] = {
    "fp32": 1.0,
    "fp16": 0.55,
    "int8": 0.38,
}

# Base latency (ms) at fp32 imgsz=640 for fallback estimation
_MODEL_BASE_LATENCY: Dict[str, float] = {
    "yolo11n": 38.0,
    "yolo11s": 62.0,
    "yolo11m": 100.0,
    "yolo11l": 160.0,
    "yolo11x": 250.0,
}


# ---------------------------------------------------------------------------
# Code Templates per (family, task_type)
# ---------------------------------------------------------------------------

_TRAIN_TEMPLATES: Dict[Tuple[str, str], str] = {}
_INFER_TEMPLATES: Dict[Tuple[str, str], str] = {}


def register_train_template(family: str, task_type: str, template: str) -> None:
    """Register a training code template for (family, task_type)."""
    _TRAIN_TEMPLATES[(family, task_type)] = template


def register_infer_template(family: str, task_type: str, template: str) -> None:
    """Register an inference code template for (family, task_type)."""
    _INFER_TEMPLATES[(family, task_type)] = template


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Central registry for all supported models."""

    _cards: Dict[str, ModelCard] = {}

    @classmethod
    def register(cls, card: ModelCard) -> None:
        """Register a model card."""
        cls._cards[card.name] = card
        logger.debug(f"ModelRegistry: registered {card.name} ({card.family})")

    @classmethod
    def get(cls, name: str) -> Optional[ModelCard]:
        """Get a model card by name."""
        return cls._cards.get(name)

    @classmethod
    def query(
        cls,
        modality: Optional[Modality] = None,
        task_type: Optional[TaskType] = None,
        family: Optional[str] = None,
        max_params_m: Optional[float] = None,
        edge_only: bool = False,
    ) -> List[ModelCard]:
        """Query models matching the given criteria."""
        results = []
        for card in cls._cards.values():
            if modality and card.modality != modality:
                continue
            if task_type and task_type not in card.supported_tasks:
                continue
            if family and card.family != family:
                continue
            if max_params_m and card.params_m > max_params_m:
                continue
            if edge_only and not card.edge_compatible:
                continue
            results.append(card)
        return sorted(results, key=lambda c: c.params_m)

    @classmethod
    def list_names(cls, modality: Optional[Modality] = None) -> List[str]:
        """List all registered model names."""
        if modality:
            return [c.name for c in cls._cards.values() if c.modality == modality]
        return list(cls._cards.keys())

    @classmethod
    def search_space_description(cls, modality: Modality, task_type: TaskType) -> str:
        """Auto-generate search space description for LLM prompts."""
        cards = cls.query(modality=modality, task_type=task_type)
        if not cards:
            return "No models registered for this modality/task combination."

        lines = [f"Available models for {modality.value}/{task_type.value}:"]

        # Group by family
        families: Dict[str, List[ModelCard]] = {}
        for card in cards:
            families.setdefault(card.family, []).append(card)

        for family, family_cards in families.items():
            names = [c.name for c in family_cards]
            lines.append(f"  {family}: {', '.join(names)}")

        # Add common config axes
        lines.append("")
        lines.append("Configuration axes:")
        lines.append("  quant_mode: [fp32, fp16, int8]")
        lines.append("  export_format: [onnx, engine, pt]")

        # Add modality-specific hints
        if modality == Modality.VISION:
            lines.append("  imgsz: [320, 416, 512, 640] (in train_code)")
            lines.append("  epochs, batch_size, lr, optimizer (in train_code)")

        return "\n".join(lines)

    @classmethod
    def get_train_template(cls, family: str, task_type: TaskType) -> str:
        """Get training code template for (family, task_type)."""
        key = (family, task_type.value)
        if key in _TRAIN_TEMPLATES:
            return _TRAIN_TEMPLATES[key]
        # Fallback to family-only key
        for k, v in _TRAIN_TEMPLATES.items():
            if k[0] == family:
                return v
        return ""

    @classmethod
    def get_infer_template(cls, family: str, task_type: TaskType) -> str:
        """Get inference code template for (family, task_type)."""
        key = (family, task_type.value)
        if key in _INFER_TEMPLATES:
            return _INFER_TEMPLATES[key]
        # Fallback to family-only key
        for k, v in _INFER_TEMPLATES.items():
            if k[0] == family:
                return v
        return ""

    @classmethod
    def estimate_latency(
        cls,
        model_name: str,
        device_id: str,
        quant_mode: str,
        imgsz: int = 640,
    ) -> Tuple[float, float]:
        """Estimate inference latency.

        Returns:
            (latency_ms, confidence) where confidence is in [0, 1].
        """
        # Try exact lookup
        key = (model_name, imgsz, quant_mode)
        base_latency = _LATENCY_TABLE.get(key)

        if base_latency is not None:
            confidence = 0.75
        else:
            # Fallback: heuristic estimation
            quant_factor = _QUANT_LATENCY_FACTOR.get(quant_mode, 1.0)
            family_base = _MODEL_BASE_LATENCY.get(model_name, 80.0)
            base_latency = family_base * (imgsz / 640) ** 2 * quant_factor
            confidence = 0.35

        # Scale by device
        scale = _DEVICE_SCALE.get(device_id, 1.0)
        return base_latency * scale, confidence

    @classmethod
    def estimate_memory(cls, model_name: str, quant_mode: str) -> float:
        """Estimate memory usage in MB."""
        base = _MEMORY_TABLE.get(model_name, 600.0)
        factor = _QUANT_MEMORY_FACTOR.get(quant_mode, 1.0)
        return base * factor

    @classmethod
    def get_default_hyperparams(cls, model_name: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model."""
        card = cls.get(model_name)
        if card:
            return dict(card.default_hyperparams)
        return {"epochs": 50, "batch_size": 16, "lr": 0.01}
