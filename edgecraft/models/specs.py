"""Core specifications for the EdgeCraft model zoo.

This module defines the foundational data structures for the code-centric,
family-native script generation architecture. Key design principles:

1. CODE-CENTRIC: EdgeCraft does NOT execute training internally. Instead, it
   generates train.py/infer.py scripts that run in workspace subprocesses.

2. FAMILY-NATIVE SCRIPT GENERATION: Each family plugin generates scripts
   using its native ecosystem (ultralytics, timm, transformers, etc.),
   but the execution layer remains unified.

3. UNIFIED REGISTRY: All models use namespaced IDs (e.g., 'ultralytics:yolo11n')
   to avoid collisions across families.

Architecture layers:
- ModelSpec: Model metadata (upgraded from ModelCard)
- FamilyPlugin: Family-level plugin for model registration and template generation
- TemplateProvider: Generates train.py/infer.py based on context
- ProfileStore: Stores offline profiling results
- DeviceSpec/RuntimeSpec: Device and runtime capability descriptions
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

from pydantic import BaseModel, Field, computed_field

from edgecraft.core.modality import Modality, TaskType


# ---------------------------------------------------------------------------
# Enums and Literals
# ---------------------------------------------------------------------------

class QuantMode(str, Enum):
    """Quantization modes supported by EdgeCraft."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class ExportFormat(str, Enum):
    """Model export formats."""
    PT = "pt"
    ONNX = "onnx"
    ENGINE = "engine"  # TensorRT
    TFLITE = "tflite"
    EXECUTORCH = "executorch"
    NCNN = "ncnn"
    MNN = "mnn"


class RuntimeId(str, Enum):
    """Supported inference runtimes."""
    PYTORCH = "pytorch"
    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"
    TFLITE = "tflite"
    EXECUTORCH = "executorch"
    NCNN = "ncnn"
    MNN = "mnn"


class DeviceClass(str, Enum):
    """Device classification for template selection."""
    JETSON = "jetson"
    RASPBERRY_PI = "raspberry_pi"
    ANDROID = "android"
    IOS = "ios"
    X86_GPU = "x86_gpu"
    X86_CPU = "x86_cpu"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# ModelSpec: Upgraded from ModelCard
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """Complete model specification for the EdgeCraft model zoo.

    This replaces the simpler ModelCard with a more comprehensive spec
    that includes runtime compatibility, benchmark signatures, and more.
    """

    # Identity (namespaced to avoid collisions)
    name: str = Field(..., description="Short model name, e.g., 'yolo11n'")
    family_id: str = Field(..., description="Family identifier, e.g., 'ultralytics'")

    @computed_field
    @property
    def model_id(self) -> str:
        """Fully qualified model ID: 'family_id:name'."""
        return f"{self.family_id}:{self.name}"

    # Classification
    modality: Modality
    supported_tasks: List[TaskType] = Field(default_factory=list)

    # Model characteristics
    params_m: float = Field(0.0, description="Millions of parameters")
    default_input_shape: List[int] = Field(
        default_factory=lambda: [3, 640, 640],
        description="Default input shape [C, H, W] or [C, T] for audio"
    )

    # Training defaults
    default_hyperparams: Dict[str, Any] = Field(
        default_factory=lambda: {
            "epochs": 50,
            "batch_size": 16,
            "lr": 0.01,
            "imgsz": 640,
        }
    )

    # Dependencies
    pip_packages: List[str] = Field(
        default_factory=list,
        description="Required pip packages for this model"
    )

    # Export and runtime compatibility
    supported_export_formats: List[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.ONNX, ExportFormat.PT]
    )
    supported_runtimes: List[RuntimeId] = Field(
        default_factory=lambda: [RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME]
    )

    # Edge deployment
    edge_compatible: bool = Field(
        True,
        description="Whether model is suitable for edge devices"
    )
    recommended_devices: List[DeviceClass] = Field(
        default_factory=lambda: [DeviceClass.JETSON, DeviceClass.X86_GPU]
    )

    # Benchmark signature for profiling
    benchmark_input_signature: Dict[str, Any] = Field(
        default_factory=lambda: {
            "imgsz": [320, 416, 512, 640],
            "batch_size": [1],
        },
        description="Input variations for systematic profiling"
    )

    # Metadata
    source_url: Optional[str] = Field(
        None,
        description="URL to model source (GitHub, HuggingFace, etc.)"
    )
    paper_url: Optional[str] = Field(None, description="URL to related paper")
    pretrained_weights: Optional[str] = Field(
        None,
        description="Default pretrained checkpoint identifier"
    )

    # Search space hints for LLM
    neighbor_models: List[str] = Field(
        default_factory=list,
        description="Similar models for search exploration (by name)"
    )
    lighter_alternative: Optional[str] = Field(
        None,
        description="Lighter model in same family for downsizing"
    )
    heavier_alternative: Optional[str] = Field(
        None,
        description="Heavier model in same family for upsizing"
    )


# ---------------------------------------------------------------------------
# DeviceSpec: Device capability description
# ---------------------------------------------------------------------------

class DeviceSpec(BaseModel):
    """Specification for a target edge device."""

    device_id: str = Field(..., description="Unique device identifier")
    device_class: DeviceClass
    display_name: str = Field(..., description="Human-readable name")

    # Hardware capabilities
    has_gpu: bool = False
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None  # e.g., "8.7" for Orin

    # Memory
    memory_gb: float = Field(4.0, description="Total RAM in GB")
    gpu_memory_gb: Optional[float] = None

    # Supported runtimes
    supported_runtimes: List[RuntimeId] = Field(
        default_factory=lambda: [RuntimeId.ONNXRUNTIME]
    )
    preferred_runtime: RuntimeId = RuntimeId.ONNXRUNTIME

    # Docker/container support
    docker_image: Optional[str] = None
    supports_docker: bool = True

    # Profiling scale factors (relative to the model-registry reference profile)
    latency_scale: float = Field(
        1.0,
        description="Latency multiplier relative to the reference profile"
    )

    # Connection
    connection_type: Literal["ssh", "adb", "http", "mqtt"] = "ssh"


# ---------------------------------------------------------------------------
# RuntimeSpec: Runtime capability description
# ---------------------------------------------------------------------------

class RuntimeSpec(BaseModel):
    """Specification for an inference runtime."""

    runtime_id: RuntimeId
    display_name: str

    # Supported formats
    supported_formats: List[ExportFormat] = Field(default_factory=list)

    # Quantization support
    supported_quant_modes: List[QuantMode] = Field(
        default_factory=lambda: [QuantMode.FP32, QuantMode.FP16]
    )

    # Device compatibility
    supported_device_classes: List[DeviceClass] = Field(default_factory=list)

    # Dependencies
    pip_packages: List[str] = Field(default_factory=list)
    system_dependencies: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TemplateContext: Context for script generation
# ---------------------------------------------------------------------------

class TemplateContext(BaseModel):
    """Context passed to TemplateProvider for script generation."""

    model_spec: ModelSpec
    task_type: TaskType

    # Execution context
    runtime_id: RuntimeId = RuntimeId.PYTORCH
    device_class: DeviceClass = DeviceClass.JETSON
    export_format: ExportFormat = ExportFormat.ONNX
    quant_mode: QuantMode = QuantMode.FP16

    # Dataset
    dataset_format: str = "yolo"  # yolo, imagefolder, coco, custom
    dataset_path: str = "config/data.yaml"

    # Hyperparameters (override defaults)
    hyperparams: Dict[str, Any] = Field(default_factory=dict)

    # Execution mode
    execution_mode: Literal["train", "eval", "export", "benchmark"] = "train"


# ---------------------------------------------------------------------------
# FamilyPlugin Protocol
# ---------------------------------------------------------------------------

class FamilyPlugin(Protocol):
    """Protocol for family-level plugins.

    Each family (ultralytics, timm, transformers, etc.) implements this
    protocol to register models and generate family-native scripts.

    Key principle: Plugins generate CODE, they do NOT execute training.
    """

    @property
    def family_id(self) -> str:
        """Unique family identifier."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable family name."""
        ...

    @property
    def modalities(self) -> List[Modality]:
        """Supported modalities."""
        ...

    def get_model_specs(self) -> List[ModelSpec]:
        """Return all model specs provided by this family."""
        ...

    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py content for the given context."""
        ...

    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py content for the given context."""
        ...

    def render_export_script(self, context: TemplateContext) -> str:
        """Generate export.py content for the given context."""
        ...

    def get_search_neighbors(
        self,
        model_name: str,
        direction: Literal["lighter", "heavier", "similar"]
    ) -> List[str]:
        """Get neighboring models for search exploration."""
        ...


# ---------------------------------------------------------------------------
# BaseFamilyPlugin: Abstract base class for family plugins
# ---------------------------------------------------------------------------

class BaseFamilyPlugin(ABC):
    """Abstract base class for family plugins.

    Provides default implementations and enforces the code-centric architecture.
    """

    @property
    @abstractmethod
    def family_id(self) -> str:
        """Unique family identifier."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable family name."""
        pass

    @property
    @abstractmethod
    def modalities(self) -> List[Modality]:
        """Supported modalities."""
        pass

    @abstractmethod
    def get_model_specs(self) -> List[ModelSpec]:
        """Return all model specs provided by this family."""
        pass

    @abstractmethod
    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py content for the given context."""
        pass

    @abstractmethod
    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py content for the given context."""
        pass

    def render_export_script(self, context: TemplateContext) -> str:
        """Generate export.py content. Default: empty (handled in infer)."""
        return ""

    def get_search_neighbors(
        self,
        model_name: str,
        direction: Literal["lighter", "heavier", "similar"]
    ) -> List[str]:
        """Get neighboring models for search exploration.

        Default implementation uses ModelSpec.neighbor_models and alternatives.
        """
        specs = {s.name: s for s in self.get_model_specs()}
        spec = specs.get(model_name)
        if not spec:
            return []

        if direction == "lighter" and spec.lighter_alternative:
            return [spec.lighter_alternative]
        if direction == "heavier" and spec.heavier_alternative:
            return [spec.heavier_alternative]
        if direction == "similar":
            return spec.neighbor_models

        return []

    def get_pip_requirements(self) -> List[str]:
        """Get pip packages required by this family."""
        packages = set()
        for spec in self.get_model_specs():
            packages.update(spec.pip_packages)
        return sorted(packages)

    def get_supported_profile_runtimes(self, model_spec: ModelSpec) -> List[RuntimeId]:
        """Return runtimes this plugin can benchmark for a model.

        Default policy keeps only edge inference runtimes.
        Families can override this when they have richer benchmark support.
        """
        edge_runtimes = {RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT}
        return [r for r in model_spec.supported_runtimes if r in edge_runtimes]

    def render_profile_benchmark_script(
        self,
        model_spec: ModelSpec,
        runtime_id: RuntimeId,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        """Return benchmark.py content for offline profiling.

        Families should override this when runtime-specific benchmark logic
        is needed. Empty string means "no provider for this runtime".
        """
        _ = (model_spec, runtime_id, imgsz, warmup, iterations)
        return ""
