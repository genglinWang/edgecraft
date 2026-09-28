from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from .modality import Modality, TaskType


class RuntimeConfig(BaseModel):
    """Hardware runtime capabilities probed during preflight.

    Populated by DeviceAnalyzer / synth_checks preflight.  Stored in
    AgentState and used by ProposalGenerator to constrain the
    SolutionVariant search space (e.g. no 'engine' export when TRT absent).
    """
    available_runtimes: List[str] = Field(
        default_factory=list,
        description="E.g. ['onnxruntime', 'tensorrt', 'torch'].",
    )
    docker_image: str = Field(
        default="unset",
        description="Container image or native runtime label used for EdgeBenchmark on this device.",
    )
    device_arch: str = Field(
        default="aarch64",
        description="CPU architecture: 'aarch64' | 'x86_64' | 'armv7l'.",
    )
    python_version: Optional[str] = Field(
        None,
        description="Python version in the exact edge inference environment.",
    )
    has_gpu: bool = False
    trt_version: Optional[str] = Field(
        None, description="TensorRT version string if available, e.g. '8.6.1'."
    )
    onnxruntime_version: Optional[str] = None
    onnxruntime_providers: List[str] = Field(
        default_factory=list,
        description="Execution providers reported by ONNXRuntime on the edge runtime.",
    )
    torch_version: Optional[str] = None
    l4t_version: Optional[str] = Field(
        None, description="L4T version for Jetson devices, e.g. 'r36.4.0'."
    )
    python_packages: Dict[str, str] = Field(
        default_factory=dict,
        description="Python package availability in the edge runtime: package -> version or unavailable:<reason>.",
    )
    runtime_python_packages: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "Package availability grouped by the exact edge runtime environment, "
            "for devices that use separate native Python environments."
        ),
    )
    cloud_python_packages: Dict[str, str] = Field(
        default_factory=dict,
        description="Python package availability in the cloud/server training environment: package -> version or unavailable:<reason>.",
    )
    edge_artifact_bundle_max_mb: int = Field(
        default=256,
        ge=1,
        description=(
            "Maximum combined size of artifacts and evaluation payload staged to "
            "the edge device for one trial."
        ),
    )

    @property
    def supports_tensorrt(self) -> bool:
        return "tensorrt" in self.available_runtimes

    @property
    def supports_onnxruntime(self) -> bool:
        return "onnxruntime" in self.available_runtimes

    @property
    def supports_tflite(self) -> bool:
        return "tflite" in self.available_runtimes

    def cloud_buildable_formats(self) -> List[str]:
        """Formats the cloud trainer is allowed to produce.

        Cloud always produces ``pt`` and ``onnx``.  Engine is *never* built on
        the cloud because TRT engines are bound to the edge GPU architecture
        and TRT version; engines must be compiled on the target device.
        """
        return ["pt", "onnx"]

    def deploy_target_formats(self) -> List[str]:
        """Formats the edge device can actually load at inference time.

        This is the search space for ``SolutionVariant.export_format``.  When
        ``engine`` appears here, the edge runner will compile ONNX → engine
        on the device before running ``infer.py``.
        """
        formats = ["pt"]  # always available (PyTorch native)
        if self.supports_onnxruntime:
            formats.append("onnx")
        if self.supports_tensorrt:
            formats.append("engine")
        if self.supports_tflite:
            formats.append("tflite")
        return formats

    # Backwards-compatible alias.  Existing callers that ask "what export
    # formats are allowed?" mean "deploy targets" — the cloud is constrained
    # separately via ``cloud_buildable_formats``.
    def allowed_export_formats(self) -> List[str]:
        return self.deploy_target_formats()


class Constraint(BaseModel):
    """Numeric bound from user intent; aligned with USER_SPEC_PARSING_PROMPT JSON."""

    metric: str
    comparison: str = Field("gte", pattern="^(gte|lte|eq)$")
    target: float
    unit: Optional[str] = None
    scope: Optional[str] = None


class Preference(BaseModel):
    """Optimization direction from user intent; pairs with Constraint on UserSpec."""

    metric: str
    direction: str = Field("minimize", pattern="^(minimize|maximize)$")
    unit: Optional[str] = None
    scope: Optional[str] = None


class UserSpec(BaseModel):
    """Structured task spec from LLM parsing of **user intent only** (not device/dataset runtime).

    ``dataset_path`` may be unset in intent-only copies; preflight merges the CLI dataset root
    via ``model_copy(update=...)`` rather than mutating in place.
    """
    description: str = Field(..., description="Task description")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset")
    dataset_path: Optional[str] = Field(None, description="Physical path or location of dataset")
    task_type: Optional[TaskType] = Field(None, description="Type of task (classification, etc.)")
    input_type: Optional[Modality] = Field(None, description="Input modality")
    output_type: Optional[str] = Field(None, description="Expected output format/type")
    constraints: List[Constraint] = Field(default_factory=list)
    preferences: List[Preference] = Field(default_factory=list)
    eval_metrics: List[str] = Field(default_factory=list, description="Specific evaluation metrics requested")


def merge_user_spec_dataset_path(user_spec: UserSpec, dataset_root: Optional[str]) -> UserSpec:
    """Return a copy of ``user_spec`` with ``dataset_path`` set from runtime (e.g. CLI ``-d``)."""
    if dataset_root is None:
        return user_spec.model_copy(deep=True)
    return user_spec.model_copy(update={"dataset_path": dataset_root})
