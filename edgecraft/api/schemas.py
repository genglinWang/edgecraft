"""API request/response schemas."""
from typing import Annotated, Any, Dict, List, Optional

from pydantic import AfterValidator, BaseModel, Field

from edgecraft.utils.network import validate_ssh_target
from edgecraft.core.task import Constraint


SshTarget = Annotated[str, AfterValidator(validate_ssh_target)]


class CreateTaskRequest(BaseModel):
    """Request to create an asynchronous synthesis task."""

    intent: str = Field(..., description="Natural language synthesis intent")
    dataset_path: Optional[str] = Field(
        None,
        description="Local-operator dataset path; forbidden in authenticated mode",
    )
    dataset_handle: Optional[str] = Field(
        None,
        description="Tenant-owned dataset handle for authenticated mode",
    )
    device_id: str = Field("jetson_orin_agx", description="Target edge device id")
    device_ip: SshTarget = Field(..., description="Target edge device SSH endpoint (user@host)")
    ssh_key_path: Optional[str] = Field(
        None,
        min_length=1,
        description="Local-operator SSH key path; forbidden in authenticated mode",
    )
    credential_handle: Optional[str] = Field(
        None,
        description="Server-managed tenant credential handle for authenticated mode",
    )
    docker_image: Optional[str] = Field(
        None,
        description="Docker image for containerized edge hosts; omit for native-runtime devices",
    )
    iterations: int = Field(24, ge=1, description="Maximum admitted search trials")
    branching_factor: Optional[int] = Field(None, ge=1, description="Children per tree expansion")
    run_id: Optional[str] = Field(None, description="Optional run id to resume or label a tenant tree")
    tenant_id: str = Field("default", description="Tenant id for scheduler fairness")
    profile: str = Field(
        "paper",
        pattern="^(paper|paper-audit|offline)$",
        description="Service-wide mechanism profile expected by this request",
    )
    constraints: List[Constraint] = Field(
        default_factory=list,
        description="Optional structured SLOs that override intent-parsed constraints",
    )
    force_real_edgebench: bool = Field(
        False,
        description="Reserved compatibility field for benchmark callers",
    )


class TaskResponse(BaseModel):
    """Task state response used by /tasks APIs."""

    task_id: str
    status: str
    created_at: str
    updated_at: str
    tenant_id: str = "default"
    run_id: Optional[str] = None
    trial_bank_path: Optional[str] = None
    request: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    artifact_manifest: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SynthesizeRequest(BaseModel):
    """Request to synthesize a model."""

    intent: str = Field(..., description="Natural language description of the task")
    dataset_path: Optional[str] = Field(None, description="Path to dataset")
    target_device: str = Field("jetson_orin_agx", description="Target edge device")
    max_iterations: int = Field(24, description="Maximum admitted search trials")


class SynthesizeResponse(BaseModel):
    """Response from synthesis task."""

    task_id: str
    status: str
    run_id: Optional[str] = None
    best_trial: Optional[Dict[str, Any]] = None
    selection_kind: str = "none"
    constraints_satisfied: bool = False
    remaining_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    all_trials: List[str] = Field(default_factory=list)
    iterations: int = 0
    modality: Optional[str] = None
    task_type: Optional[str] = None
    dataset_info: Optional[Dict[str, Any]] = None
    reflector_diagnosis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskStatus(BaseModel):
    """Task status response."""

    task_id: str
    status: str
    iteration: int = 0
    error: Optional[str] = None


class DeviceInfo(BaseModel):
    """Edge device information."""

    device_id: str
    name: str
    ip_address: Optional[str] = None
    status: str = "unknown"
    specs: Dict[str, Any] = Field(default_factory=dict)


class DeployRequest(BaseModel):
    """Request to deploy model to device."""

    artifact_path: str
    device_id: str
    device_ip: SshTarget
    ssh_key_path: str = Field(..., min_length=1)
    docker_image: Optional[str] = None


class DeployResponse(BaseModel):
    """Response from deployment."""

    status: str
    device_id: str
    job_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class BenchmarkRequest(BaseModel):
    """Request to run benchmark."""

    model_path: str
    device_id: str
    device_ip: SshTarget
    ssh_key_path: str = Field(..., min_length=1)
    docker_image: Optional[str] = None
    timeout: Optional[int] = None
    num_iterations: int = 100
    warmup: int = 10


class BenchmarkResponse(BaseModel):
    """Benchmark results."""

    status: str
    device_id: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
