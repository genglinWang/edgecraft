"""Synth preflight checks: UserSpec parsing, SSH/device probe, dataset validation and analysis.

Orchestrates constraint_parser, network, device_analyzer, and analyzer to run all checks
required before synthesis. Returns a structured result or failure message so CLI/API can
present a concise interface.
"""
import json
import os
import shlex
import time
import importlib
import importlib.metadata
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from pydantic import BaseModel, Field, computed_field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from langchain_core.messages import HumanMessage, SystemMessage

from edgecraft.config.settings import edgecraft_env, settings
from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import RuntimeConfig, UserSpec
from edgecraft.utils.network import (
    ensure_edge_runner_active,
    test_ssh_connection,
    validate_ssh_target,
)
from edgecraft.utils.docker_preflight import run_docker_image_preflight
from edgecraft.utils.analyzer import DatasetAnalyzer
from edgecraft.utils.device_analyzer import run as run_device_analyzer, get_device_info_display_rows
from edgecraft.utils.device_analyzer import DeviceInfo
from edgecraft.utils.constraint_parser import parse_user_spec_from_intent, get_user_spec_display_rows
from edgecraft.utils.llm import create_chat_llm


REQUEST_VERIFICATION_SYSTEM = """You are a verifier for EdgeCraft, an MCaaS edge AI model synthesis service. Your job is to judge whether a user request is suitable for automated model synthesis and deployment to edge devices.

You must check:
1. **Relevance**: Is the request relevant to machine learning or artificial intelligence tasks that can be executed on edge devices (e.g. vision, audio, text, time-series models)?
2. **Adequacy**: Given the user intent, parsed task/dataset/constraint summary, and target device, do we have essential information (e.g. problem description, dataset indication, and target device) to run an AutoML-style synthesis pipeline? Focus on essential requirements only; answer 'yes' even if details are brief. If the dataset has a single labeled split, do not reject solely because official train/validation/test splits are missing; the synthesis loop may create trial-local internal splits and record the limitation as evidence. Reject only when essential data, labels/targets, task meaning, or device information is truly absent.

Answer in this exact format: `yes or no; your reasons` using a semicolon to separate the answer and reasons. If the answer is 'no', include alternative solutions or what is missing."""

REQUEST_VERIFICATION_PROMPT = """Verify the following request for edge AI model synthesis.

**User intent:**
{intent}

**Parsed context (task, dataset, device):**
{context}

Is this request (1) relevant to edge/ML tasks and (2) adequate (we have essential info to proceed)? Answer: `yes or no; your reasons`."""


class IntentPreflight(BaseModel):
    """User intent: verbatim text + LLM-parsed UserSpec (dataset_path not canonical; use dataset.root_path)."""

    raw_user_intent: str
    user_spec: UserSpec


class DatasetPreflight(BaseModel):
    """Dataset root from CLI, resolved config yaml path, and analyzer output."""

    root_path: str
    config_path: str
    info: Dict[str, Any] = Field(default_factory=dict)


class DevicePreflight(BaseModel):
    """Resolved device id, SSH target, hardware probe, and ML runtime capabilities."""

    device_id: str
    device_ip: str
    device_info: DeviceInfo
    runtime_config: RuntimeConfig = Field(
        ...,
        description="Available runtimes / versions probed after Docker image preflight (host or container).",
    )
    captured_at: Optional[str] = Field(
        None,
        description="UTC capture time for reusable device facts; absent on legacy snapshots.",
    )
    source: str = Field(
        "legacy_snapshot",
        description="Evidence source, e.g. live_probe or versioned_device_report.",
    )


def _enum_or_none(enum_cls, value: Any):
    try:
        return enum_cls(value) if value not in (None, "", "unknown") else None
    except ValueError:
        return None


def _dataset_modality_is_strong(info: Dict[str, Any]) -> bool:
    """Return True when dataset evidence should override weak intent parsing."""
    modality = str(info.get("modality") or "").lower()
    fmt = str(info.get("format") or "").lower()
    text = " ".join(
        str(info.get(k) or "")
        for k in ("description", "structure_summary", "exploration_report")
    ).lower()
    if modality in {"structured", "text"}:
        return True
    if modality == "time_series":
        return True
    if modality == "audio" and any(token in text for token in (".wav", "waveform", "audio", "speech", "sound")):
        return True
    if fmt in {"csv", "tsv", "parquet", "jsonl", "arrow", "arff", "ts"}:
        return True
    return False


def merge_user_spec_with_dataset_evidence(
    user_spec: UserSpec,
    dataset_root: Optional[str],
    dataset_info: Optional[Dict[str, Any]],
) -> UserSpec:
    """Merge runtime dataset evidence into UserSpec without dataset-specific glue.

    User intent is valuable for goals and constraints, but dataset modality/task
    should win when the file evidence strongly contradicts a weak/default LLM
    parse.  This prevents tabular CSV tasks from being routed to vision
    ImageFolder/YOLO code merely because the natural-language intent said
    "classify" without naming the modality.
    """
    updates: Dict[str, Any] = {"dataset_path": dataset_root} if dataset_root else {}
    info = dataset_info or {}
    requested_modality = _enum_or_none(Modality, info.get("requested_modality_hint"))
    requested_task = _enum_or_none(TaskType, info.get("requested_task_type_hint"))
    ds_modality = _enum_or_none(Modality, info.get("modality"))
    ds_task = _enum_or_none(TaskType, info.get("task_type"))
    current_modality = requested_modality or user_spec.input_type
    current_task = requested_task or user_spec.task_type
    if requested_modality:
        updates["input_type"] = requested_modality
    if requested_task:
        updates["task_type"] = requested_task

    if ds_modality and not requested_modality and _dataset_modality_is_strong(info):
        weak_or_contradictory = current_modality in {
            None,
            Modality.VISION,
            Modality.MULTIMODAL,
        }
        tabular_or_text = ds_modality in {Modality.STRUCTURED, Modality.TEXT}
        time_series_evidence = ds_modality == Modality.TIME_SERIES and current_modality != Modality.AUDIO
        if weak_or_contradictory or tabular_or_text or time_series_evidence:
            updates["input_type"] = ds_modality

    if ds_task and not requested_task and current_task in {None, TaskType.OBJECT_DETECTION, TaskType.TEXT_GENERATION}:
        updates["task_type"] = ds_task

    return user_spec.model_copy(update=updates)


class SynthCheckResult(BaseModel):
    """Result of synth preflight checks.

    Domain grouping: ``intent``, ``dataset``, ``device``. When success=True, these are set.
    Flat accessors (user_spec merged with dataset root, dataset_info, device, …) are computed.
    """

    success: bool = Field(..., description="Whether all checks passed")
    message: Optional[str] = Field(None, description="Error or warning message when success=False")
    intent: Optional[IntentPreflight] = None
    dataset: Optional[DatasetPreflight] = None
    device_ctx: Optional[DevicePreflight] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def user_spec(self) -> Optional[UserSpec]:
        """UserSpec with ``dataset_path`` merged from CLI root when dataset preflight succeeded."""
        if self.intent is None:
            return None
        root = self.dataset.root_path if self.dataset else None
        info = self.dataset.info if self.dataset else None
        return merge_user_spec_with_dataset_evidence(self.intent.user_spec, root, info)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_info(self) -> Optional[Dict[str, Any]]:
        return self.dataset.info if self.dataset else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_config_path(self) -> Optional[str]:
        return self.dataset.config_path if self.dataset else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def device_info(self) -> Optional[DeviceInfo]:
        return self.device_ctx.device_info if self.device_ctx else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def device(self) -> Optional[str]:
        """device_id for run_agent."""
        return self.device_ctx.device_id if self.device_ctx else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def device_ip(self) -> Optional[str]:
        return self.device_ctx.device_ip if self.device_ctx else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def runtime_config(self) -> Optional[RuntimeConfig]:
        return self.device_ctx.runtime_config if self.device_ctx else None


def _build_intent_analysis_panel(
    user_spec: UserSpec,
    device_info: DeviceInfo,
    dataset_info: Dict[str, Any],
    dataset_config_path: str,
    runtime_config: Optional[RuntimeConfig] = None,
) -> Panel:
    """Build the unified Intent analysis table (dataset + user spec + device) and wrap in a Panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan", width=12)
    table.add_column("Value", style="white")

    # Section: Dataset analysis
    table.add_row("", "[bold]Dataset analysis[/]")
    if dataset_info.get("status") == "success":
        description = dataset_info.get("description", "Dataset analyzed successfully")
        if description and description != "Dataset analyzed successfully":
            table.add_row("Summary", description)
        table.add_row("Modality", f"[bold]{dataset_info.get('modality', 'N/A')}[/]")
        table.add_row("Task", f"[bold]{dataset_info.get('task_type', 'N/A')}[/]")
        table.add_row("Format", dataset_info.get("format", "N/A"))
        classes = dataset_info.get("classes", [])
        if classes:
            class_str = ", ".join(str(value) for value in classes[:8]) + (
                f" [dim](+{len(classes)-8} more)[/]" if len(classes) > 8 else ""
            )
            table.add_row("Classes", f"{len(classes)} - {class_str}")
        splits = dataset_info.get("splits", {})
        if splits:
            split_parts = [f"[green]{k}[/]: {v}" for k, v in splits.items() if v]
            table.add_row("Splits", " | ".join(split_parts))
        structure = dataset_info.get("structure_summary", "")
        if structure:
            table.add_row("Structure", structure[:80] + ("..." if len(structure) > 80 else ""))
        rec_config = dataset_info.get("recommended_config", {})
        if rec_config.get("key_path"):
            table.add_row("Config", f"[green]{dataset_config_path}[/]")
    else:
        table.add_row("(analysis failed)", dataset_info.get("error", "N/A"))

    # Section: User specification (from intent)
    table.add_row("", "[bold]User specification[/]")
    for field, value in get_user_spec_display_rows(user_spec):
        table.add_row(field, value)

    # Section: Device
    table.add_row("", "[bold]Device[/]")
    for field, value in get_device_info_display_rows(device_info):
        table.add_row(field, value)

    if runtime_config is not None:
        table.add_row("", "[bold]ML runtimes[/]")
        table.add_row("Docker image", runtime_config.docker_image or "N/A")
        table.add_row(
            "Available",
            ", ".join(runtime_config.available_runtimes) or "N/A",
        )
        if runtime_config.trt_version:
            table.add_row("TensorRT", runtime_config.trt_version)
        if runtime_config.torch_version:
            table.add_row("PyTorch", runtime_config.torch_version)
        if runtime_config.l4t_version:
            table.add_row("L4T", runtime_config.l4t_version[:80] + ("..." if len(runtime_config.l4t_version) > 80 else ""))

    return Panel(
        table,
        title="[bold]Intent analysis[/]",
        subtitle="[dim italic]Dataset and user spec[/]",
        border_style="blue",
    )


def _verify_request(
    intent: str,
    user_spec: UserSpec,
    dataset_info: Dict[str, Any],
    device_display_name: str,
) -> Tuple[bool, str]:
    """Use LLM to verify the request is relevant to edge/ML and has adequate info for synthesis.

    Returns:
        (True, reason) if verification passes, (False, reason) otherwise.
    """
    context_parts = [
        f"Task description: {user_spec.description}",
        f"Task type: {user_spec.task_type.value if user_spec.task_type else 'N/A'}",
        f"Input type: {user_spec.input_type.value if user_spec.input_type else 'N/A'}",
        f"Constraints: {len(user_spec.constraints)} specified",
        f"Local dataset provided: {'yes' if user_spec.dataset_path else 'no'}",
        f"Local dataset path (existence already verified by preflight): {user_spec.dataset_path or 'N/A'}",
        f"Dataset analysis status: {dataset_info.get('status', 'unknown')}",
        f"Dataset modality: {dataset_info.get('modality', 'N/A')}",
        f"Dataset task: {dataset_info.get('task_type', 'N/A')}",
        f"Dataset summary: {dataset_info.get('description', 'N/A')[:200]}",
        f"Dataset structure: {str(dataset_info.get('structure_summary', 'N/A'))[:300]}",
        f"Dataset splits: {dataset_info.get('splits', 'N/A')}",
        f"Dataset classes: {dataset_info.get('classes', 'N/A')}",
        f"Target device: {device_display_name}",
    ]
    context = "\n".join(context_parts)

    llm = create_chat_llm(temperature=0, purpose="request_verification")

    prompt = REQUEST_VERIFICATION_PROMPT.format(intent=intent, context=context)
    messages = [
        SystemMessage(content=REQUEST_VERIFICATION_SYSTEM),
        HumanMessage(content=prompt),
    ]
    retry = 0
    max_retries = 5
    reason = "Verification could not be completed."
    while retry < max_retries:
        try:
            response = llm.invoke(messages)
            content = (response.content or "").strip()
            if ";" in content:
                ans, reason = content.split(";", 1)
                reason = reason.strip()
            else:
                ans = content
                reason = content
            if "yes" in ans.strip().lower():
                return True, reason
            return False, reason
        except Exception as e:
            retry += 1
            reason = str(e)
            if retry >= max_retries:
                return False, f"Request verification failed after {max_retries} attempts: {reason}"
    return False, reason


def _has_labeled_local_samples(dataset_info: Dict[str, Any]) -> bool:
    """Return True when local data is enough for trial-local train/val splits.

    This is not a quality claim. It only prevents the preflight verifier from
    killing a real labeled dataset before the synthesis loop can turn its
    limitation into evidence.
    """
    if not dataset_info or dataset_info.get("status") not in {None, "success"}:
        return False

    splits = dataset_info.get("splits") or {}
    has_rows = False
    if isinstance(splits, dict):
        has_rows = any(isinstance(v, int) and v > 0 for v in splits.values())

    sample = dataset_info.get("sample_observation") or {}
    has_label_key = False
    has_label_values = False
    if isinstance(sample, dict):
        has_rows = has_rows or bool(sample.get("num_observed_samples"))
        label_summary = sample.get("label_summary") or {}
        has_label_key = bool(sample.get("label_key"))
        has_label_values = bool(label_summary.get("unique_labels_observed")) or bool(
            label_summary.get("label_counts")
        )

    has_classes = bool(dataset_info.get("classes"))
    return bool(has_rows and (has_label_key or has_label_values or has_classes))


def _add_preflight_warning(dataset_info: Dict[str, Any], warning: str) -> None:
    warnings = dataset_info.setdefault("preflight_warnings", [])
    if isinstance(warnings, list):
        warnings.append(warning)


def _sanitize_external_dataset_notes(dataset_info: Dict[str, Any]) -> None:
    """Keep analyzer notes local-data-only; move external-source advice to provenance.

    Dataset analyzers may mention where a benchmark originally came from. That
    is useful provenance, but it is not permission for generated train.py to
    fetch replacement data. Keep this boundary in the evidence structure instead
    of hoping every downstream prompt interprets free text correctly.
    """
    rec = dataset_info.get("recommended_config") or {}
    if not isinstance(rec, dict):
        return
    notes = str(rec.get("notes") or "").strip()
    if not notes:
        return
    lower = notes.lower()
    external_markers = (
        "huggingface hub",
        "huggingface",
        "load_dataset",
        "download",
        "obtain full dataset",
        "external",
    )
    if not any(marker in lower for marker in external_markers):
        return
    provenance = dataset_info.setdefault("provenance_notes", [])
    if isinstance(provenance, list):
        provenance.append(notes)
    rec["notes"] = (
        "Use only files under the local dataset root. External source names or "
        "SOURCE.txt entries are provenance only; do not fetch replacement "
        "training data. If labeled local data has a single split, create "
        "trial-local train/val/test splits in the workspace and record that "
        "limitation in metrics."
    )


def _native_envs_for_device(device_id: str) -> Dict[str, str]:
    try:
        from edgecraft.config.device_manifest import load_declared_devices

        device = load_declared_devices().get(device_id) or {}
        if str(device.get("execution_mode") or "").lower() != "native":
            return {}
        envs = device.get("native_envs") or {}
        return {str(k): str(v).strip() for k, v in envs.items() if str(v).strip()}
    except (OSError, ValueError):
        return {}
    return {}


def _native_python_for_device(device_id: str) -> Optional[str]:
    env_value = edgecraft_env("EDGE_NATIVE_PYTHON").strip()
    if env_value:
        return env_value
    native_envs = _native_envs_for_device(device_id)
    return str(native_envs.get("onnxruntime") or native_envs.get("litert") or "").strip() or None


def _resolved_dataset_config_path(dataset: str, recommended_config: Dict[str, Any]) -> str:
    """Return an actual dataset config file, otherwise preserve the CLI root.

    Analyzer anchor files such as README files are useful evidence, but they are
    not loader inputs.  DatasetPreflight.config_path is deliberately narrower:
    it names a YAML config that a materializer can consume directly.
    """
    key_path = str((recommended_config or {}).get("key_path") or "").strip()
    if not key_path or Path(key_path).suffix.lower() not in {".yaml", ".yml"}:
        return dataset
    dataset_base = dataset if os.path.isdir(dataset) else os.path.dirname(dataset)
    candidate = key_path if os.path.isabs(key_path) else os.path.join(dataset_base, key_path)
    if os.path.isfile(candidate):
        return candidate
    basename_candidate = os.path.join(dataset_base, os.path.basename(key_path))
    if os.path.isfile(basename_candidate):
        return basename_candidate
    import glob as _glob

    yamls = sorted(
        _glob.glob(os.path.join(dataset_base, "*.yaml"))
        + _glob.glob(os.path.join(dataset_base, "*.yml"))
    )
    return yamls[0] if yamls else dataset


_CLOUD_PACKAGE_PROBES = {
    "torch": ("torch", "torch"),
    "torchvision": ("torchvision", "torchvision"),
    "torchaudio": ("torchaudio", "torchaudio"),
    "torchcodec": ("torchcodec", "torchcodec"),
    "numpy": ("numpy", "numpy"),
    "pandas": ("pandas", "pandas"),
    "sklearn": ("sklearn", "scikit-learn"),
    "scipy": ("scipy", "scipy"),
    "soundfile": ("soundfile", "soundfile"),
    "librosa": ("librosa", "librosa"),
    "datasets": ("datasets", "datasets"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "onnx": ("onnx", "onnx"),
    "onnxruntime": ("onnxruntime", "onnxruntime"),
    "ultralytics": ("ultralytics", "ultralytics"),
    "transformers": ("transformers", "transformers"),
    "tokenizers": ("tokenizers", "tokenizers"),
    "cv2": ("cv2", "opencv-python"),
    "PIL": ("PIL", "Pillow"),
    "yaml": ("yaml", "PyYAML"),
    "tensorflow": ("tensorflow", "tensorflow"),
    "ai_edge_litert": ("ai_edge_litert", "ai-edge-litert"),
    "tflite_runtime": ("tflite_runtime", "tflite-runtime"),
    "psutil": ("psutil", "psutil"),
    "joblib": ("joblib", "joblib"),
    "xgboost": ("xgboost", "xgboost"),
    "lightgbm": ("lightgbm", "lightgbm"),
    "catboost": ("catboost", "catboost"),
    "autogluon": ("autogluon.tabular", "autogluon.tabular"),
    "flaml": ("flaml", "flaml"),
    "skl2onnx": ("skl2onnx", "skl2onnx"),
    "onnxmltools": ("onnxmltools", "onnxmltools"),
}


def probe_cloud_python_packages() -> Dict[str, str]:
    """Return importability evidence for server-side ``train.py`` code.

    This is preflight evidence, not a control system: it does not install,
    ban, rewrite, or select packages.  It tells the LLM what the current
    training process can actually import.
    """
    packages: Dict[str, str] = {}
    for name, (module_name, dist_name) in _CLOUD_PACKAGE_PROBES.items():
        try:
            module = importlib.import_module(module_name)
            version = str(getattr(module, "__version__", "") or "")
            if not version:
                try:
                    version = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    version = "importable"
            packages[name] = version
        except Exception as exc:  # noqa: BLE001 - import failure is useful evidence.
            packages[name] = f"unavailable:{exc.__class__.__name__}"
    return packages


def run_synth_preflight(
    intent: str,
    dataset: Optional[str],
    ip: Optional[str],
    ssh_key: Optional[str],
    console: Console,
    *,
    docker_image: Optional[str],
    sleep_after_steps: bool = True,
    modality_hint: Optional[str] = None,
    task_type_hint: Optional[str] = None,
    device_preflight: Optional[DevicePreflight] = None,
) -> SynthCheckResult:
    """Run all preflight checks for synth: parse UserSpec, SSH/device probe, dataset validation and analysis.

    Prints progress messages and the intent analysis panel to the given console.
    Returns SynthCheckResult; when success=False, callers should display result.message and abort.
    """
    # 1. Parse UserSpec from intent
    console.print("[dim]Parsing user specification from intent...[/dim]")
    user_spec_llm = parse_user_spec_from_intent(intent)
    user_spec_intent = user_spec_llm.model_copy(update={"dataset_path": None})
    console.print("[dim]User specification parsed.[/dim]")
    if sleep_after_steps:
        time.sleep(2)

    # 2. Require device IP
    if not ip:
        return SynthCheckResult(
            success=False,
            message="No device IP provided. Synthesis requires an edge device IP.",
        )
    try:
        validate_ssh_target(ip)
    except ValueError as exc:
        return SynthCheckResult(success=False, message=str(exc))
    if not ssh_key:
        return SynthCheckResult(
            success=False,
            message=(
                "No SSH key provided. EdgeCraft requires an explicit private key "
                "and never relies on implicit SSH identity selection."
            ),
        )

    if device_preflight is not None:
        if device_preflight.device_ip != ip:
            return SynthCheckResult(
                success=False,
                message="Device preflight snapshot does not match the requested device IP.",
            )
        if (
            device_preflight.device_info.status != "success"
            or device_preflight.device_info.device_id != device_preflight.device_id
            or not device_preflight.runtime_config.available_runtimes
        ):
            return SynthCheckResult(
                success=False,
                message="Device preflight snapshot lacks verified identity or runtime evidence.",
            )
        device_info = device_preflight.device_info.model_copy(deep=True)
        runtime_cfg = device_preflight.runtime_config.model_copy(deep=True)
        runtime_cfg.cloud_python_packages = probe_cloud_python_packages()
        device_id = device_preflight.device_id
        console.print(
            "[dim]Reusing measured device/runtime facts "
            f"for {device_info.display_name}; live edge execution remains required.[/dim]"
        )
    else:
        # 3. Test SSH connection and edge runner availability.
        console.print(f"[dim]Testing SSH connection: {ip}...[/dim]")
        is_connected, message = test_ssh_connection(ip, ssh_key)
        if not is_connected:
            return SynthCheckResult(
                success=False,
                message=f"Cannot connect to edge device. Synthesis fails. Message: {message}",
            )
        console.print("[dim]Connection successful.[/dim]")
        console.print("[dim]Checking edge runner status...[/dim]")
        is_runner_active, runner_message = ensure_edge_runner_active(ip, ssh_key)
        if not is_runner_active:
            return SynthCheckResult(
                success=False,
                message=f"Edge runner not ready. {runner_message}",
            )
        console.print(f"[dim]{runner_message}.[/dim]")

        # 4. Probe device.
        with console.status("[dim]Analyzing device model and capabilities...[/dim]"):
            device_info = run_device_analyzer(ip, ssh_key)
        if device_info.status == "success":
            console.print(f"[dim]Device identified: {device_info.display_name}.[/dim]")
        else:
            console.print(
                f"[yellow]Warning: Device probe failed: {device_info.error}.[/]"
            )
        if sleep_after_steps:
            time.sleep(2)

        if device_info.status != "success" or not device_info.device_id:
            arch = getattr(device_info, "arch", "") or "unknown"
            model = getattr(device_info, "display_name", "") or getattr(device_info, "raw_model", "") or "unknown"
            return SynthCheckResult(
                success=False,
                message=(
                    "Device probe did not produce a known device_id "
                    f"(model={model}, arch={arch}). Add the device to the "
                    "device mapping instead of falling back to a Jetson default."
                ),
            )

        device_id = device_info.device_id
        native_python = _native_python_for_device(device_id)
        use_native_runtime = bool(native_python) and not (docker_image or "").strip()
        dp = None
        if use_native_runtime:
            console.print(f"[dim]Native runtime preflight on edge: {native_python}[/dim]")
        else:
            if not (docker_image or "").strip():
                return SynthCheckResult(
                    success=False,
                    message=(
                        "Docker image is required for this device. Native runtime is only "
                        "available when the device mapping declares execution_mode=native."
                    ),
                )
            console.print("[dim]Docker image preflight on edge (local tag, arch, imports)...[/dim]")
            dp = run_docker_image_preflight(
                device_ip=ip,
                ssh_key=ssh_key,
                docker_image=docker_image or "",
                device_id=device_id,
                flavor="edge_runner",
                total_timeout_sec=60,
                attempts=2,
            )
            if not dp.ok:
                return SynthCheckResult(
                    success=False,
                    message=f"Docker preflight failed: {dp.message}",
                )
            console.print("[dim]Docker image preflight OK.[/dim]")

        # 4.6 Probe ML runtimes after the exact edge environment is known-good.
        with console.status("[dim]Probing ML runtimes on device…[/dim]"):
            runtime_cfg = probe_device_runtime(
                device_id,
                ip,
                ssh_key,
                docker_image=docker_image,
                native_python=native_python if use_native_runtime else None,
            )
        runtime_cfg.cloud_python_packages = probe_cloud_python_packages()
        if dp and dp.details:
            if dp.details.get("python"):
                runtime_cfg.python_version = str(dp.details.get("python"))
            packages = dp.details.get("python_packages")
            if isinstance(packages, dict):
                runtime_cfg.python_packages = {
                    str(name): str(status)
                    for name, status in packages.items()
                    if name and status is not None
                }
            if not runtime_cfg.onnxruntime_version and dp.details.get("onnxruntime_version"):
                runtime_cfg.onnxruntime_version = str(dp.details.get("onnxruntime_version"))
            providers = dp.details.get("available_providers")
            if isinstance(providers, list):
                runtime_cfg.onnxruntime_providers = [str(p) for p in providers if p]
            if not runtime_cfg.torch_version and dp.details.get("torch_version"):
                runtime_cfg.torch_version = str(dp.details.get("torch_version"))
            if not runtime_cfg.trt_version and dp.details.get("tensorrt_version"):
                runtime_cfg.trt_version = str(dp.details.get("tensorrt_version"))
    console.print(
        f"[dim]Runtimes: {', '.join(runtime_cfg.available_runtimes)} "
        f"(docker_image={runtime_cfg.docker_image})[/dim]"
    )

    # 5. Require dataset path
    if not dataset:
        return SynthCheckResult(
            success=False,
            message="No dataset path provided.",
        )
    if not os.path.exists(dataset):
        return SynthCheckResult(
            success=False,
            message=f"Dataset path '{dataset}' does not exist.",
        )
    console.print(f"[dim]Dataset path '{dataset}' exists.[/dim]")

    # 6. Analyze dataset
    dataset_config_path = dataset
    with console.status("[dim]Analyzing dataset structure...[/dim]"):
        analyzer = DatasetAnalyzer()
        dataset_info = analyzer.run(
            dataset_path=dataset,
            requested_modality=modality_hint or "",
            requested_task_type=task_type_hint or "",
        )

    if modality_hint:
        dataset_info["requested_modality_hint"] = modality_hint
    if task_type_hint:
        dataset_info["requested_task_type_hint"] = task_type_hint

    if dataset_info.get("status") == "success":
        console.print("[dim]Dataset structure analyzed.[/dim]")
    else:
        console.print(
            f"[yellow]Warning: Dataset analysis failed: {dataset_info.get('error')}. Using raw path.[/]"
        )
    _sanitize_external_dataset_notes(dataset_info)

    dataset_config_path = _resolved_dataset_config_path(
        dataset,
        dataset_info.get("recommended_config", {}),
    )

    if sleep_after_steps:
        time.sleep(2)

    # 7. Build and print intent analysis panel. Dataset evidence can correct
    # weak/default intent modality parses, e.g. single-table CSV classification
    # must remain structured rather than drifting into vision.
    user_spec_merged = merge_user_spec_with_dataset_evidence(user_spec_intent, dataset, dataset_info)
    panel = _build_intent_analysis_panel(
        user_spec_merged,
        device_info,
        dataset_info,
        dataset_config_path,
        runtime_config=runtime_cfg,
    )
    console.print(panel)

    # 8. Request verification: relevant and adequate for ML / user objectives
    console.print("[dim]Verifying request (relevance and adequacy for synthesis)...[/dim]")
    time.sleep(2)
    verified, verification_reason = _verify_request(
        intent, user_spec_merged, dataset_info, device_info.display_name
    )
    if not verified:
        if _has_labeled_local_samples(dataset_info):
            warning = (
                "Request verifier was conservative, but local labeled samples exist; "
                "continuing with trial-local splits. Verifier reason: "
                f"{verification_reason}"
            )
            _add_preflight_warning(dataset_info, warning)
            console.print(f"[yellow]Warning: {warning}[/]")
        else:
            return SynthCheckResult(
                success=False,
                message=verification_reason,
            )
    else:
        console.print("[dim]Request verified.[/dim]")
    if sleep_after_steps:
        time.sleep(2)

    return SynthCheckResult(
        success=True,
        intent=IntentPreflight(raw_user_intent=intent, user_spec=user_spec_intent),
        dataset=DatasetPreflight(
            root_path=dataset,
            config_path=dataset_config_path,
            info=dataset_info,
        ),
        device_ctx=DevicePreflight(
            device_id=device_id,
            device_ip=ip,
            device_info=device_info,
            runtime_config=runtime_cfg,
            captured_at=(
                device_preflight.captured_at
                if device_preflight is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            source=(
                device_preflight.source
                if device_preflight is not None
                else "live_probe"
            ),
        ),
    )


# ---------------------------------------------------------------------------
# RuntimeConfig probe (used during synth preflight after Docker image check)
# ---------------------------------------------------------------------------

def probe_device_runtime(
    device_id: str,
    device_ip: Optional[str] = None,
    ssh_key: Optional[str] = None,
    docker_image: Optional[str] = None,
    native_python: Optional[str] = None,
) -> RuntimeConfig:
    """Probe a device's available ML runtimes via SSH and return a RuntimeConfig.

    Falls back to a safe default RuntimeConfig when the probe fails (e.g. device
    is offline during development).
    """
    from edgecraft.models import ensure_registries_initialized
    from edgecraft.models.family_registry import DeviceRegistry
    from edgecraft.models.specs import RuntimeId

    ensure_registries_initialized()
    device_spec = DeviceRegistry.get(device_id)
    has_gpu = device_spec.has_gpu if device_spec else False
    native_py = (native_python or _native_python_for_device(device_id) or "").strip()
    native_envs = _native_envs_for_device(device_id) if native_py else {}
    img = (docker_image or "").strip() or (f"native:{native_py}" if native_py else "unset")
    device_arch = "aarch64"
    if device_spec and str(device_spec.device_class.value).startswith("x86"):
        device_arch = "x86_64"

    # Attempt live probe
    available_runtimes = ["onnxruntime"] if not device_ip else []
    trt_version = None
    ort_version = None
    ort_providers = []
    torch_version = None
    l4t_version = None
    python_packages: Dict[str, str] = {}
    runtime_python_packages: Dict[str, Dict[str, str]] = {}

    if device_ip:
        from edgecraft.utils.network import run_remote_command
        from edgecraft.tools.deploy.docker_opts import gpu_opts_shell

        use_docker = bool(img and img != "unset" and not native_py)
        docker_opts = f"--rm {gpu_opts_shell(has_gpu)}" if use_docker else ""

        try:
            def _run(cmd: str, python_bin: Optional[str] = None) -> str:
                if use_docker:
                    wrapped_cmd = f"docker run {docker_opts} {img} {cmd}"
                elif (python_bin or native_py) and cmd.startswith("python3 "):
                    py = shlex.quote(python_bin or native_py)
                    wrapped_cmd = cmd.replace("python3 ", f"{py} ", 1)
                else:
                    wrapped_cmd = cmd
                # Returns (returncode, stdout, stderr)
                rc, out, err = run_remote_command(device_ip, wrapped_cmd, ssh_key=ssh_key, timeout=15)
                if rc == 0 and out:
                    return out.strip()
                return ""

            # Check TensorRT
            trt_out = _run(
                "python3 -c \"import tensorrt; print(tensorrt.__version__)\" 2>/dev/null"
            )
            if trt_out and not trt_out.startswith("Traceback"):
                available_runtimes.append("tensorrt")
                trt_version = trt_out

            # Check PyTorch
            torch_out = _run(
                "python3 -c \"import torch; print(torch.__version__)\" 2>/dev/null",
                python_bin=native_envs.get("pytorch") or native_envs.get("torch") or native_py,
            )
            if torch_out and not torch_out.startswith("Traceback"):
                available_runtimes.append("torch")
                torch_version = torch_out

            # Check ONNXRuntime providers, not just importability. CPU-only ORT
            # is still useful evidence, but it should not masquerade as GPU ORT.
            ort_out = _run(
                "python3 -c \"import json, onnxruntime as ort; print(json.dumps({'version': ort.__version__, 'providers': ort.get_available_providers()}))\" 2>/dev/null"
            )
            if ort_out and not ort_out.startswith("Traceback"):
                available_runtimes.append("onnxruntime")
                try:
                    ort_details = json.loads(ort_out.splitlines()[-1])
                    ort_version = str(ort_details.get("version") or "") or None
                    providers = ort_details.get("providers") or []
                    if isinstance(providers, list):
                        ort_providers = [str(p) for p in providers if p]
                except Exception:
                    ort_version = ort_out

            # Check TFLite/LiteRT.  Some images expose tflite_runtime, newer
            # ones may expose ai_edge_litert.  This is evidence only.
            tflite_out = _run(
                "python3 -c \"import importlib.util, json; "
                "mods=['tflite_runtime','ai_edge_litert']; "
                "hits=[m for m in mods if importlib.util.find_spec(m) is not None]; "
                "print(json.dumps({'modules': hits}))\" 2>/dev/null",
                python_bin=native_envs.get("litert") or native_py,
            )
            if tflite_out and not tflite_out.startswith("Traceback"):
                try:
                    details = json.loads(tflite_out.splitlines()[-1])
                    modules = details.get("modules") or []
                    if modules:
                        available_runtimes.append("tflite")
                        python_packages.update({str(name): "available" for name in modules})
                except Exception:
                    pass

            package_script = (
                "import importlib.util,json; "
                "mods=['numpy','pandas','scipy','sklearn','pyarrow','joblib','torch',"
                "'torchvision','torchaudio','onnx','onnxruntime','datasets','PIL','cv2',"
                "'yaml','ultralytics','timm','librosa','soundfile','transformers','tokenizers',"
                "'ai_edge_litert','tflite_runtime','tensorflow','psutil']; "
                "print(json.dumps({m:('available' if importlib.util.find_spec(m) else "
                "'unavailable:ModuleNotFoundError') for m in mods}))"
            )
            package_targets = (
                {"docker": None}
                if use_docker
                else {
                    "onnxruntime": native_envs.get("onnxruntime") or native_py,
                    "pytorch": native_envs.get("pytorch") or native_envs.get("torch"),
                    "litert": native_envs.get("litert"),
                }
            )
            for runtime_name, python_path in package_targets.items():
                if not use_docker and not python_path:
                    continue
                packages_out = _run(
                    f"python3 -c {shlex.quote(package_script)}",
                    python_bin=python_path,
                )
                if not packages_out:
                    continue
                try:
                    parsed = json.loads(packages_out.splitlines()[-1])
                    if isinstance(parsed, dict):
                        runtime_python_packages[runtime_name] = {
                            str(name): str(status)
                            for name, status in parsed.items()
                        }
                except Exception:
                    continue
            if use_docker:
                python_packages = dict(runtime_python_packages.pop("docker", {}))
            elif native_py:
                python_packages = dict(runtime_python_packages.get("onnxruntime") or {})

            # L4T version (Jetson only)
            l4t_out = _run("cat /etc/nv_tegra_release 2>/dev/null | head -1")
            if l4t_out:
                l4t_version = l4t_out

        except Exception as exc:
            pass  # Offline / SSH failure → use catalogue defaults

    # Catalogue capabilities remain useful for offline planning, but a live
    # target must expose only runtimes proven by the probe above.
    if not device_ip and has_gpu and "tensorrt" not in available_runtimes:
        available_runtimes.append("tensorrt")
    if (
        not device_ip
        and device_spec
        and RuntimeId.TFLITE in device_spec.supported_runtimes
        and "tflite" not in available_runtimes
    ):
        available_runtimes.append("tflite")

    return RuntimeConfig(
        available_runtimes=list(dict.fromkeys(available_runtimes)),
        docker_image=img,
        device_arch=device_arch,
        has_gpu=has_gpu,
        trt_version=trt_version,
        onnxruntime_version=ort_version,
        onnxruntime_providers=ort_providers,
        torch_version=torch_version,
        l4t_version=l4t_version,
        python_packages=python_packages,
        runtime_python_packages=runtime_python_packages,
        edge_artifact_bundle_max_mb=max(
            1, int(edgecraft_env("EDGE_ARTIFACT_BUNDLE_MAX_MB", "256"))
        ),
    )
