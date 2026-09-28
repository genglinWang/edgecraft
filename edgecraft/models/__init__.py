"""EdgeCraft model-zoo public API and explicit registry initialization."""
from loguru import logger

# New architecture components
from edgecraft.models.specs import (
    ModelSpec,
    BaseFamilyPlugin,
    TemplateContext,
    DeviceSpec,
    RuntimeSpec,
    QuantMode,
    ExportFormat,
    RuntimeId,
    DeviceClass,
)
from edgecraft.models.family_registry import (
    FamilyRegistry,
    DeviceRegistry,
    RuntimeRegistry,
)
from edgecraft.models.profile_store import (
    ProfileStore,
    ProfileRequest,
    ProfileResult,
    get_profile_store,
)
from edgecraft.models.offline_profiler import OfflineProfiler
from edgecraft.config.device_manifest import load_declared_devices, normalized_runtimes


def _init_family_registry() -> None:
    """Initialize family plugins."""
    from edgecraft.models.families import (
        ultralytics_plugin,
        timm_plugin,
        torchvision_plugin,
        anomaly_detection_plugin,
        crowd_counting_plugin,
        structured_log_plugin,
        transformers_plugin,
        tiny_audio_asr_plugin,
        whisper_plugin,
        speechbrain_plugin,
    )

    # Vision families
    FamilyRegistry.register_plugin(ultralytics_plugin)
    FamilyRegistry.register_plugin(timm_plugin)
    FamilyRegistry.register_plugin(torchvision_plugin)
    FamilyRegistry.register_plugin(anomaly_detection_plugin)
    FamilyRegistry.register_plugin(crowd_counting_plugin)

    # Text families
    FamilyRegistry.register_plugin(transformers_plugin)
    FamilyRegistry.register_plugin(structured_log_plugin)

    # Audio families
    FamilyRegistry.register_plugin(tiny_audio_asr_plugin)
    FamilyRegistry.register_plugin(whisper_plugin)
    FamilyRegistry.register_plugin(speechbrain_plugin)

    stats = FamilyRegistry.get_stats()
    # logger.debug(
    #     f"FamilyRegistry initialized: {stats['total_families']} families, "
    #     f"{stats['total_models']} models"
    # )


def _init_device_registry() -> None:
    """Initialize device registry with known devices."""
    devices = [
        DeviceSpec(
            device_id="jetson_orin_agx",
            device_class=DeviceClass.JETSON,
            display_name="NVIDIA Jetson AGX Orin",
            has_gpu=True,
            gpu_name="Orin",
            compute_capability="8.7",
            memory_gb=32.0,
            gpu_memory_gb=32.0,
            supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
            preferred_runtime=RuntimeId.TENSORRT,
            latency_scale=1.0,
        ),
        DeviceSpec(
            device_id="jetson_xavier_nx",
            device_class=DeviceClass.JETSON,
            display_name="NVIDIA Jetson Xavier NX",
            has_gpu=True,
            gpu_name="Xavier",
            compute_capability="7.2",
            memory_gb=8.0,
            gpu_memory_gb=8.0,
            supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
            preferred_runtime=RuntimeId.TENSORRT,
            latency_scale=1.6,
        ),
        DeviceSpec(
            device_id="jetson_tx2",
            device_class=DeviceClass.JETSON,
            display_name="NVIDIA Jetson TX2",
            has_gpu=True,
            gpu_name="Pascal",
            compute_capability="6.2",
            memory_gb=4.0,
            gpu_memory_gb=4.0,
            supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
            preferred_runtime=RuntimeId.ONNXRUNTIME,
            latency_scale=3.2,
        ),
        DeviceSpec(
            device_id="raspberry_pi_5",
            device_class=DeviceClass.RASPBERRY_PI,
            display_name="Raspberry Pi 5",
            has_gpu=False,
            memory_gb=8.0,
            supported_runtimes=[RuntimeId.ONNXRUNTIME],
            preferred_runtime=RuntimeId.ONNXRUNTIME,
            latency_scale=8.0,
        ),
        DeviceSpec(
            device_id="x86_cpu_desktop",
            device_class=DeviceClass.X86_CPU,
            display_name="Desktop CPU",
            has_gpu=False,
            memory_gb=32.0,
            supported_runtimes=[RuntimeId.ONNXRUNTIME, RuntimeId.TFLITE],
            preferred_runtime=RuntimeId.ONNXRUNTIME,
            latency_scale=2.0,
            supports_docker=False,
        ),
        DeviceSpec(
            device_id="x86_gpu_workstation",
            device_class=DeviceClass.X86_GPU,
            display_name="x86 GPU Workstation",
            has_gpu=True,
            memory_gb=64.0,
            gpu_memory_gb=24.0,
            supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
            preferred_runtime=RuntimeId.TENSORRT,
            latency_scale=0.3,
        ),
    ]

    try:
        declared_devices = load_declared_devices()
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not load declared device resources: {exc}")
        declared_devices = {}

    for device in devices:
        facts = declared_devices.get(device.device_id)
        if facts:
            known_runtime_values = {runtime.value for runtime in RuntimeId}
            runtimes = [
                RuntimeId(value)
                for value in normalized_runtimes(facts)
                if value in known_runtime_values
            ]
            device = device.model_copy(update={
                "supported_runtimes": runtimes,
                "supports_docker": str(facts.get("execution_mode") or "docker").lower() == "docker",
            })
        DeviceRegistry.register(device)


def _init_runtime_registry() -> None:
    """Initialize runtime registry with known runtimes."""
    runtimes = [
        RuntimeSpec(
            runtime_id=RuntimeId.PYTORCH,
            display_name="PyTorch",
            supported_formats=[ExportFormat.PT],
            supported_quant_modes=[QuantMode.FP32, QuantMode.FP16],
            supported_device_classes=[
                DeviceClass.JETSON,
                DeviceClass.X86_GPU,
                DeviceClass.X86_CPU,
            ],
            pip_packages=["torch"],
        ),
        RuntimeSpec(
            runtime_id=RuntimeId.ONNXRUNTIME,
            display_name="ONNX Runtime",
            supported_formats=[ExportFormat.ONNX],
            supported_quant_modes=[QuantMode.FP32, QuantMode.FP16, QuantMode.INT8],
            supported_device_classes=[
                DeviceClass.JETSON,
                DeviceClass.RASPBERRY_PI,
                DeviceClass.ANDROID,
                DeviceClass.X86_GPU,
                DeviceClass.X86_CPU,
            ],
            pip_packages=["onnxruntime-gpu"],
        ),
        RuntimeSpec(
            runtime_id=RuntimeId.TENSORRT,
            display_name="NVIDIA TensorRT",
            supported_formats=[ExportFormat.ENGINE, ExportFormat.ONNX],
            supported_quant_modes=[QuantMode.FP32, QuantMode.FP16, QuantMode.INT8],
            supported_device_classes=[DeviceClass.JETSON, DeviceClass.X86_GPU],
            pip_packages=["tensorrt"],
            system_dependencies=["cuda", "cudnn"],
        ),
        RuntimeSpec(
            runtime_id=RuntimeId.TFLITE,
            display_name="TensorFlow Lite",
            supported_formats=[ExportFormat.TFLITE],
            supported_quant_modes=[QuantMode.FP32, QuantMode.FP16, QuantMode.INT8],
            supported_device_classes=[
                DeviceClass.ANDROID,
                DeviceClass.RASPBERRY_PI,
                DeviceClass.X86_CPU,
            ],
            pip_packages=["tflite-runtime"],
        ),
    ]

    for runtime in runtimes:
        RuntimeRegistry.register(runtime)


_initialized = False


def initialize_registries(force: bool = False) -> None:
    """Explicitly initialize model/device/runtime registries once."""
    global _initialized
    if _initialized and not force:
        return
    if force:
        FamilyRegistry.clear()
        DeviceRegistry._devices.clear()  # reset singleton state for tests
        RuntimeRegistry._runtimes.clear()  # reset singleton state for tests
    _init_family_registry()
    _init_device_registry()
    _init_runtime_registry()
    _initialized = True
    stats = FamilyRegistry.get_stats()
    # logger.info(
    #     f"Model registries initialized: {stats['total_families']} families, "
    #     f"{stats['total_models']} models"
    # )


def ensure_registries_initialized() -> None:
    """Backstop for call sites that require registry access."""
    initialize_registries(force=False)


__all__ = [
    # New architecture
    "ModelSpec",
    "BaseFamilyPlugin",
    "TemplateContext",
    "DeviceSpec",
    "RuntimeSpec",
    "QuantMode",
    "ExportFormat",
    "RuntimeId",
    "DeviceClass",
    "FamilyRegistry",
    "DeviceRegistry",
    "RuntimeRegistry",
    "ProfileStore",
    "ProfileRequest",
    "ProfileResult",
    "get_profile_store",
    "OfflineProfiler",
    "initialize_registries",
    "ensure_registries_initialized",
]
