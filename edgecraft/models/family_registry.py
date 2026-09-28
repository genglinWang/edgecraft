"""FamilyRegistry: Central registry for model family plugins.

This module coordinates all family plugins and provides unified access to:
- Model specifications across all families
- Template generation for train.py/infer.py
- Search space descriptions for LLM proposals
- Profile estimation integration

Key principle: This is a COORDINATION layer, not an EXECUTION layer.
All training/inference happens through generated scripts, not internal calls.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from loguru import logger

from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import RuntimeConfig
from edgecraft.models.specs import (
    BaseFamilyPlugin,
    DeviceClass,
    ExportFormat,
    ModelSpec,
    QuantMode,
    RuntimeId,
    TemplateContext,
)


# Public, task-level solution paths shown to the proposer.  These are generic
# model-zoo capabilities rather than dataset-to-model recommendations.  Dataset
# names, reference recipes, and benchmark results do not belong here.
_PUBLIC_SOLUTION_PATHS: Dict[tuple[Modality, TaskType], List[Dict[str, Any]]] = {
    (Modality.VISION, TaskType.CLASSIFICATION): [
        {
            "id": "vision_pretrained_mobile_cnn",
            "representation": "RGB images at the model's native resolution",
            "preprocessing": "task-preserving resize/crop, train augmentation, and pretrained-weight normalization",
            "models": "timm/torchvision MobileNetV3, EfficientNet-B0, ResNet18, ConvNeXt-Tiny",
            "initialization": "ImageNet pretrained fine-tuning or frozen features followed by fine-tuning",
            "training": "validation-driven fine-tuning with an adequate multi-epoch budget, learning-rate scheduling, and early stopping",
            "packages": ["torch", "torchvision"],
        },
        {
            "id": "vision_compact_scratch_cnn",
            "representation": "RGB images with resolution chosen from the latency budget",
            "preprocessing": "dataset-derived normalization and label-preserving augmentation",
            "models": "compact residual CNN, MobileNetV3-Small, ShuffleNetV2",
            "initialization": "random initialization when transfer learning is unsuitable",
            "training": "validation-driven optimization to convergence with label-preserving augmentation and early stopping",
            "packages": ["torch", "torchvision"],
        },
    ],
    (Modality.VISION, TaskType.OBJECT_DETECTION): [
        {
            "id": "vision_pretrained_edge_detector",
            "representation": "RGB images with evaluator-owned boxes and class IDs",
            "preprocessing": "aspect-ratio-aware resize plus box-preserving augmentation",
            "models": "Ultralytics YOLO nano/small or torchvision SSDLite-MobileNetV3",
            "initialization": "public detection-pretrained weights followed by task fine-tuning",
            "training": "full-data fine-tuning with box-safe augmentation, validation mAP selection, and early stopping",
            "packages": ["torch", "torchvision"],
        },
    ],
    (Modality.VISION, TaskType.SEGMENTATION): [
        {
            "id": "vision_torchvision_mobile_segmenter",
            "representation": "paired RGB image and categorical mask",
            "preprocessing": "joint image-mask resize/crop and mask-safe geometric augmentation",
            "models": "torchvision LR-ASPP-MobileNetV3-Large or DeepLabV3-MobileNetV3-Large",
            "initialization": "torchvision segmentation-pretrained weights followed by full fine-tuning",
            "training": "start from the public fine-tuning recipe rather than a one-epoch smoke run: 30-50 full-data epochs, joint augmentation, segmentation loss, learning-rate scheduling, validation-metric checkpointing, and early stopping",
            "packages": ["torch", "torchvision"],
        },
        {
            "id": "vision_segformer_mobile",
            "representation": "paired RGB image and categorical mask",
            "preprocessing": "pinned image processor with joint image-mask resize/crop and mask-safe augmentation",
            "models": "Hugging Face SegFormer-B0 with a pinned public checkpoint",
            "initialization": "public SegFormer-B0 pretraining followed by task-head replacement and full fine-tuning",
            "training": "start from the public fine-tuning recipe rather than a one-epoch smoke run: 30-50 full-data AdamW epochs with a segmentation loss, learning-rate scheduling, validation-metric checkpointing, and early stopping",
            "packages": ["torch", "transformers"],
        },
        {
            "id": "vision_ultralytics_nano_segmenter",
            "representation": "paired RGB image and polygon or categorical-mask labels supported by the observed contract",
            "preprocessing": "Ultralytics-compatible image/label layout with mask-safe geometric augmentation; use a 512-640 input for a quality-oriented root when the measured latency budget has room, and reserve 320-416 for a compact root",
            "models": "Ultralytics yolo11n-seg.pt, yolo11s-seg.pt, or their pinned YOLOv8 counterparts",
            "initialization": "pinned public segmentation checkpoint followed by task fine-tuning",
            "training": "start from the public Ultralytics fine-tuning recipe rather than a one-epoch smoke run: 30-50 full-data epochs with mask-safe augmentation, frozen-metric validation checkpointing, and early stopping; the root portfolio should cover a quality-first resolution/capacity route as well as a compact route when both fit the declared device constraints",
            "packages": ["torch", "ultralytics"],
        },
        {
            "id": "vision_compact_encoder_decoder",
            "representation": "paired RGB image and mask at a latency-aware resolution",
            "preprocessing": "joint normalization and mask-preserving augmentation",
            "models": "compact U-Net or depthwise encoder-decoder",
            "initialization": "random initialization or pretrained mobile encoder",
            "training": "multi-epoch optimization with a segmentation loss, joint augmentation, and validation-based early stopping",
            "packages": ["torch", "torchvision"],
        },
    ],
    (Modality.VISION, TaskType.POSE_ESTIMATION): [
        {
            "id": "vision_pretrained_edge_pose",
            "representation": "RGB image with evaluator-owned keypoints and visibility flags",
            "preprocessing": "keypoint-preserving affine augmentation and heatmap/coordinate target construction",
            "models": "Ultralytics YOLO nano pose or compact MobileNet heatmap regressor",
            "initialization": "public pose/image pretraining followed by fine-tuning",
            "training": "multi-epoch fine-tuning with keypoint-safe augmentation and validation-metric checkpointing",
            "packages": ["torch", "torchvision"],
        },
    ],
    (Modality.VISION, TaskType.CROWD_COUNTING): [
        {
            "id": "vision_pretrained_density_counter",
            "representation": "RGB image with point annotations converted to density targets",
            "preprocessing": "resolution-preserving crops and count-preserving density-map construction",
            "models": "MobileNet density regressor or compact CSRNet-style counter",
            "initialization": "ImageNet pretrained encoder with task-specific density head",
            "training": "count-preserving crop training with validation MAE selection and early stopping",
            "packages": ["torch", "torchvision"],
        },
    ],
    (Modality.VISION, TaskType.ANOMALY_DETECTION): [
        {
            "id": "vision_pretrained_anomaly_features",
            "representation": "normalized image or patch embeddings",
            "preprocessing": "category-consistent resize/crop without label-changing augmentation",
            "models": "pretrained ResNet18 feature model, compact convolutional autoencoder, or patch-distance detector",
            "initialization": "ImageNet pretrained frozen/features or normal-only training",
            "training": "normal-only or supervised training selected from the label contract, with validation thresholding and no test tuning",
            "packages": ["torch", "torchvision", "scikit-learn"],
        },
    ],
    (Modality.AUDIO, TaskType.AUDIO_CLASSIFICATION): [
        {
            "id": "audio_logmel_edge_classifier",
            "representation": "fixed-duration log-mel spectrogram",
            "preprocessing": "resample from observed sample rate, mono conversion, bounded crop/pad, and log-mel normalization; preserve the 2D time-frequency map for DS-CNN/BC-ResNet rather than averaging away the temporal axis",
            "models": "DS-CNN, BC-ResNet, compact CRNN, or SpeechBrain command classifier",
            "initialization": "public audio pretraining when task-compatible, otherwise random initialization",
            "training": "use a complete 30-50 epoch starting recipe with validation checkpointing, learning-rate scheduling, waveform shift/gain/noise augmentation, optional SpecAugment, and early stopping; shrink the budget only after a verification trace shows convergence",
            "deployment": "export a fixed-shape ONNX graph that reuses the training frontend; when a quality-strong candidate narrowly misses latency or energy limits, validate static INT8 or quantization-aware training as a child while preserving the representation and trained model family",
            "packages": ["torch", "torchaudio"],
        },
        {
            "id": "audio_waveform_temporal_model",
            "representation": "normalized raw waveform windows",
            "preprocessing": "resample, amplitude normalization, bounded crop/pad",
            "models": "compact 1D residual CNN or depthwise temporal CNN",
            "initialization": "random initialization or compatible public waveform encoder",
            "training": "adequate multi-epoch optimization with waveform augmentation, learning-rate scheduling, and validation-based early stopping",
            "deployment": "export a fixed-length waveform ONNX graph; use measured post-training quantization or quantization-aware training only when the child preserves task quality and improves the constrained edge metric",
            "packages": ["torch", "torchaudio"],
        },
    ],
    (Modality.AUDIO, TaskType.SPEECH_RECOGNITION): [
        {
            "id": "audio_compact_ctc_asr",
            "representation": "waveform or log-mel sequence with transcript vocabulary",
            "preprocessing": "resample, bounded utterance handling, tokenizer/vocabulary fixed in artifact sidecars",
            "models": "tiny character CTC, compact CRDNN, or small public ASR encoder",
            "initialization": "public ASR pretraining when exportable, otherwise compact CTC training",
            "training": "multi-epoch CTC fine-tuning with length-aware batching, validation error-rate selection, and early stopping",
            "packages": ["torch", "torchaudio"],
        },
    ],
    (Modality.TEXT, TaskType.TEXT_CLASSIFICATION): [
        {
            "id": "text_pretrained_compact_transformer",
            "representation": "subword token IDs and attention mask",
            "preprocessing": "pinned tokenizer, bounded sequence length, evaluator-owned label mapping",
            "models": "quality-oriented compact checkpoints such as distilroberta-base or distilbert-base-uncased, balanced microsoft/MiniLM-L12-H384-uncased, and compact google/mobilebert-uncased or huawei-noah/TinyBERT_General_4L_312D",
            "initialization": "public pretrained checkpoint fine-tuning",
            "training": "use the mature compact-transformer starting recipe: 2-4 full-data AdamW epochs, low fine-tuning learning rate, warmup, weight decay, validation checkpointing, and early stopping rather than a one-epoch development run; when hard constraints permit, the root portfolio should include a quality-oriented checkpoint rather than only neighboring tiny variants",
            "deployment": "export the fine-tuned model with fixed-shape token-ID and attention-mask ONNX inputs; when CPU latency or artifact size requires it, validate post-training dynamic INT8 against the uncompressed checkpoint before selecting it",
            "packages": ["torch", "transformers"],
        },
        {
            "id": "text_compact_neural_or_linear",
            "representation": "word/subword sequence or TF-IDF/hashed sparse features",
            "preprocessing": "training-owned vocabulary/vectorizer serialized with the artifact",
            "models": "TextCNN, EmbeddingBag classifier, or calibrated linear classifier",
            "initialization": "random initialization or fitted feature pipeline",
            "training": "fit on the full training split to validation convergence, with class weighting or calibration when supported by observed imbalance",
            "packages": ["torch", "scikit-learn"],
        },
    ],
    (Modality.TIME_SERIES, TaskType.CLASSIFICATION): [
        {
            "id": "timeseries_raw_window_temporal_cnn",
            "representation": "multichannel windows preserving temporal order",
            "preprocessing": "subject-safe/window-safe split, train-only normalization, and explicit channel order; when multiple asynchronous sensor streams are present, align them by timestamp within the same device/session before windowing rather than concatenating windows by ordinal position; derive classes only from observed labeled rows and do not turn missing labels into a task class",
            "models": "DS-CNN, TCN, ResNet1D, or InceptionTime-style compact network",
            "initialization": "random initialization with a complete temporal training recipe",
            "training": "use a complete 30-60 epoch starting recipe with light jitter/channel dropout, learning-rate scheduling, subject-safe validation checkpointing, and early stopping; preserve all verified aligned channels unless a coherent child tests an ablation",
            "packages": ["torch"],
        },
        {
            "id": "timeseries_statistical_ensemble",
            "representation": "train-fitted per-channel window statistics, correlations, and compact spectral features",
            "preprocessing": "subject-safe/window-safe split, explicit channel order, and a serialized deterministic feature extractor",
            "models": "ExtraTrees, HistGradientBoosting, calibrated linear SVM, or compact MLP",
            "initialization": "fitted public-library estimator",
            "training": "fit the full training split with class-aware loss/weights and bounded validation model selection without test tuning",
            "packages": ["scikit-learn"],
        },
        {
            "id": "timeseries_frequency_image_model",
            "representation": "STFT or compact time-frequency map when periodic structure is observed",
            "preprocessing": "train-fitted scaling and fixed transform parameters serialized with the artifact",
            "models": "compact 2D CNN or mobile pretrained image encoder when channel semantics permit",
            "initialization": "random initialization or public image pretraining",
            "training": "multi-epoch optimization with fixed train-fitted transforms and validation-based checkpoint selection",
            "packages": ["torch"],
        },
        {
            "id": "timeseries_sequence_encoder",
            "representation": "multichannel normalized sequence",
            "preprocessing": "windowing and padding/masking derived from observed sequence lengths",
            "models": "GRU, lightweight Transformer encoder, or temporal convolution-attention hybrid",
            "initialization": "random initialization",
            "training": "multi-epoch sequence training with masking-aware batches, learning-rate scheduling, and validation early stopping",
            "packages": ["torch"],
        },
    ],
    (Modality.TIME_SERIES, TaskType.ANOMALY_DETECTION): [
        {
            "id": "timeseries_anomaly_sequence_model",
            "representation": "ordered windows or sessions with evaluator-owned anomaly labels",
            "preprocessing": "train-only normalization and split-safe window/session construction",
            "models": "TCN/GRU classifier, compact autoencoder, or boosted window features",
            "initialization": "random initialization or fitted structured model",
            "training": "imbalance-aware training with threshold selection on validation only and early stopping for neural routes",
            "packages": ["torch", "scikit-learn"],
        },
    ],
    (Modality.STRUCTURED, TaskType.CLASSIFICATION): [
        {
            "id": "tabular_gradient_boosting",
            "representation": "numeric and categorical columns with target-derived columns excluded",
            "preprocessing": "train-fitted missing-value and categorical handling serialized with the artifact",
            "models": "CatBoost, LightGBM, XGBoost, or sklearn HistGradientBoosting",
            "initialization": "fitted public-library estimator",
            "training": "fit the full training split with validation-based early stopping or bounded model selection and no test-set tuning",
            "packages": ["scikit-learn"],
        },
        {
            "id": "tabular_residual_mlp",
            "representation": "standardized numeric features plus bounded categorical embeddings/encoding",
            "preprocessing": "train-only fitted scaler/encoder serialized or embedded in the graph",
            "models": "compact residual MLP or TabNet-style small network",
            "initialization": "random initialization",
            "training": "multi-epoch optimization with validation checkpointing, class-aware loss when needed, and early stopping",
            "packages": ["torch", "scikit-learn"],
        },
    ],
    (Modality.STRUCTURED, TaskType.REGRESSION): [
        {
            "id": "tabular_boosting_regressor",
            "representation": "audited numeric/categorical feature table",
            "preprocessing": "train-fitted missing-value/categorical processing serialized with the artifact",
            "models": "CatBoost, LightGBM, XGBoost, RandomForest, or HistGradientBoosting regressor",
            "initialization": "fitted public-library estimator",
            "training": "fit the full training split with validation-based early stopping or bounded model selection and no test-set tuning",
            "packages": ["scikit-learn"],
        },
        {
            "id": "tabular_neural_regressor",
            "representation": "standardized numeric features with bounded categorical encoding",
            "preprocessing": "train-only fitted transforms embedded in the deployable path",
            "models": "compact residual MLP",
            "initialization": "random initialization",
            "training": "multi-epoch optimization with validation checkpointing and early stopping",
            "packages": ["torch", "scikit-learn"],
        },
    ],
    (Modality.STRUCTURED, TaskType.ANOMALY_DETECTION): [
        {
            "id": "structured_supervised_anomaly",
            "representation": "audited table features or session event-count vectors with explicit labels",
            "preprocessing": "train-fitted vectorization/scaling, explicit session grouping, leakage columns excluded, and exactly one evaluation row per opaque manifest ID when split IDs denote groups or sessions",
            "models": "event-count linear/boosted classifier, compact MLP, or sequence classifier",
            "initialization": "fitted public-library estimator or random neural initialization",
            "training": "imbalance-aware fitting with validation-only threshold selection and early stopping for neural routes",
            "packages": ["scikit-learn", "torch"],
        },
    ],
    (Modality.TEXT, TaskType.ANOMALY_DETECTION): [
        {
            "id": "log_text_anomaly",
            "representation": "session event-count vector, message features, or ordered event sequence fixed by the label contract",
            "preprocessing": "training-owned tokenizer/vectorizer and explicit label join serialized with the artifact; when split IDs denote groups or sessions, aggregate source rows before modeling and emit exactly one evaluation row per opaque manifest ID",
            "models": "session event-count linear/boosted classifier, hashed TF-IDF classifier, DeepLog-style sequence model, or compact Transformer encoder",
            "initialization": "fitted linear model, random sequence model, or public language pretraining",
            "training": "imbalance-aware full-split training with validation F1 thresholding/checkpointing and no test-set tuning",
            "packages": ["scikit-learn", "torch"],
        },
    ],
}


_PACKAGE_ALIASES: Dict[str, tuple[str, ...]] = {
    "scikit-learn": ("scikit-learn", "sklearn"),
    "pyyaml": ("pyyaml", "yaml"),
    "pillow": ("pillow", "pil"),
}

_RUNTIME_ALIASES: Dict[str, str] = {
    "torch": "pytorch",
    "pt": "pytorch",
    "onnx": "onnxruntime",
    "ort": "onnxruntime",
    "engine": "tensorrt",
    "litert": "tflite",
}


def _normalized_package_state(packages: Dict[str, str], package: str) -> Optional[str]:
    aliases = _PACKAGE_ALIASES.get(package.lower(), (package.lower(),))
    normalized = {str(key).lower(): str(value) for key, value in packages.items()}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _packages_are_possible(required: List[str], runtime_config: Optional[RuntimeConfig]) -> bool:
    if runtime_config is None or not runtime_config.cloud_python_packages:
        return True
    for package in required:
        state = _normalized_package_state(runtime_config.cloud_python_packages, package)
        if state is not None and state.lower().startswith("unavailable"):
            return False
    return True


def _available_runtime_ids(runtime_config: Optional[RuntimeConfig]) -> set[str]:
    if runtime_config is None or not runtime_config.available_runtimes:
        return set()
    values = set()
    for item in runtime_config.available_runtimes:
        value = str(item).strip().lower()
        values.add(_RUNTIME_ALIASES.get(value, value))
    return values


def _model_has_runtime_path(spec: ModelSpec, runtime_config: Optional[RuntimeConfig]) -> bool:
    available = _available_runtime_ids(runtime_config)
    if not available:
        return True
    supported = {
        item.value if hasattr(item, "value") else str(item)
        for item in spec.supported_runtimes
    }
    return bool(available.intersection(supported))


def _representative_models(models: List[ModelSpec], *, limit: int) -> List[ModelSpec]:
    grouped: Dict[str, List[ModelSpec]] = {}
    for spec in models:
        grouped.setdefault(spec.family_id, []).append(spec)

    selected: List[ModelSpec] = []
    for family_id in sorted(grouped):
        family = sorted(
            (spec for spec in grouped[family_id] if spec.params_m <= 100.0),
            key=lambda spec: (not bool(spec.pretrained_weights), spec.params_m, spec.name),
        )
        if not family:
            continue
        selected.append(family[0])
        if len(family) > 2:
            selected.append(family[len(family) // 2])
    return selected[:limit]


class FamilyRegistry:
    """Central registry for model family plugins.

    Manages registration and lookup of family plugins, providing unified
    access to models, templates, and search space information.
    """

    _plugins: Dict[str, BaseFamilyPlugin] = {}
    _models: Dict[str, ModelSpec] = {}  # model_id -> ModelSpec

    @classmethod
    def register_plugin(cls, plugin: BaseFamilyPlugin) -> None:
        """Register a family plugin."""
        family_id = plugin.family_id

        if family_id in cls._plugins:
            logger.warning(f"FamilyRegistry: overwriting plugin {family_id}")

        cls._plugins[family_id] = plugin

        # Register all models from this plugin
        specs = plugin.get_model_specs()
        for spec in specs:
            cls._models[spec.model_id] = spec
            logger.trace(f"FamilyRegistry: registered model {spec.model_id}")

        logger.trace(f"FamilyRegistry: registered plugin {family_id} with {len(specs)} models")

    @classmethod
    def get_plugin(cls, family_id: str) -> Optional[BaseFamilyPlugin]:
        """Get a family plugin by ID."""
        return cls._plugins.get(family_id)

    @classmethod
    def get_model(cls, model_id: str) -> Optional[ModelSpec]:
        """Get a model spec by full ID (family:name)."""
        return cls._models.get(model_id)

    @classmethod
    def get_model_by_name(cls, name: str, family_id: Optional[str] = None) -> Optional[ModelSpec]:
        """Get a model spec by short name, optionally filtered by family."""
        if family_id:
            return cls._models.get(f"{family_id}:{name}")

        # Search all families
        for model_id, spec in cls._models.items():
            if spec.name == name:
                return spec
        return None

    @classmethod
    def list_families(cls) -> List[str]:
        """List all registered family IDs."""
        return list(cls._plugins.keys())

    @classmethod
    def list_models(
        cls,
        family_id: Optional[str] = None,
        modality: Optional[Modality] = None,
        task_type: Optional[TaskType] = None,
        edge_only: bool = False,
    ) -> List[ModelSpec]:
        """List models matching the given criteria."""
        results = []
        for spec in cls._models.values():
            if family_id and spec.family_id != family_id:
                continue
            if modality and spec.modality != modality:
                continue
            if task_type and task_type not in spec.supported_tasks:
                continue
            if edge_only and not spec.edge_compatible:
                continue
            results.append(spec)
        return sorted(results, key=lambda s: (s.family_id, s.params_m))

    @classmethod
    def render_train_script(cls, context: TemplateContext) -> str:
        """Generate train.py content using the appropriate family plugin."""
        plugin = cls._plugins.get(context.model_spec.family_id)
        if not plugin:
            logger.warning(f"FamilyRegistry: no plugin for family {context.model_spec.family_id}")
            return ""
        return plugin.render_train_script(context)

    @classmethod
    def render_infer_script(cls, context: TemplateContext) -> str:
        """Generate infer.py content using the appropriate family plugin."""
        plugin = cls._plugins.get(context.model_spec.family_id)
        if not plugin:
            logger.warning(f"FamilyRegistry: no plugin for family {context.model_spec.family_id}")
            return ""
        return plugin.render_infer_script(context)

    @classmethod
    def render_export_script(cls, context: TemplateContext) -> str:
        """Generate export.py content using the appropriate family plugin."""
        plugin = cls._plugins.get(context.model_spec.family_id)
        if not plugin:
            return ""
        return plugin.render_export_script(context)

    @classmethod
    def get_search_space_description(
        cls,
        modality: Modality,
        task_type: TaskType,
    ) -> str:
        """Generate search space description for LLM prompts."""
        models = cls.list_models(modality=modality, task_type=task_type)
        if not models:
            return "No models registered for this modality/task combination."

        lines = [f"Available models for {modality.value}/{task_type.value}:"]

        # Group by family
        families: Dict[str, List[ModelSpec]] = {}
        for spec in models:
            families.setdefault(spec.family_id, []).append(spec)

        for family_id, family_models in sorted(families.items()):
            names = []
            for model in family_models:
                detail = f"{model.name} ({model.params_m:.1f}M"
                if model.pretrained_weights:
                    detail += f", pretrained={model.pretrained_weights}"
                if model.supported_export_formats:
                    formats = "/".join(
                        item.value if hasattr(item, "value") else str(item)
                        for item in model.supported_export_formats
                    )
                    detail += f", export={formats}"
                names.append(detail + ")")
            lines.append(f"  {family_id}: {', '.join(names)}")

        # Add common configuration axes
        lines.append("")
        lines.append("Configuration axes:")
        lines.append("  quant_mode: [fp32, fp16, int8]")
        lines.append("  export_format: [onnx, engine, pt]")

        # Add modality-specific hints
        if modality == Modality.VISION:
            lines.append("  imgsz: [320, 416, 512, 640] (in train_code)")
            lines.append("  epochs, batch_size, lr, optimizer (in train_code)")
        elif modality == Modality.AUDIO:
            lines.append("  sample_rate, window_size, hop_length (in train_code)")
            lines.append(
                "  ASR policy: prefer tiny_audio_asr/tiny_char_ctc_asr for "
                "artifact-first edge runs (<10M params); avoid Whisper-sized "
                "seq2seq models unless the user explicitly asks for Whisper."
            )
        elif modality == Modality.TEXT:
            lines.append("  max_length, batch_size, learning_rate (in train_code)")

        return "\n".join(lines)

    @classmethod
    def public_solution_zoo_description(
        cls,
        modality: Modality,
        task_type: TaskType,
        *,
        runtime_config: Optional[RuntimeConfig] = None,
        parent_model_name: Optional[str] = None,
        parent_family_id: Optional[str] = None,
        has_executable_parent: bool = False,
    ) -> str:
        """Render a task-compatible public solution catalog for the LLM.

        Static entries describe public model/preprocessing capabilities.  Live
        preflight facts only remove paths known to be impossible in the current
        cloud or target runtime.  Dataset identity and benchmark outcomes never
        participate in selection.
        """
        paths = [
            path
            for path in _PUBLIC_SOLUTION_PATHS.get((modality, task_type), [])
            if _packages_are_possible(list(path.get("packages") or []), runtime_config)
        ]
        models = [
            spec
            for spec in cls.list_models(
                modality=modality,
                task_type=task_type,
                edge_only=True,
            )
            if _packages_are_possible(spec.pip_packages, runtime_config)
            and _model_has_runtime_path(spec, runtime_config)
        ]

        model_limit = 4 if has_executable_parent else 8
        representatives = _representative_models(models, limit=model_limit)
        parent = cls.get_model_by_name(parent_model_name, parent_family_id) if parent_model_name else None
        if parent is not None and parent not in representatives:
            representatives = [parent, *representatives[: max(0, model_limit - 1)]]

        path_limit = 3 if has_executable_parent else 6
        paths = paths[:path_limit]
        lines = [
            f"Public solution zoo for {modality.value}/{task_type.value}.",
            "These are task-compatible public priors, not dataset-specific recommendations or code templates.",
            "Full training must use a validation-bearing multi-epoch recipe and must not report success after a non-finite loss; one-epoch and sampled runs belong only to probes.",
            "Keep full-data preprocessing inside the stage budget: do not rescan the same large raw sources independently for each split or epoch. Materialize split-aligned features once per invocation, or use streaming/chunked reads when the observed data are large.",
            "Derived windows, patches, and grouped records must preserve evaluator-owned sample identity: every emitted source_id must equal an opaque ID in the requested split. A namespaced ID such as field:value may use its field and value only for a verified join to raw data; keep the original full ID when writing evaluation provenance.",
            "Treat the observed loader output as an executable interface: inspect and preserve its actual item keys and tuple structure across train.py and infer.py, rather than inventing aliases such as id versus sample_id.",
            "Apply evaluator split or group membership before any max_samples or raw-row truncation. For grouped data, scan enough source data to find the requested groups, construct their examples, and only then cap the returned examples.",
            "An efficiency probe may skip optimization but not the data and evaluation contract: use the same loader and non-empty validation IDs, copy split_manifest.content_hash exactly, and declare the resulting edge_eval_manifest and edge_eval_payload in artifact_paths.",
            "When preflight proves that the selected edge runtime exposes an accelerator, place the model and every runtime input on that same device and benchmark that actual path. CPU execution is valid only when deliberately selected or when the runtime has no usable accelerator.",
            "Treat export compatibility as part of the solution: honor target-runtime versions from preflight, and use PyTorch's legacy ONNX exporter (dynamo=False) when the default exporter would emit a newer ONNX IR or opset than the target can load.",
        ]
        if has_executable_parent:
            lines.append(
                "For this child, preserve the parent by default; use an alternative below only when the "
                "verification-grounded trial record justifies a coherent recipe pivot."
            )
        else:
            lines.append(
                "For planning roots, form a diverse portfolio of complete hypotheses. Include a mature "
                "public model route when compatible, and avoid repeating a representation/model route "
                "already used by a sibling root."
            )
            lines.append(
                "Across the root portfolio, cover a quality-oriented mature route that can plausibly meet "
                "the hard constraints, a compact route with resource slack, and a structurally different "
                "quality-efficiency balance; these are starting hypotheses, not fixed model choices."
            )
        lines.append(
            "A public path is a complete starting recipe, not a model-name hint: keep its representation, "
            "preprocessing, initialization, optimization budget, and validation policy together. The LLM may "
            "adapt that recipe to observed constraints, but must state why it is weakening a mature default."
        )
        lines.append(
            "For a full solution, use a task-capable checkpoint with established public pretraining. "
            "Testing-only or random checkpoints (for example tiny-random and hf-internal-testing assets) "
            "may support a contract smoke probe, but are not a quality-bearing initialization."
        )

        if paths:
            lines.append("\nTask-compatible solution paths:")
        for path in paths:
            rendered = (
                "- {id}: representation={representation}; preprocessing={preprocessing}; "
                "public_models={models}; initialization={initialization}; "
                "training={training}.".format(**path)
            )
            if path.get("deployment"):
                rendered += f" deployment={path['deployment']}."
            lines.append(rendered)

        if representatives:
            lines.append("\nEnvironment-compatible registered models:")
        for spec in representatives:
            formats = "/".join(
                item.value if hasattr(item, "value") else str(item)
                for item in spec.supported_export_formats
            )
            runtimes = "/".join(
                item.value if hasattr(item, "value") else str(item)
                for item in spec.supported_runtimes
            )
            pretrained = spec.pretrained_weights or "none"
            lines.append(
                f"- {spec.model_id}: {spec.params_m:.1f}M params; pretrained={pretrained}; "
                f"export={formats}; runtime={runtimes}."
            )

        if not paths and not representatives:
            lines.append(
                "No prequalified public path matched. The LLM may still propose a compact implementation "
                "from installed public libraries, but must state its input, preprocessing, model, and export contracts."
            )
        return "\n".join(lines)

    @classmethod
    def get_search_neighbors(
        cls,
        model_id: str,
        direction: str = "similar",
    ) -> List[str]:
        """Get neighboring models for search exploration."""
        spec = cls._models.get(model_id)
        if not spec:
            return []

        plugin = cls._plugins.get(spec.family_id)
        if not plugin:
            return []

        return plugin.get_search_neighbors(spec.name, direction)

    @classmethod
    def get_default_hyperparams(cls, model_id: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model."""
        spec = cls._models.get(model_id)
        if spec:
            return dict(spec.default_hyperparams)
        return {"epochs": 50, "batch_size": 16, "lr": 0.01}

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get registry statistics."""
        modality_counts = {}
        family_counts = {}

        for spec in cls._models.values():
            modality_counts[spec.modality.value] = modality_counts.get(spec.modality.value, 0) + 1
            family_counts[spec.family_id] = family_counts.get(spec.family_id, 0) + 1

        return {
            "total_families": len(cls._plugins),
            "total_models": len(cls._models),
            "models_by_modality": modality_counts,
            "models_by_family": family_counts,
        }

    @classmethod
    def clear(cls) -> None:
        """Clear registered plugins/models (used by explicit init)."""
        cls._plugins.clear()
        cls._models.clear()

    @classmethod
    def search_space_description(cls, modality: Modality, task_type: TaskType) -> str:
        """Compatibility wrapper for older call sites."""
        return cls.get_search_space_description(modality, task_type)

    @classmethod
    def get_train_template(
        cls,
        family_id: str,
        task_type: TaskType,
        model_name: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compatibility helper to generate train script text."""
        spec = cls.get_model_by_name(model_name, family_id) if model_name else None
        if spec is None:
            candidates = cls.list_models(family_id=family_id, task_type=task_type)
            spec = candidates[0] if candidates else None
        if spec is None:
            return ""
        context = TemplateContext(
            model_spec=spec,
            task_type=task_type,
            runtime_id=RuntimeId.PYTORCH,
            device_class=DeviceClass.JETSON,
            export_format=ExportFormat.ONNX,
            quant_mode=QuantMode.FP16,
            dataset_path="config/data.yaml",
            hyperparams=hyperparams or {},
        )
        return cls.render_train_script(context)

    @classmethod
    def get_infer_template(
        cls,
        family_id: str,
        task_type: TaskType,
        model_name: Optional[str] = None,
        export_format: str = "onnx",
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compatibility helper to generate infer script text."""
        spec = cls.get_model_by_name(model_name, family_id) if model_name else None
        if spec is None:
            candidates = cls.list_models(family_id=family_id, task_type=task_type)
            spec = candidates[0] if candidates else None
        if spec is None:
            return ""
        fmt = {
            "onnx": ExportFormat.ONNX,
            "engine": ExportFormat.ENGINE,
            "pt": ExportFormat.PT,
            "tflite": ExportFormat.TFLITE,
        }.get(export_format, ExportFormat.ONNX)
        context = TemplateContext(
            model_spec=spec,
            task_type=task_type,
            runtime_id=RuntimeId.PYTORCH,
            device_class=DeviceClass.JETSON,
            export_format=fmt,
            quant_mode=QuantMode.FP16,
            dataset_path="config/data.yaml",
            hyperparams=hyperparams or {},
        )
        return cls.render_infer_script(context)

    @classmethod
    def estimate_latency(
        cls,
        model_name: str,
        model_family: str,
        device_id: str,
        quant_mode: str,
        imgsz: int = 640,
    ) -> tuple[float, float]:
        """Heuristic latency estimate when no validated profile is available."""
        spec = cls.get_model_by_name(model_name, model_family)
        if not spec:
            return 0.0, 0.0
        ref = max(spec.params_m, 0.5)
        # Rough baseline on the reference Jetson profile for FP16 @ 640.
        base_ms = 6.0 + (ref * 2.2)
        q_scale = {"fp32": 1.5, "fp16": 1.0, "int8": 0.7}.get(quant_mode, 1.0)
        size_scale = (imgsz / 640.0) ** 2 if imgsz > 0 else 1.0
        d_scale = DeviceRegistry.get_latency_scale(device_id)
        return base_ms * q_scale * size_scale * d_scale, 0.35

    @classmethod
    def estimate_memory(
        cls,
        model_name: str,
        model_family: str,
        quant_mode: str,
    ) -> float:
        """Heuristic memory estimate in MB."""
        spec = cls.get_model_by_name(model_name, model_family)
        if not spec:
            return 0.0
        # Very rough estimate from params (1M params ~= 4MB FP32).
        fp32_mb = max(spec.params_m, 0.5) * 4.0
        q_scale = {"fp32": 1.0, "fp16": 0.55, "int8": 0.30}.get(quant_mode, 0.55)
        return fp32_mb * q_scale * 8.0


# ---------------------------------------------------------------------------
# DeviceRegistry: Registry for device specifications
# ---------------------------------------------------------------------------

class DeviceRegistry:
    """Registry for device specifications."""

    from edgecraft.models.specs import DeviceSpec

    _devices: Dict[str, "DeviceSpec"] = {}

    @classmethod
    def register(cls, spec: "DeviceSpec") -> None:
        """Register a device specification."""
        cls._devices[spec.device_id] = spec
        logger.trace(f"DeviceRegistry: registered device {spec.device_id}")

    @classmethod
    def get(cls, device_id: str) -> Optional["DeviceSpec"]:
        """Get a device specification by ID."""
        return cls._devices.get(device_id)

    @classmethod
    def list_devices(
        cls,
        device_class: Optional[DeviceClass] = None,
        has_gpu: Optional[bool] = None,
    ) -> List["DeviceSpec"]:
        """List devices matching criteria."""
        results = []
        for spec in cls._devices.values():
            if device_class and spec.device_class != device_class:
                continue
            if has_gpu is not None and spec.has_gpu != has_gpu:
                continue
            results.append(spec)
        return results

    @classmethod
    def get_latency_scale(cls, device_id: str) -> float:
        """Get latency scale factor for a device."""
        spec = cls.get(device_id)
        return spec.latency_scale if spec else 1.0


# ---------------------------------------------------------------------------
# RuntimeRegistry: Registry for runtime specifications
# ---------------------------------------------------------------------------

class RuntimeRegistry:
    """Registry for runtime specifications."""

    from edgecraft.models.specs import RuntimeSpec

    _runtimes: Dict[RuntimeId, "RuntimeSpec"] = {}

    @classmethod
    def register(cls, spec: "RuntimeSpec") -> None:
        """Register a runtime specification."""
        cls._runtimes[spec.runtime_id] = spec
        logger.trace(f"RuntimeRegistry: registered runtime {spec.runtime_id.value}")

    @classmethod
    def get(cls, runtime_id: RuntimeId) -> Optional["RuntimeSpec"]:
        """Get a runtime specification by ID."""
        return cls._runtimes.get(runtime_id)

    @classmethod
    def list_runtimes(
        cls,
        device_class: Optional[DeviceClass] = None,
    ) -> List["RuntimeSpec"]:
        """List runtimes, optionally filtered by device compatibility."""
        results = []
        for spec in cls._runtimes.values():
            if device_class and device_class not in spec.supported_device_classes:
                continue
            results.append(spec)
        return results

    @classmethod
    def get_compatible_runtimes(
        cls,
        export_format: ExportFormat,
        device_class: Optional[DeviceClass] = None,
    ) -> List["RuntimeSpec"]:
        """Get runtimes compatible with an export format."""
        results = []
        for spec in cls._runtimes.values():
            if export_format not in spec.supported_formats:
                continue
            if device_class and device_class not in spec.supported_device_classes:
                continue
            results.append(spec)
        return results
