"""Model Sources Roadmap for the EdgeCraft model zoo.

This module defines the expansion roadmap for the model zoo, organized by:
1. Phase 1: Vision (Python ecosystem) - CURRENT PRIORITY
2. Phase 2: Audio/Text (basic coverage)
3. Phase 3: Mobile/Edge runtimes (future expansion)

==========================================================================
CURRENT STATUS (as of implementation)
==========================================================================

IMPLEMENTED FAMILIES:
┌──────────────────┬────────────────────────────────────────────────────┐
│ Family           │ Models                                              │
├──────────────────┼────────────────────────────────────────────────────┤
│ ultralytics      │ yolo11n, yolo11s, yolo11m, yolo11l, yolo11x        │
│                  │ (5 models, detection/classification/segmentation)  │
├──────────────────┼────────────────────────────────────────────────────┤
│ timm             │ mobilenetv3_small_100, mobilenetv3_large_100,      │
│                  │ efficientnet_b0/b1/b2/b3, resnet18/34/50,          │
│                  │ efficientvit_b0/b1, mobilevit_xxs/xs,              │
│                  │ edgenext_xx_small/x_small                          │
│                  │ (14 models, classification)                         │
├──────────────────┼────────────────────────────────────────────────────┤
│ torchvision      │ ssdlite320_mobilenet_v3_large,                     │
│                  │ fasterrcnn_mobilenet_v3_large_fpn,                 │
│                  │ fasterrcnn_resnet50_fpn_v2, retinanet_resnet50,    │
│                  │ fcos_resnet50_fpn, mobilenet_v2/v3_small/v3_large, │
│                  │ efficientnet_b0/b1, squeezenet1_1                  │
│                  │ (12 models, detection + classification)             │
├──────────────────┼────────────────────────────────────────────────────┤
│ transformers     │ distilbert-base-uncased, mobilebert-uncased,       │
│                  │ albert-base-v2, tinybert-general-4l-312d,          │
│                  │ bert-base-uncased                                   │
│                  │ (5 models, text classification)                     │
├──────────────────┼────────────────────────────────────────────────────┤
│ whisper          │ whisper-tiny, whisper-base, whisper-small          │
│                  │ (3 models, ASR)                                     │
├──────────────────┼────────────────────────────────────────────────────┤
│ speechbrain      │ ecapa-tdnn-voxceleb, xvector-voxceleb,             │
│                  │ crdnn-commands                                      │
│                  │ (3 models, speaker recognition / commands)          │
└──────────────────┴────────────────────────────────────────────────────┘

TOTAL: 6 families, 44 models

==========================================================================
PHASE 1: VISION EXPANSION (HIGH PRIORITY)
==========================================================================

Target: Comprehensive coverage of edge-optimized vision models.

ADDITIONAL MODELS TO ADD (from existing families):

timm additions:
- mobilenetv4_conv_small, mobilenetv4_hybrid_medium (newest MobileNet)
- convnext_tiny, convnext_small (modern ConvNets)
- repvit_m0_9, repvit_m1_0, repvit_m1_1 (RepVGG-style)
- fastervit_0, fastervit_1 (fast vision transformers)
- tf_efficientnet_lite0/1/2 (TF-style EfficientNet for edge)
- mnasnet_small, mnasnet_100 (NAS-designed)

ultralytics additions:
- yolov8n/s/m/l/x (YOLOv8 series)
- yolo-world-s/m/l (open-vocabulary detection)
- rt-detr-l (real-time DETR)
- yolov8-obb (oriented bounding box)

torchvision additions:
- ssd300_vgg16 (classic SSD)
- detr_resnet50 (DETR transformer)

NEW FAMILY: yolo-nas (from Deci AI)
- yolo_nas_s, yolo_nas_m, yolo_nas_l

==========================================================================
PHASE 2: AUDIO/TEXT EXPANSION
==========================================================================

Target: 5-10 foundational models per modality.

AUDIO ADDITIONS:

torchaudio family (new):
- wav2vec2_base, wav2vec2_large (speech representation)
- hubert_base, hubert_large (self-supervised speech)

openai-whisper additions:
- whisper-medium, whisper-large-v3 (higher accuracy, non-edge)

faster-whisper family (new):
- faster-whisper-tiny, faster-whisper-base, faster-whisper-small
- (CTranslate2-accelerated Whisper)

keyword-spotting family (new):
- openwakeword (wake word detection)
- microasr (tiny ASR for commands)

TEXT ADDITIONS:

transformers additions:
- roberta-base (improved BERT)
- xlm-roberta-base (multilingual)
- gpt2-small (generative)
- t5-small (encoder-decoder)

sentence-transformers family (new):
- all-MiniLM-L6-v2 (fast embedding)
- paraphrase-MiniLM-L3-v2 (tiny embedding)
- multi-qa-MiniLM-L6-cos-v1 (QA embedding)

==========================================================================
PHASE 3: MOBILE/EDGE RUNTIME EXPANSION (FUTURE)
==========================================================================

Target: Native mobile runtime support for deployment.

ONNX Model Zoo family (new):
- Models directly from ONNX Model Zoo
- Pre-converted, validated ONNX models
- Focus on inference-only (no training)

TFLite Models family (new):
- Official TensorFlow Lite models
- EfficientDet-Lite series
- MobileNet TFLite variants
- MediaPipe-compatible models

MediaPipe family (new):
- MediaPipe object detection
- MediaPipe pose estimation
- MediaPipe face detection
- MediaPipe hand tracking

ExecuTorch family (new):
- PyTorch models optimized for ExecuTorch
- Focus on iOS/Android deployment

NCNN family (new):
- NCNN-optimized models from nihui/ncnn-models
- Focus on mobile CPU inference

==========================================================================
REGISTRATION PATTERN FOR NEW FAMILIES
==========================================================================

To add a new family, follow this pattern:

1. Create plugin file: edgecraft/models/families/{family}_plugin.py
2. Implement BaseFamilyPlugin:
   - family_id: str
   - display_name: str
   - modalities: List[Modality]
   - get_model_specs() -> List[ModelSpec]
   - render_train_script(context) -> str
   - render_infer_script(context) -> str
3. Add to families/__init__.py
4. Register in models/__init__.py:_init_family_registry()
5. Run profiling for new models

Example skeleton:

```python
class NewFamilyPlugin(BaseFamilyPlugin):
    @property
    def family_id(self) -> str:
        return "newfamily"

    @property
    def display_name(self) -> str:
        return "New Family Display Name"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.VISION]

    def get_model_specs(self) -> List[ModelSpec]:
        return [
            ModelSpec(
                name="model_variant",
                family_id="newfamily",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CLASSIFICATION],
                params_m=5.0,
                ...
            ),
        ]

    def render_train_script(self, context: TemplateContext) -> str:
        # Generate family-native train.py
        return f'''#!/usr/bin/env python3
...
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        # Generate family-native infer.py
        return f'''#!/usr/bin/env python3
...
'''

newfamily_plugin = NewFamilyPlugin()
```

==========================================================================
PROFILING WORKFLOW FOR NEW MODELS
==========================================================================

After adding new models, run offline profiling:

1. Export models to ONNX/Engine format
2. Run OfflineProfiler on target devices:

```python
from edgecraft.models.offline_profiler import OfflineProfiler
from edgecraft.models.profile_store import ProfileRequest

profiler = OfflineProfiler()

request = ProfileRequest(
    model_id="newfamily:model_variant",
    family_id="newfamily",
    task_type="classification",
    device_id="jetson_orin_agx",
    runtime_id=RuntimeId.ONNXRUNTIME,
    quant_mode=QuantMode.FP16,
    export_format=ExportFormat.ONNX,
    input_signature={"imgsz": 224, "batch_size": 1},
)

result = profiler.profile_and_store(request, model_path="/path/to/model.onnx")
```

3. Store results in ProfileStore for surrogate use
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ExpansionPhase(str, Enum):
    """Expansion phase for model additions."""
    PHASE1_VISION = "phase1_vision"
    PHASE2_AUDIO_TEXT = "phase2_audio_text"
    PHASE3_MOBILE_RUNTIME = "phase3_mobile_runtime"


@dataclass
class PlannedModel:
    """A model planned for future addition."""
    name: str
    family_id: str
    phase: ExpansionPhase
    priority: int  # 1 = highest
    notes: str = ""


@dataclass
class PlannedFamily:
    """A family planned for future addition."""
    family_id: str
    display_name: str
    phase: ExpansionPhase
    priority: int
    modalities: List[str]
    notes: str = ""


# Planned models for existing families
PLANNED_MODELS: List[PlannedModel] = [
    # Phase 1: Vision
    PlannedModel("mobilenetv4_conv_small", "timm", ExpansionPhase.PHASE1_VISION, 1),
    PlannedModel("mobilenetv4_hybrid_medium", "timm", ExpansionPhase.PHASE1_VISION, 1),
    PlannedModel("convnext_tiny", "timm", ExpansionPhase.PHASE1_VISION, 2),
    PlannedModel("convnext_small", "timm", ExpansionPhase.PHASE1_VISION, 2),
    PlannedModel("repvit_m0_9", "timm", ExpansionPhase.PHASE1_VISION, 2),
    PlannedModel("repvit_m1_0", "timm", ExpansionPhase.PHASE1_VISION, 2),
    PlannedModel("fastervit_0", "timm", ExpansionPhase.PHASE1_VISION, 3),
    PlannedModel("yolov8n", "ultralytics", ExpansionPhase.PHASE1_VISION, 1),
    PlannedModel("yolov8s", "ultralytics", ExpansionPhase.PHASE1_VISION, 1),
    PlannedModel("rt-detr-l", "ultralytics", ExpansionPhase.PHASE1_VISION, 2),
    # Phase 2: Audio/Text
    PlannedModel("roberta-base", "transformers", ExpansionPhase.PHASE2_AUDIO_TEXT, 2),
    PlannedModel("whisper-medium", "whisper", ExpansionPhase.PHASE2_AUDIO_TEXT, 3),
]

# Planned new families
PLANNED_FAMILIES: List[PlannedFamily] = [
    PlannedFamily(
        "yolo_nas",
        "YOLO-NAS (Deci AI)",
        ExpansionPhase.PHASE1_VISION,
        2,
        ["vision"],
        "Neural Architecture Search optimized YOLO",
    ),
    PlannedFamily(
        "faster_whisper",
        "Faster Whisper (CTranslate2)",
        ExpansionPhase.PHASE2_AUDIO_TEXT,
        2,
        ["audio"],
        "CTranslate2-accelerated Whisper variants",
    ),
    PlannedFamily(
        "sentence_transformers",
        "Sentence Transformers",
        ExpansionPhase.PHASE2_AUDIO_TEXT,
        2,
        ["text"],
        "Lightweight embedding models",
    ),
    PlannedFamily(
        "torchaudio",
        "TorchAudio",
        ExpansionPhase.PHASE2_AUDIO_TEXT,
        3,
        ["audio"],
        "wav2vec2, HuBERT for speech",
    ),
    PlannedFamily(
        "onnx_zoo",
        "ONNX Model Zoo",
        ExpansionPhase.PHASE3_MOBILE_RUNTIME,
        3,
        ["vision", "text"],
        "Pre-converted ONNX models",
    ),
    PlannedFamily(
        "tflite_models",
        "TFLite Model Hub",
        ExpansionPhase.PHASE3_MOBILE_RUNTIME,
        3,
        ["vision"],
        "TensorFlow Lite optimized models",
    ),
    PlannedFamily(
        "mediapipe",
        "MediaPipe",
        ExpansionPhase.PHASE3_MOBILE_RUNTIME,
        3,
        ["vision"],
        "MediaPipe Tasks models",
    ),
]


def get_models_by_phase(phase: ExpansionPhase) -> List[PlannedModel]:
    """Get all planned models for a given phase."""
    return [m for m in PLANNED_MODELS if m.phase == phase]


def get_families_by_phase(phase: ExpansionPhase) -> List[PlannedFamily]:
    """Get all planned families for a given phase."""
    return [f for f in PLANNED_FAMILIES if f.phase == phase]


def get_roadmap_summary() -> Dict[str, int]:
    """Get summary statistics for the roadmap."""
    return {
        "planned_models": len(PLANNED_MODELS),
        "planned_families": len(PLANNED_FAMILIES),
        "phase1_models": len(get_models_by_phase(ExpansionPhase.PHASE1_VISION)),
        "phase2_models": len(get_models_by_phase(ExpansionPhase.PHASE2_AUDIO_TEXT)),
        "phase3_models": len(get_models_by_phase(ExpansionPhase.PHASE3_MOBILE_RUNTIME)),
        "phase1_families": len(get_families_by_phase(ExpansionPhase.PHASE1_VISION)),
        "phase2_families": len(get_families_by_phase(ExpansionPhase.PHASE2_AUDIO_TEXT)),
        "phase3_families": len(get_families_by_phase(ExpansionPhase.PHASE3_MOBILE_RUNTIME)),
    }
