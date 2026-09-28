"""Family plugins for the EdgeCraft model zoo.

This package contains family-level plugins that provide:
- Model specifications (ModelSpec)
- Script generation for train.py/infer.py
- Search neighbor information for MCTS exploration

Available families (current):
- ultralytics: YOLO11, YOLOv8, RT-DETR (vision/detection)
- timm: MobileNet, EfficientNet, ResNet, EfficientViT, etc. (vision/classification)
- torchvision: SSD-Lite, Faster R-CNN, MobileNet (vision/detection+classification)
- transformers: BERT, DistilBERT, MobileBERT, ALBERT (text)
- structured_log: supervised structured/log anomaly baselines
- tiny_audio_asr: tiny character CTC ASR baseline (audio/ASR)
- whisper: Whisper ASR models (audio/ASR)
- speechbrain: ECAPA-TDNN, X-Vector (audio/speaker)

Planned families (see model_sources_roadmap.py):
- tflite_models: TFLite Model Hub models (mobile)
- mediapipe: MediaPipe Solutions models (mobile)
- onnx_zoo: ONNX Model Zoo (cross-platform)
"""
from edgecraft.models.families.ultralytics_plugin import UltralyticsPlugin, ultralytics_plugin
from edgecraft.models.families.timm_plugin import TimmPlugin, timm_plugin
from edgecraft.models.families.torchvision_plugin import TorchVisionPlugin, torchvision_plugin
from edgecraft.models.families.transformers_plugin import TransformersPlugin, transformers_plugin
from edgecraft.models.families.structured_log_plugin import StructuredLogPlugin, structured_log_plugin
from edgecraft.models.families.anomaly_detection_plugin import (
    AnomalyDetectionPlugin,
    anomaly_detection_plugin,
)
from edgecraft.models.families.crowd_counting_plugin import (
    CrowdCountingPlugin,
    crowd_counting_plugin,
)
from edgecraft.models.families.audio_plugins import (
    TinyAudioASRPlugin,
    tiny_audio_asr_plugin,
    WhisperPlugin,
    whisper_plugin,
    SpeechBrainPlugin,
    speechbrain_plugin,
)

__all__ = [
    # Vision
    "UltralyticsPlugin",
    "ultralytics_plugin",
    "TimmPlugin",
    "timm_plugin",
    "TorchVisionPlugin",
    "torchvision_plugin",
    # Text
    "TransformersPlugin",
    "transformers_plugin",
    "StructuredLogPlugin",
    "structured_log_plugin",
    "AnomalyDetectionPlugin",
    "anomaly_detection_plugin",
    "CrowdCountingPlugin",
    "crowd_counting_plugin",
    # Audio
    "TinyAudioASRPlugin",
    "tiny_audio_asr_plugin",
    "WhisperPlugin",
    "whisper_plugin",
    "SpeechBrainPlugin",
    "speechbrain_plugin",
]
