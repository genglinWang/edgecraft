"""Benchmark backends for different modalities.

Each backend implements the BaseBenchmark interface to provide
consistent benchmarking across different model types and frameworks.

Available backends:
- ultralytics: YOLO models (.pt, .onnx, .engine)
- onnx: Generic ONNX Runtime inference

Future backends (placeholder):
- transformers: HuggingFace models for NLP
- whisper: Audio transcription models
"""

from .base import BaseBenchmark
from .ultralytics_backend import UltralyticsBackend
from .onnx_backend import ONNXBackend

__all__ = ["BaseBenchmark", "UltralyticsBackend", "ONNXBackend"]
