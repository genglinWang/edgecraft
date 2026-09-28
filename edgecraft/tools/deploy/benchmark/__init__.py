"""Edge benchmark module for EdgeCraft.

This module provides extensible inference benchmarking capabilities
for edge devices. It supports multiple modalities (vision, NLP, audio, etc.)
through a plugin-based backend architecture.

MVP Implementation:
- Vision: ultralytics (YOLO) backend
- Generic: ONNX Runtime backend

Future Extensions:
- NLP: transformers backend
- Audio: whisper backend
- Tabular: sklearn/onnx backend

Usage:
    # As a CLI tool (on edge device)
    python benchmark.py --model model.pt --backend ultralytics

    # Programmatically
    from edgecraft.tools.deploy.benchmark import run_benchmark, BACKENDS
    result = run_benchmark("model.pt", backend_name="ultralytics")
"""

# Import from benchmark module
try:
    from .benchmark import run_benchmark, BACKENDS, get_backend_class
except ImportError:
    # May fail if running standalone on edge without full edgecraft package
    run_benchmark = None
    BACKENDS = {}
    get_backend_class = None

# Import backends
from .backends import BaseBenchmark, UltralyticsBackend, ONNXBackend

__all__ = [
    "run_benchmark",
    "BACKENDS",
    "get_backend_class",
    "BaseBenchmark",
    "UltralyticsBackend",
    "ONNXBackend",
]
