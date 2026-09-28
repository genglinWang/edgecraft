"""CLI utilities for EdgeCraft."""
import os
import sys

# Suppress ONNX Runtime TensorRT/CUDA provider warnings BEFORE any imports
# ChromaDB uses ONNX Runtime for embeddings which tries to load TensorRT
os.environ["ORT_LOGGING_LEVEL"] = "3"  # ERROR only

# Redirect stderr temporarily to suppress the C++ level warnings
# that can't be suppressed via Python API
import io
import contextlib

class _SuppressONNXWarnings:
    """Context manager to suppress ONNX Runtime TensorRT warnings."""

    def __init__(self):
        self._original_stderr = None
        self._devnull = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

# Try to configure onnxruntime before chromadb imports it
try:
    import onnxruntime as ort
    # Set to only show errors (severity 3)
    ort.set_default_logger_severity(3)
    # Disable telemetry
    ort.disable_telemetry_events()
except ImportError:
    pass

from edgecraft.cli.commands import main

__all__ = ["main"]
