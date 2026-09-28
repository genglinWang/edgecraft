"""EdgeCraft: MCaaS (Model Crafting as a Service) prototype for edge AI."""
import os
import sys

# Keep import-time logging quiet without changing process-wide provider
# discovery. Runtime selection belongs to the benchmark backend that invokes
# ONNX Runtime, not to package initialization.
os.environ["ORT_LOGGING_LEVEL"] = "3"  # ERROR only

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # 3 = ERROR only
except ImportError:
    pass

__version__ = "0.1.0"
