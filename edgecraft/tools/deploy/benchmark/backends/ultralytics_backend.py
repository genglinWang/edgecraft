"""Ultralytics YOLO benchmark backend for computer vision tasks.

This backend supports YOLO models in various formats:
- PyTorch (.pt)
- ONNX (.onnx)
- TensorRT (.engine)

The ultralytics library automatically handles format detection and
uses the appropriate inference backend.
"""

from typing import Any, List
import numpy as np

from .base import BaseBenchmark


class UltralyticsBackend(BaseBenchmark):
    """Benchmark backend for Ultralytics YOLO models.

    Supports object detection, segmentation, classification, and pose
    estimation tasks using YOLO models.

    Example:
        >>> backend = UltralyticsBackend()
        >>> backend.load_model("yolo11n.pt", imgsz=640)
        >>> metrics = backend.benchmark(iterations=100, warmup=10)
    """

    name = "ultralytics"
    description = "Ultralytics YOLO models (detection, segmentation, classification, pose)"
    supported_formats = ["pt", "onnx", "engine", "torchscript"]

    def __init__(self):
        """Initialize the backend."""
        self.model = None
        self.imgsz = 640
        self.task = None

    def load_model(self, model_path: str, imgsz: int = 640, **kwargs) -> None:
        """Load a YOLO model.

        Args:
            model_path: Path to the model file (.pt, .onnx, or .engine).
            imgsz: Input image size (default: 640).
            **kwargs: Additional arguments passed to YOLO().
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. "
                "Install it with: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.task = self.model.task

    def create_dummy_input(self, **kwargs) -> np.ndarray:
        """Create a random image for benchmarking.

        Args:
            **kwargs: May contain 'imgsz' to override default size.

        Returns:
            Random RGB image as numpy array (H, W, 3).
        """
        imgsz = kwargs.get("imgsz", self.imgsz)
        # Create random RGB image
        return np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    def run_inference(self, input_data: np.ndarray) -> Any:
        """Run YOLO inference on the input image.

        Args:
            input_data: Input image as numpy array.

        Returns:
            YOLO Results object.
        """
        # verbose=False suppresses per-inference logging
        return self.model.predict(input_data, verbose=False, imgsz=self.imgsz)

    def get_info(self) -> dict:
        """Get backend information including model details."""
        info = super().get_info()
        if self.model is not None:
            info.update({
                "task": self.task,
                "imgsz": self.imgsz,
                "model_type": type(self.model).__name__,
            })
        return info
