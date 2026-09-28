"""Generic ONNX Runtime benchmark backend.

This backend provides a modality-agnostic way to benchmark any ONNX model.
It automatically detects input shapes and data types from the model metadata.

Useful for:
- Models exported from any framework to ONNX
- Custom models not covered by specialized backends
- CPU inference on devices without GPU support (e.g., Raspberry Pi)
"""

from typing import Any, List, Dict
import numpy as np

from .base import BaseBenchmark


class ONNXBackend(BaseBenchmark):
    """Benchmark backend for generic ONNX models.

    Automatically adapts to model input requirements by reading
    ONNX metadata. Supports dynamic batch sizes.

    Example:
        >>> backend = ONNXBackend()
        >>> backend.load_model("model.onnx")
        >>> metrics = backend.benchmark(iterations=100, warmup=10)
    """

    name = "onnx"
    description = "Generic ONNX Runtime inference (any modality)"
    supported_formats = ["onnx"]

    def __init__(self):
        """Initialize the backend."""
        self.session = None
        self.input_info: List[Dict[str, Any]] = []
        self.output_names: List[str] = []

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load an ONNX model.

        Args:
            model_path: Path to the .onnx model file.
            **kwargs: Additional arguments (e.g., providers for GPU/CPU).
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is not installed. "
                "Install it with: pip install onnxruntime"
            )

        # Configure providers (prefer GPU if available)
        providers = kwargs.get("providers", None)
        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "TensorrtExecutionProvider" in available:
                providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Extract input metadata
        self.input_info = []
        for inp in self.session.get_inputs():
            self.input_info.append({
                "name": inp.name,
                "shape": inp.shape,
                "dtype": inp.type,
            })

        # Extract output names
        self.output_names = [out.name for out in self.session.get_outputs()]

    def _onnx_dtype_to_numpy(self, onnx_type: str) -> np.dtype:
        """Convert ONNX type string to numpy dtype.

        Args:
            onnx_type: ONNX type string (e.g., 'tensor(float)').

        Returns:
            Corresponding numpy dtype.
        """
        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        return type_map.get(onnx_type, np.float32)

    def _resolve_shape(self, shape: List, **kwargs) -> List[int]:
        """Resolve dynamic dimensions in shape.

        Args:
            shape: Shape list that may contain strings or None for dynamic dims.
            **kwargs: May contain 'batch_size', 'seq_length', 'imgsz'.

        Returns:
            Shape with all dimensions resolved to integers.
        """
        batch_size = kwargs.get("batch_size", 1)
        seq_length = kwargs.get("seq_length", 128)
        imgsz = kwargs.get("imgsz", 640)

        resolved = []
        for i, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                resolved.append(dim)
            elif dim is None or isinstance(dim, str):
                # Dynamic dimension - use sensible defaults
                if i == 0:  # Usually batch dimension
                    resolved.append(batch_size)
                elif i == 1 and len(shape) == 4:  # Channels
                    resolved.append(3)
                elif len(shape) == 4:  # Spatial dims for images
                    resolved.append(imgsz)
                else:  # Sequence length for NLP
                    resolved.append(seq_length)
            else:
                resolved.append(1)

        return resolved

    def create_dummy_input(self, **kwargs) -> Dict[str, np.ndarray]:
        """Create dummy inputs matching model requirements.

        Args:
            **kwargs: Shape overrides (batch_size, seq_length, imgsz).

        Returns:
            Dict mapping input names to numpy arrays.
        """
        inputs = {}
        for info in self.input_info:
            shape = self._resolve_shape(info["shape"], **kwargs)
            dtype = self._onnx_dtype_to_numpy(info["dtype"])

            # Generate appropriate random data
            if dtype in [np.float32, np.float16, np.float64]:
                data = np.random.randn(*shape).astype(dtype)
            elif dtype in [np.int64, np.int32]:
                data = np.random.randint(0, 1000, shape, dtype=dtype)
            elif dtype == np.uint8:
                data = np.random.randint(0, 255, shape, dtype=dtype)
            else:
                data = np.zeros(shape, dtype=dtype)

            inputs[info["name"]] = data

        return inputs

    def run_inference(self, input_data: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run ONNX inference.

        Args:
            input_data: Dict mapping input names to numpy arrays.

        Returns:
            List of output arrays.
        """
        return self.session.run(self.output_names, input_data)

    def get_info(self) -> dict:
        """Get backend information including model metadata."""
        info = super().get_info()
        if self.session is not None:
            info.update({
                "inputs": self.input_info,
                "outputs": self.output_names,
                "providers": self.session.get_providers(),
            })
        return info
