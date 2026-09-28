"""Base class for all benchmark backends.

All modality-specific backends (vision, NLP, audio, etc.) should inherit
from BaseBenchmark and implement the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import time


class BaseBenchmark(ABC):
    """Abstract base class for inference benchmarking.

    This class defines the interface that all benchmark backends must implement.
    The benchmark() method provides common timing logic shared across all modalities.

    Subclasses must implement:
        - load_model(): Load the model from disk
        - create_dummy_input(): Generate appropriate test input
        - run_inference(): Execute single inference pass

    Example:
        >>> backend = UltralyticsBackend()
        >>> backend.load_model("yolo11n.pt", imgsz=640)
        >>> metrics = backend.benchmark(iterations=100, warmup=10)
        >>> print(metrics["latency_avg_ms"])
    """

    # Backend metadata - subclasses should override
    name: str = "base"
    description: str = "Base benchmark backend"
    supported_formats: List[str] = []

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load model from the given path.

        Args:
            model_path: Path to the model file.
            **kwargs: Backend-specific arguments (e.g., imgsz for vision).
        """
        pass

    @abstractmethod
    def create_dummy_input(self, **kwargs) -> Any:
        """Create dummy input data for benchmarking.

        This should generate input that matches the model's expected format.
        For vision models: image array
        For NLP models: tokenized text
        For audio models: audio waveform

        Args:
            **kwargs: Backend-specific arguments.

        Returns:
            Input data in the format expected by run_inference().
        """
        pass

    @abstractmethod
    def run_inference(self, input_data: Any) -> Any:
        """Execute a single inference pass.

        Args:
            input_data: Input data from create_dummy_input().

        Returns:
            Model output (format depends on backend).
        """
        pass

    def benchmark(
        self,
        iterations: int = 100,
        warmup: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """Run benchmark and collect timing metrics.

        This method is shared across all backends. It handles:
        1. Warmup runs to stabilize GPU/CPU caches
        2. Timed inference iterations
        3. Statistical aggregation of results

        Args:
            iterations: Number of timed inference runs.
            warmup: Number of warmup runs before timing.
            **kwargs: Passed to create_dummy_input().

        Returns:
            Dict with timing metrics:
                - latency_avg_ms: Mean latency in milliseconds
                - latency_p95_ms: 95th percentile latency
                - latency_p99_ms: 99th percentile latency
                - latency_min_ms: Minimum latency
                - latency_max_ms: Maximum latency
                - throughput_fps: Frames/samples per second
        """
        import numpy as np

        # Create input once (reuse for all iterations)
        dummy_input = self.create_dummy_input(**kwargs)

        # Warmup runs
        for _ in range(warmup):
            self.run_inference(dummy_input)

        # Timed runs
        latencies: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.run_inference(dummy_input)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

        # Calculate statistics
        latencies_np = np.array(latencies)

        return {
            "latency_avg_ms": float(np.mean(latencies_np)),
            "latency_p95_ms": float(np.percentile(latencies_np, 95)),
            "latency_p99_ms": float(np.percentile(latencies_np, 99)),
            "latency_min_ms": float(np.min(latencies_np)),
            "latency_max_ms": float(np.max(latencies_np)),
            "throughput_fps": float(1000.0 / np.mean(latencies_np)),
        }

    def get_info(self) -> Dict[str, Any]:
        """Get backend information.

        Returns:
            Dict with backend metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "supported_formats": self.supported_formats,
        }
