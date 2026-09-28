"""Base classes for deployment tools."""
from abc import ABC, abstractmethod
from typing import Dict, Any
from edgecraft.tools.base import BaseTool


class BaseExporter(BaseTool, ABC):
    """Base class for model exporters."""

    @abstractmethod
    def export(
        self,
        model_path: str,
        output_path: str,
        format: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Export model to deployment format.

        Args:
            model_path: Path to trained model.
            output_path: Output path for exported model.
            format: Export format (onnx, engine, tflite, etc.).

        Returns:
            Dict with export results.
        """
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        return self.export(**kwargs)


class BaseDeployer(BaseTool, ABC):
    """Base class for model deployers."""

    @abstractmethod
    def deploy(
        self,
        artifact_path: str,
        device_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy artifact to target device.

        Args:
            artifact_path: Path to deployment artifact.
            device_id: Target device identifier.

        Returns:
            Dict with deployment results.
        """
        pass

    @abstractmethod
    def run_benchmark(
        self,
        device_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run benchmark on deployed model.

        Args:
            device_id: Device to benchmark on.

        Returns:
            Dict with benchmark results (latency, throughput, etc.).
        """
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        return self.deploy(**kwargs)
