"""TensorRT exporter for vision models."""
from typing import Dict, Any
from pathlib import Path

from edgecraft.tools.deploy.base import BaseExporter
from edgecraft.tools.base import ToolRegistry
from edgecraft.core.modality import Modality


class TensorRTExporter(BaseExporter):
    """Export models to TensorRT for Jetson deployment.

    Note: For YOLO models, use the built-in ultralytics export.
    This class provides additional TensorRT utilities.
    """

    name = "tensorrt_exporter"
    description = "Export models to TensorRT format for Jetson devices"
    modality = Modality.VISION

    def export(
        self,
        model_path: str,
        output_path: str = None,
        format: str = "onnx",  # Default to ONNX (TensorRT built on edge)
        precision: str = "fp16",
        workspace_gb: int = 4,
        **kwargs
    ) -> Dict[str, Any]:
        """Export model to ONNX or TensorRT.

        For YOLO models, this wraps ultralytics export.
        For ONNX models, this uses trtexec or TensorRT Python API.

        Args:
            model_path: Path to input model (.pt or .onnx).
            output_path: Output path for exported file.
            format: Export format (onnx, engine/tensorrt).
            precision: fp32, fp16, or int8.
            workspace_gb: GPU workspace in GB.

        Returns:
            Export results.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            return {
                "status": "error",
                "error": f"Model not found: {model_path}"
            }

        # Determine model type
        suffix = model_path.suffix.lower()

        # ONNX export requested
        if format == "onnx" and suffix == ".pt":
            return self._export_yolo_to_onnx(model_path, output_path, precision, **kwargs)

        if suffix == ".pt":
            # YOLO model - use ultralytics export
            return self._export_yolo(model_path, output_path, precision, **kwargs)
        elif suffix == ".onnx":
            # ONNX model - convert to TensorRT
            return self._export_onnx_to_trt(model_path, output_path, precision, workspace_gb, **kwargs)
        else:
            return {
                "status": "error",
                "error": f"Unsupported model format: {suffix}. Expected .pt or .onnx"
            }

    def _export_yolo_to_onnx(
        self,
        model_path: Path,
        output_path: str,
        precision: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Export YOLO model to ONNX format."""
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))

            export_path = model.export(
                format="onnx",
                half=(precision == "fp16"),
                simplify=True,
                **kwargs
            )

            return {
                "status": "success",
                "export_path": str(export_path),
                "format": "onnx",
                "precision": precision,
                "note": "TensorRT engine will be built on edge device"
            }

        except ImportError:
            return {
                "status": "error",
                "error": "ultralytics not installed"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _export_yolo(
        self,
        model_path: Path,
        output_path: str,
        precision: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Export YOLO model to TensorRT using ultralytics."""
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))

            export_path = model.export(
                format="engine",
                half=(precision == "fp16"),
                int8=(precision == "int8"),
                **kwargs
            )

            return {
                "status": "success",
                "export_path": str(export_path),
                "format": "engine",
                "precision": precision
            }

        except ImportError:
            return {
                "status": "error",
                "error": "ultralytics not installed"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _export_onnx_to_trt(
        self,
        model_path: Path,
        output_path: str,
        precision: str,
        workspace_gb: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Convert ONNX to TensorRT using trtexec or Python API.

        Note: TensorRT engines must be built on the target device
        (or a device with the same GPU architecture).
        """
        if output_path is None:
            output_path = str(model_path.with_suffix(".engine"))

        # Build trtexec command
        # This should be run on the edge device
        trtexec_cmd = f"""
trtexec \\
    --onnx={model_path} \\
    --saveEngine={output_path} \\
    --workspace={workspace_gb * 1024}
"""

        if precision == "fp16":
            trtexec_cmd += "    --fp16 \\\n"
        elif precision == "int8":
            trtexec_cmd += "    --int8 \\\n"

        return {
            "status": "info",
            "message": "TensorRT engines should be built on the target device",
            "trtexec_command": trtexec_cmd.strip(),
            "output_path": output_path,
            "note": "Use EdgeRunner to build the engine on the target Jetson device"
        }


# Register the tool
ToolRegistry.register(TensorRTExporter())
