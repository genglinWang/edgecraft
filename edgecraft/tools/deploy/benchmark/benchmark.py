#!/usr/bin/env python3
"""Edge inference benchmark script.

This script runs on edge devices inside Docker containers.
It supports multiple backends for different model types and modalities.

Usage:
    python benchmark.py --model model.pt --backend ultralytics --iterations 100

The script outputs a JSON file with benchmark metrics that can be
collected by the EdgeCraft controller.
"""

import argparse
import json
import sys
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Type

# Backend registry - add new backends here
# Format: "backend_name": ("module_path", "ClassName")
BACKENDS: Dict[str, tuple] = {
    "ultralytics": ("backends.ultralytics_backend", "UltralyticsBackend"),
    "onnx": ("backends.onnx_backend", "ONNXBackend"),
    # Future backends (uncomment when implemented):
    # "transformers": ("backends.transformers_backend", "TransformersBackend"),
    # "whisper": ("backends.whisper_backend", "WhisperBackend"),
}


def get_backend_class(name: str) -> Type:
    """Dynamically load a backend class by name.

    Args:
        name: Backend name (e.g., 'ultralytics', 'onnx').

    Returns:
        Backend class (not instance).

    Raises:
        ValueError: If backend name is unknown.
        ImportError: If backend module cannot be imported.
    """
    if name not in BACKENDS:
        available = list(BACKENDS.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    module_path, class_name = BACKENDS[name]

    # Handle both absolute and relative imports
    try:
        # Try relative import (when running as part of edgecraft package)
        module = __import__(module_path, fromlist=[class_name])
    except ImportError:
        # Try importing from current directory (when running standalone on edge)
        import importlib.util
        import os

        # Construct full path to module file
        script_dir = Path(__file__).parent
        module_file = script_dir / f"{module_path.replace('.', '/')}.py"

        if module_file.exists():
            spec = importlib.util.spec_from_file_location(module_path, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            raise ImportError(f"Cannot import backend module: {module_path}")

    return getattr(module, class_name)


def get_device_info() -> Dict[str, Any]:
    """Collect device information.

    Returns:
        Dict with hostname, platform info, and optional GPU info.
    """
    import platform

    info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    # Try to get memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["memory_total_mb"] = mem.total // (1024 * 1024)
        info["memory_available_mb"] = mem.available // (1024 * 1024)
    except ImportError:
        pass

    return info


def run_benchmark(
    model_path: str,
    backend_name: str = "ultralytics",
    iterations: int = 100,
    warmup: int = 10,
    output_path: str = "metrics.json",
    **kwargs
) -> Dict[str, Any]:
    """Run benchmark and save results.

    Args:
        model_path: Path to the model file.
        backend_name: Name of the backend to use.
        iterations: Number of timed inference runs.
        warmup: Number of warmup runs.
        output_path: Path to save JSON results.
        **kwargs: Backend-specific arguments (e.g., imgsz).

    Returns:
        Dict with benchmark results.
    """
    # Load backend
    BackendClass = get_backend_class(backend_name)
    backend = BackendClass()

    # Load model
    print(f"[benchmark] Loading model: {model_path}")
    print(f"[benchmark] Backend: {backend_name}")
    backend.load_model(model_path, **kwargs)

    # Run benchmark
    print(f"[benchmark] Running {warmup} warmup iterations...")
    print(f"[benchmark] Running {iterations} benchmark iterations...")

    metrics = backend.benchmark(iterations=iterations, warmup=warmup, **kwargs)

    # Build result
    result = {
        "status": "success",
        "backend": backend_name,
        "model_path": str(model_path),
        "model_format": Path(model_path).suffix.lstrip("."),
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            **{k: v for k, v in kwargs.items() if v is not None}
        },
        "metrics": metrics,
        "device": get_device_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[benchmark] Results saved to: {output_path}")

    return result


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Run inference benchmark on edge device",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the model file"
    )
    parser.add_argument(
        "--backend", "-b",
        default="ultralytics",
        choices=list(BACKENDS.keys()),
        help="Inference backend to use"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--output", "-o",
        default="metrics.json",
        help="Output file path for results"
    )

    # Backend-specific arguments
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (for vision models)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length (for NLP models)"
    )

    args = parser.parse_args()

    try:
        result = run_benchmark(
            model_path=args.model,
            backend_name=args.backend,
            iterations=args.iterations,
            warmup=args.warmup,
            output_path=args.output,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
        )

        # Print summary to stdout
        print("\n" + "=" * 50)
        print("[benchmark] RESULTS")
        print("=" * 50)
        print(json.dumps(result, indent=2))

        return 0

    except Exception as e:
        # Write error result
        error_result = {
            "status": "error",
            "error": str(e),
            "backend": args.backend,
            "model_path": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(args.output, "w") as f:
            json.dump(error_result, f, indent=2)

        print(f"[benchmark] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
