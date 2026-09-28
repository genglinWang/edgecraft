"""Ultralytics family plugin for the EdgeCraft model zoo.

This plugin provides model specs and script generation for the Ultralytics
ecosystem (YOLO11, YOLOv8, RT-DETR, etc.).

Key principle: This plugin generates SCRIPTS, it does NOT execute training.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from edgecraft.core.modality import Modality, TaskType
from edgecraft.models.specs import (
    BaseFamilyPlugin,
    DeviceClass,
    ExportFormat,
    ModelSpec,
    QuantMode,
    RuntimeId,
    TemplateContext,
)
from edgecraft.models._tegrastats_code import wrap_with_tegrastats

_DEFAULT_MODEL_CACHE_DIR = str((Path.home() / ".cache" / "edgecraft" / "model_weights").resolve())


class UltralyticsPlugin(BaseFamilyPlugin):
    """Family plugin for Ultralytics models."""

    @property
    def family_id(self) -> str:
        return "ultralytics"

    @property
    def display_name(self) -> str:
        return "Ultralytics (YOLO)"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.VISION]

    def get_model_specs(self) -> List[ModelSpec]:
        """Return all Ultralytics model specs."""
        specs = [
            # YOLO11 Detection/Classification/Segmentation
            ModelSpec(
                name="yolo11n",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[
                    TaskType.OBJECT_DETECTION,
                    TaskType.CLASSIFICATION,
                    TaskType.SEGMENTATION,
                ],
                params_m=2.6,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 16,
                    "lr": 0.01,
                    "imgsz": 640,
                },
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[
                    ExportFormat.PT,
                    ExportFormat.ONNX,
                    ExportFormat.ENGINE,
                ],
                supported_runtimes=[
                    RuntimeId.PYTORCH,
                    RuntimeId.ONNXRUNTIME,
                    RuntimeId.TENSORRT,
                ],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.X86_GPU,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "imgsz": [320, 416, 512, 640],
                    "batch_size": [1],
                },
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolo11n.pt",
                neighbor_models=["yolo11s"],
                lighter_alternative=None,
                heavier_alternative="yolo11s",
            ),
            ModelSpec(
                name="yolo11s",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[
                    TaskType.OBJECT_DETECTION,
                    TaskType.CLASSIFICATION,
                    TaskType.SEGMENTATION,
                ],
                params_m=9.4,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 16,
                    "lr": 0.01,
                    "imgsz": 640,
                },
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[
                    ExportFormat.PT,
                    ExportFormat.ONNX,
                    ExportFormat.ENGINE,
                ],
                supported_runtimes=[
                    RuntimeId.PYTORCH,
                    RuntimeId.ONNXRUNTIME,
                    RuntimeId.TENSORRT,
                ],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [320, 416, 512, 640],
                    "batch_size": [1],
                },
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolo11s.pt",
                neighbor_models=["yolo11n", "yolo11m"],
                lighter_alternative="yolo11n",
                heavier_alternative="yolo11m",
            ),
            ModelSpec(
                name="yolo11m",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[
                    TaskType.OBJECT_DETECTION,
                    TaskType.CLASSIFICATION,
                    TaskType.SEGMENTATION,
                ],
                params_m=20.1,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 50,
                    "batch_size": 16,
                    "lr": 0.01,
                    "imgsz": 640,
                },
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[
                    ExportFormat.PT,
                    ExportFormat.ONNX,
                    ExportFormat.ENGINE,
                ],
                supported_runtimes=[
                    RuntimeId.PYTORCH,
                    RuntimeId.ONNXRUNTIME,
                    RuntimeId.TENSORRT,
                ],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [416, 512, 640],
                    "batch_size": [1],
                },
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolo11m.pt",
                neighbor_models=["yolo11s", "yolo11l"],
                lighter_alternative="yolo11s",
                heavier_alternative="yolo11l",
            ),
            ModelSpec(
                name="yolo11l",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[
                    TaskType.OBJECT_DETECTION,
                    TaskType.CLASSIFICATION,
                    TaskType.SEGMENTATION,
                ],
                params_m=25.3,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 100,
                    "batch_size": 8,
                    "lr": 0.01,
                    "imgsz": 640,
                },
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[
                    ExportFormat.PT,
                    ExportFormat.ONNX,
                    ExportFormat.ENGINE,
                ],
                supported_runtimes=[
                    RuntimeId.PYTORCH,
                    RuntimeId.ONNXRUNTIME,
                    RuntimeId.TENSORRT,
                ],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [416, 512, 640],
                    "batch_size": [1],
                },
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolo11l.pt",
                neighbor_models=["yolo11m", "yolo11x"],
                lighter_alternative="yolo11m",
                heavier_alternative="yolo11x",
            ),
            ModelSpec(
                name="yolo11x",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[
                    TaskType.OBJECT_DETECTION,
                    TaskType.CLASSIFICATION,
                    TaskType.SEGMENTATION,
                ],
                params_m=56.9,
                default_input_shape=[3, 640, 640],
                default_hyperparams={
                    "epochs": 100,
                    "batch_size": 8,
                    "lr": 0.005,
                    "imgsz": 640,
                },
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[
                    ExportFormat.PT,
                    ExportFormat.ONNX,
                    ExportFormat.ENGINE,
                ],
                supported_runtimes=[
                    RuntimeId.PYTORCH,
                    RuntimeId.ONNXRUNTIME,
                    RuntimeId.TENSORRT,
                ],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "imgsz": [512, 640],
                    "batch_size": [1],
                },
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolo11x.pt",
                neighbor_models=["yolo11l"],
                lighter_alternative="yolo11l",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="yolov8n",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
                params_m=3.2,
                default_input_shape=[3, 640, 640],
                default_hyperparams={"epochs": 50, "batch_size": 16, "lr": 0.01, "imgsz": 640},
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX, ExportFormat.ENGINE],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU, DeviceClass.RASPBERRY_PI],
                benchmark_input_signature={"imgsz": [320, 416, 512, 640], "batch_size": [1]},
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolov8n.pt",
                neighbor_models=["yolo11n", "yolov8s"],
                lighter_alternative=None,
                heavier_alternative="yolov8s",
            ),
            ModelSpec(
                name="yolov8s",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
                params_m=11.2,
                default_input_shape=[3, 640, 640],
                default_hyperparams={"epochs": 50, "batch_size": 16, "lr": 0.01, "imgsz": 640},
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX, ExportFormat.ENGINE],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [320, 416, 512, 640], "batch_size": [1]},
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolov8s.pt",
                neighbor_models=["yolov8n", "yolov8m"],
                lighter_alternative="yolov8n",
                heavier_alternative="yolov8m",
            ),
            ModelSpec(
                name="yolov8m",
                family_id="ultralytics",
                modality=Modality.VISION,
                supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
                params_m=25.9,
                default_input_shape=[3, 640, 640],
                default_hyperparams={"epochs": 50, "batch_size": 8, "lr": 0.01, "imgsz": 640},
                pip_packages=["ultralytics>=8.0"],
                supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX, ExportFormat.ENGINE],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [416, 512, 640], "batch_size": [1]},
                source_url="https://github.com/ultralytics/ultralytics",
                pretrained_weights="yolov8m.pt",
                neighbor_models=["yolov8s", "yolo11m"],
                lighter_alternative="yolov8s",
                heavier_alternative="yolo11m",
            ),
        ]
        specs.extend(self._build_extended_specs())
        dedup_specs = []
        seen_ids = set()
        for spec in specs:
            if spec.model_id in seen_ids:
                continue
            seen_ids.add(spec.model_id)
            spec.edge_compatible = True
            dedup_specs.append(spec)
        return dedup_specs

    def _build_extended_specs(self) -> List[ModelSpec]:
        """Build a broad Ultralytics model catalog (docs-aligned)."""
        size_params = {
            "n": 3.0,
            "s": 11.0,
            "m": 26.0,
            "b": 38.0,
            "l": 44.0,
            "x": 69.0,
            "t": 2.2,
            "c": 26.0,
            "e": 56.0,
        }
        base_tasks = [TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION]
        extended: List[ModelSpec] = []

        # YOLO family variants from the model docs (v5/v6/v8/v9/v10/11/26).
        series = {
            "yolo26": ["n", "s", "m", "l", "x"],
            "yolov10": ["n", "s", "m", "b", "l", "x"],
            "yolov9": ["t", "s", "m", "c", "e"],
            "yolov8": ["l", "x"],
            "yolov6": ["n", "s", "m", "l"],
            "yolov5": ["n", "s", "m", "l", "x"],
            "yolo11": ["p", "n-obb", "s-obb", "m-obb", "l-obb", "x-obb"],
        }
        for prefix, sizes in series.items():
            for size in sizes:
                name = f"{prefix}{size}"
                params = size_params.get(size[-1], 20.0)
                tasks = (
                    [TaskType.OBJECT_DETECTION]
                    if "obb" in size
                    else base_tasks
                )
                extended.append(
                    ModelSpec(
                        name=name,
                        family_id="ultralytics",
                        modality=Modality.VISION,
                        supported_tasks=tasks,
                        params_m=params,
                        default_input_shape=[3, 640, 640],
                        default_hyperparams={"epochs": 50, "batch_size": 8, "lr": 0.01, "imgsz": 640},
                        pip_packages=["ultralytics>=8.3.0"],
                        supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX, ExportFormat.ENGINE],
                        supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
                        edge_compatible=True,
                        recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                        benchmark_input_signature={"imgsz": [416, 512, 640], "batch_size": [1]},
                        source_url="https://docs.ultralytics.com/zh/models/",
                        pretrained_weights=f"{name}.pt",
                        neighbor_models=[],
                        lighter_alternative=None,
                        heavier_alternative=None,
                    )
                )

        # Other supported Ultralytics model families from docs.
        extras = [
            ("rtdetr-l", 32.0, [TaskType.OBJECT_DETECTION]),
            ("rtdetr-x", 70.0, [TaskType.OBJECT_DETECTION]),
            ("yolo_nas_s", 18.0, [TaskType.OBJECT_DETECTION]),
            ("yolo_nas_m", 36.0, [TaskType.OBJECT_DETECTION]),
            ("yolo_nas_l", 60.0, [TaskType.OBJECT_DETECTION]),
            ("yoloe-11s", 12.0, [TaskType.OBJECT_DETECTION]),
            ("yoloe-11m", 27.0, [TaskType.OBJECT_DETECTION]),
            ("yoloe-11l", 45.0, [TaskType.OBJECT_DETECTION]),
            ("yoloe-11s-seg", 13.0, [TaskType.SEGMENTATION]),
            ("yoloe-11m-seg", 30.0, [TaskType.SEGMENTATION]),
            ("yoloworld-s", 15.0, [TaskType.OBJECT_DETECTION]),
            ("yoloworld-m", 32.0, [TaskType.OBJECT_DETECTION]),
            ("yoloworld-l", 52.0, [TaskType.OBJECT_DETECTION]),
        ]
        for name, params, tasks in extras:
            extended.append(
                ModelSpec(
                    name=name,
                    family_id="ultralytics",
                    modality=Modality.VISION,
                    supported_tasks=tasks,
                    params_m=params,
                    default_input_shape=[3, 640, 640],
                    default_hyperparams={"epochs": 50, "batch_size": 8, "lr": 0.01, "imgsz": 640},
                    pip_packages=["ultralytics>=8.3.0"],
                    supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX, ExportFormat.ENGINE],
                    supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME, RuntimeId.TENSORRT],
                    edge_compatible=True,
                    recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                    benchmark_input_signature={"imgsz": [416, 512, 640], "batch_size": [1]},
                    source_url="https://docs.ultralytics.com/zh/models/",
                    pretrained_weights=f"{name}.pt",
                    neighbor_models=[],
                    lighter_alternative=None,
                    heavier_alternative=None,
                )
            )

        seen = set()
        dedup: List[ModelSpec] = []
        for spec in extended:
            if spec.name in seen:
                continue
            seen.add(spec.name)
            dedup.append(spec)
        return dedup

    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py for Ultralytics models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}

        # Determine model suffix based on task
        model_suffix = ""
        if context.task_type == TaskType.CLASSIFICATION:
            model_suffix = "-cls"
        elif context.task_type == TaskType.SEGMENTATION:
            model_suffix = "-seg"

        return f'''#!/usr/bin/env python3
"""Training script for {spec.name} ({context.task_type.value})."""
import json
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Model configuration
base_name = "{spec.name}"
model_suffix = "{model_suffix}"
model_name = base_name if (model_suffix and base_name.endswith(model_suffix)) else f"{{base_name}}{{model_suffix}}"
cache_dir = Path(os.getenv("EDGECRAFT_MODEL_CACHE_DIR", "{_DEFAULT_MODEL_CACHE_DIR}")).expanduser().resolve()
cache_dir.mkdir(parents=True, exist_ok=True)
cached_weights = cache_dir / f"{{model_name}}.pt"
workspace_weights_dir = Path("artifacts/pretrained")
workspace_weights_dir.mkdir(parents=True, exist_ok=True)
workspace_weights = workspace_weights_dir / cached_weights.name

# Download once into global cache, then copy into current trial workspace
if not cached_weights.exists():
    _ = YOLO(str(cached_weights))
if not workspace_weights.exists():
    shutil.copy2(cached_weights, workspace_weights)
model = YOLO(str(workspace_weights))

# Train the model
results = model.train(
    data="{context.dataset_path}",
    epochs={hp.get("epochs", 50)},
    imgsz={hp.get("imgsz", 640)},
    batch={hp.get("batch_size", 16)},
    lr0={hp.get("lr", 0.01)},
    device=0,
    verbose=True,
    project="outputs",
    name="train",
    exist_ok=True,
)

# Copy best weights to outputs/
best_src = Path(results.save_dir) / "weights" / "best.pt"
best_dst = Path("outputs/best.pt")
if best_src.exists():
    shutil.copy2(best_src, best_dst)
    model_path = str(best_dst)
else:
    model_path = ""

# Extract metrics
metrics_dict = results.results_dict
output = {{
    "status": "success",
    "model_path": model_path,
    "metrics": {{
        "mAP": float(metrics_dict.get("metrics/mAP50-95(B)", 0)),
        "mAP50": float(metrics_dict.get("metrics/mAP50(B)", 0)),
        "precision": float(metrics_dict.get("metrics/precision(B)", 0)),
        "recall": float(metrics_dict.get("metrics/recall(B)", 0)),
    }},
}}
print(json.dumps(output))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py for Ultralytics models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        imgsz = hp.get("imgsz", 640)

        return f'''#!/usr/bin/env python3
"""Edge fused eval script for {spec.name} (edge_eval_v1)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import yaml
from ultralytics import YOLO

try:
    import onnxruntime as ort
except ImportError:
    ort = None

def _pick_model_path() -> Path:
    for p in ("outputs/best.engine", "outputs/best.onnx", "outputs/best.pt"):
        q = Path(p)
        if q.exists():
            return q
    raise FileNotFoundError("No model artifact found under outputs/best.*")

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)

def _resolve_eval_split(cfg: dict) -> str:
    if isinstance(cfg, dict) and "test" in cfg:
        return "test"
    return "val"

def _run_dataset_eval(model: YOLO, cfg_path: Path, imgsz: int) -> dict:
    if not cfg_path.exists():
        return {{}}
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {{}}
        split = _resolve_eval_split(cfg)
        res = model.val(data=str(cfg_path), split=split, imgsz=imgsz, batch=1, verbose=False)
        rd = getattr(res, "results_dict", {{}}) or {{}}
        out = {{"edge_eval_split": split}}
        if isinstance(rd, dict):
            out["mAP"] = _safe_float(rd.get("metrics/mAP50-95(B)", 0.0))
            out["mAP50"] = _safe_float(rd.get("metrics/mAP50(B)", 0.0))
            out["Precision"] = _safe_float(rd.get("metrics/precision(B)", 0.0))
            out["Recall"] = _safe_float(rd.get("metrics/recall(B)", 0.0))
        return out
    except Exception as exc:
        return {{"edge_eval_error": str(exc)[:300]}}

def _latency_bench(model: YOLO, imgsz: int) -> tuple[float, float, float]:
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(dummy, verbose=False)
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg = float(np.mean(latencies)) if latencies else 0.0
    std = float(np.std(latencies)) if latencies else 0.0
    fps = float(1000.0 / avg) if avg > 0 else 0.0
    return avg, std, fps

def _ort_latency_bench(model_path: Path, imgsz: int) -> tuple[float, float, float, str]:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed in the edge image")
    available = set(ort.get_available_providers())
    preferred = []
    for provider in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if provider in available:
            preferred.append(provider)
    if not preferred:
        preferred = list(available) or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=preferred)
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    shape = list(input_meta.shape)
    h = int(shape[2]) if len(shape) > 2 and isinstance(shape[2], int) else imgsz
    w = int(shape[3]) if len(shape) > 3 and isinstance(shape[3], int) else imgsz
    dummy = np.random.randn(1, 3, h, w).astype(np.float32)
    for _ in range(5):
        _ = sess.run(None, {{input_name: dummy}})
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = sess.run(None, {{input_name: dummy}})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg = float(np.mean(latencies)) if latencies else 0.0
    std = float(np.std(latencies)) if latencies else 0.0
    fps = float(1000.0 / avg) if avg > 0 else 0.0
    provider = sess.get_providers()[0] if sess.get_providers() else preferred[0]
    return avg, std, fps, provider

def main():
    model_path = _pick_model_path()
    imgsz = int({imgsz})
    cfg_path = Path(os.getenv("EDGE_DATASET_CONFIG", "config/data.yaml"))
    eval_enabled = os.getenv("EDGE_EVAL_DATASET", "0") == "1"

    metrics = {{}}
    runtime_used = "onnxruntime" if model_path.suffix == ".onnx" else "ultralytics"
    runtime_provider = ""
    if model_path.suffix == ".onnx":
        avg_latency, std_latency, throughput, runtime_provider = _ort_latency_bench(model_path, imgsz)
        if eval_enabled:
            metrics["edge_eval_note"] = "dataset mAP eval skipped for direct ONNX Runtime benchmark"
    else:
        model = YOLO(str(model_path))
        if eval_enabled:
            metrics.update(_run_dataset_eval(model, cfg_path, imgsz))
        avg_latency, std_latency, throughput = _latency_bench(model, imgsz)
        runtime_provider = "ultralytics"
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    metrics.update({{
        "Latency": avg_latency,
        "Latency_std": std_latency,
        "Memory_mb": float(mem_mb),
        "Throughput": throughput,
    }})

    print(json.dumps({{
        "status": "success",
        "schema_version": "edge_eval_v1",
        "runtime_used": runtime_used,
        "runtime_provider": runtime_provider,
        "artifact_used": str(model_path),
        "metrics": metrics,
    }}))

if __name__ == "__main__":
    main()
'''

    def render_export_script(self, context: TemplateContext) -> str:
        """Generate export.py for Ultralytics models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        imgsz = hp.get("imgsz", 640)

        return f'''#!/usr/bin/env python3
"""Export script for {spec.name}."""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

model_path = Path("outputs/best.pt")
if not model_path.exists():
    print(json.dumps({{"status": "error", "error": "Model file not found"}}))
    sys.exit(1)

model = YOLO(str(model_path))
export_format = "{context.export_format.value}"
imgsz = {imgsz}

if export_format == "onnx":
    export_path = model.export(format="onnx", imgsz=imgsz, half=False, opset=12)
elif export_format == "engine":
    export_path = model.export(format="engine", imgsz=imgsz, half=True)
else:
    export_path = str(model_path)

print(json.dumps({{
    "status": "success",
    "export_path": str(export_path),
}}))
'''

    def get_supported_profile_runtimes(self, model_spec: ModelSpec) -> List[RuntimeId]:
        supported = []
        if RuntimeId.ONNXRUNTIME in model_spec.supported_runtimes:
            supported.append(RuntimeId.ONNXRUNTIME)
        if RuntimeId.TENSORRT in model_spec.supported_runtimes:
            supported.append(RuntimeId.TENSORRT)
        return supported

    def render_profile_benchmark_script(
        self,
        model_spec: ModelSpec,
        runtime_id: RuntimeId,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        if runtime_id == RuntimeId.ONNXRUNTIME:
            return wrap_with_tegrastats(self._render_profile_onnxruntime(model_spec.name, imgsz, warmup, iterations))
        if runtime_id == RuntimeId.TENSORRT:
            return wrap_with_tegrastats(self._render_profile_tensorrt(model_spec.name, imgsz, warmup, iterations))
        return ""

    def _render_profile_onnxruntime(
        self, model_name: str, imgsz: int, warmup: int, iterations: int
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import time
import numpy as np

try:
    from ultralytics import YOLO
    import onnxruntime as ort
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

try:
    model = YOLO("{model_name}.yaml")
    onnx_path = model.export(format="onnx", imgsz={imgsz}, simplify=True, verbose=False)
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    h = int(input_shape[2]) if len(input_shape) > 2 and isinstance(input_shape[2], int) else {imgsz}
    w = int(input_shape[3]) if len(input_shape) > 3 and isinstance(input_shape[3], int) else {imgsz}
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

dummy = np.random.randn(1, 3, h, w).astype(np.float32)
for _ in range({warmup}):
    _ = sess.run(None, {{input_name: dummy}})

latencies = []
for _ in range({iterations}):
    t0 = time.perf_counter()
    _ = sess.run(None, {{input_name: dummy}})
    latencies.append((time.perf_counter() - t0) * 1000)

arr = np.array(latencies, dtype=np.float64)
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
print(json.dumps({{
    "status":"success",
    "latency_avg_ms": float(arr.mean()),
    "latency_p50_ms": float(np.percentile(arr, 50)),
    "latency_p95_ms": float(np.percentile(arr, 95)),
    "latency_p99_ms": float(np.percentile(arr, 99)),
    "latency_min_ms": float(arr.min()),
    "latency_max_ms": float(arr.max()),
    "latency_std_ms": float(arr.std()),
    "throughput_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    "peak_memory_mb": float(mem_mb),
    "raw_latencies": arr.tolist(),
}}))
'''

    def _render_profile_tensorrt(
        self, model_name: str, imgsz: int, warmup: int, iterations: int
    ) -> str:
        return f'''#!/usr/bin/env python3
import contextlib
import io
import json
import time
import numpy as np

try:
    from ultralytics import YOLO
    import torch
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = YOLO("{model_name}.yaml")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        engine_path = model.export(format="engine", imgsz={imgsz}, half=True, workspace=4, simplify=True, verbose=False)
    model = YOLO(engine_path)
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export","error_message":str(e)}}))
    raise SystemExit(1)

dummy = np.random.randint(0, 255, ({imgsz}, {imgsz}, 3), dtype=np.uint8)
for _ in range({warmup}):
    _ = model.predict(dummy, verbose=False, device=device)

latencies = []
for _ in range({iterations}):
    t0 = time.perf_counter()
    _ = model.predict(dummy, verbose=False, device=device)
    latencies.append((time.perf_counter() - t0) * 1000)

arr = np.array(latencies, dtype=np.float64)
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
print(json.dumps({{
    "status":"success",
    "latency_avg_ms": float(arr.mean()),
    "latency_p50_ms": float(np.percentile(arr, 50)),
    "latency_p95_ms": float(np.percentile(arr, 95)),
    "latency_p99_ms": float(np.percentile(arr, 99)),
    "latency_min_ms": float(arr.min()),
    "latency_max_ms": float(arr.max()),
    "latency_std_ms": float(arr.std()),
    "throughput_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    "peak_memory_mb": float(mem_mb),
    "gpu_memory_mb": float(gpu_mem) if gpu_mem is not None else None,
    "raw_latencies": arr.tolist(),
}}))
'''

    def get_search_neighbors(
        self,
        model_name: str,
        direction: Literal["lighter", "heavier", "similar"],
    ) -> List[str]:
        """Get neighboring YOLO models for search exploration."""
        # YOLO size hierarchy
        yolo_hierarchy = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

        if model_name not in yolo_hierarchy:
            return super().get_search_neighbors(model_name, direction)

        idx = yolo_hierarchy.index(model_name)

        if direction == "lighter" and idx > 0:
            return [yolo_hierarchy[idx - 1]]
        if direction == "heavier" and idx < len(yolo_hierarchy) - 1:
            return [yolo_hierarchy[idx + 1]]
        if direction == "similar":
            neighbors = []
            if idx > 0:
                neighbors.append(yolo_hierarchy[idx - 1])
            if idx < len(yolo_hierarchy) - 1:
                neighbors.append(yolo_hierarchy[idx + 1])
            return neighbors

        return []


# Module-level instance for easy access
ultralytics_plugin = UltralyticsPlugin()
