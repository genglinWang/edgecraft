"""Ultralytics YOLO model definitions and code templates."""
from edgecraft.core.modality import Modality, TaskType
from edgecraft.models.registry import (
    ModelCard,
    ModelRegistry,
    register_train_template,
    register_infer_template,
)


# ---------------------------------------------------------------------------
# YOLO11 Detection Models
# ---------------------------------------------------------------------------

YOLO11_DETECTION_MODELS = [
    ModelCard(
        name="yolo11n",
        family="ultralytics",
        modality=Modality.VISION,
        supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
        params_m=2.6,
        default_input_shape=[3, 640, 640],
        default_hyperparams={"epochs": 50, "batch_size": 16, "lr": 0.01, "imgsz": 640},
        pip_packages=["ultralytics>=8.0"],
        supported_export_formats=["onnx", "engine", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="yolo11s",
        family="ultralytics",
        modality=Modality.VISION,
        supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
        params_m=9.4,
        default_input_shape=[3, 640, 640],
        default_hyperparams={"epochs": 50, "batch_size": 16, "lr": 0.01, "imgsz": 640},
        pip_packages=["ultralytics>=8.0"],
        supported_export_formats=["onnx", "engine", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="yolo11m",
        family="ultralytics",
        modality=Modality.VISION,
        supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
        params_m=20.1,
        default_input_shape=[3, 640, 640],
        default_hyperparams={"epochs": 50, "batch_size": 16, "lr": 0.01, "imgsz": 640},
        pip_packages=["ultralytics>=8.0"],
        supported_export_formats=["onnx", "engine", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="yolo11l",
        family="ultralytics",
        modality=Modality.VISION,
        supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
        params_m=25.3,
        default_input_shape=[3, 640, 640],
        default_hyperparams={"epochs": 100, "batch_size": 8, "lr": 0.01, "imgsz": 640},
        pip_packages=["ultralytics>=8.0"],
        supported_export_formats=["onnx", "engine", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="yolo11x",
        family="ultralytics",
        modality=Modality.VISION,
        supported_tasks=[TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
        params_m=56.9,
        default_input_shape=[3, 640, 640],
        default_hyperparams={"epochs": 100, "batch_size": 8, "lr": 0.005, "imgsz": 640},
        pip_packages=["ultralytics>=8.0"],
        supported_export_formats=["onnx", "engine", "pt"],
        edge_compatible=False,  # Too large for most edge devices
    ),
]

# ---------------------------------------------------------------------------
# Code Templates for Ultralytics
# ---------------------------------------------------------------------------

ULTRALYTICS_DETECTION_TRAIN_TEMPLATE = '''#!/usr/bin/env python3
"""Training script for YOLO detection model."""
import json
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Model configuration — use workspace weights/ dir (shared cache symlink)
weights_dir = Path(os.getenv("EDGECRAFT_MODEL_CACHE_DIR", "weights"))
weights_dir.mkdir(parents=True, exist_ok=True)
model_weights = weights_dir / "{model_name}.pt"
if not model_weights.exists():
    _ = YOLO(str(model_weights))
model = YOLO(str(model_weights))

# Train the model
results = model.train(
    data="config/data.yaml",
    epochs={epochs},
    imgsz={imgsz},
    batch={batch_size},
    lr0={lr},
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

ULTRALYTICS_DETECTION_INFER_TEMPLATE = '''#!/usr/bin/env python3
"""Inference benchmark script for YOLO detection model."""
import json
import time
import sys
from pathlib import Path

import numpy as np

# Check if we can use TensorRT
try:
    import tensorrt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

# Load model
model_path = Path("outputs/best.pt")
if not model_path.exists():
    print(json.dumps({{"status": "error", "error": "Model file not found"}}))
    sys.exit(1)

from ultralytics import YOLO
model = YOLO(str(model_path))

# Export to optimized format if needed
export_format = "{export_format}"
imgsz = {imgsz}
if export_format == "engine" and HAS_TRT:
    model.export(format="engine", imgsz=imgsz, half=True)
    engine_path = model_path.with_suffix(".engine")
    if engine_path.exists():
        model = YOLO(str(engine_path))
elif export_format == "onnx":
    model.export(format="onnx", imgsz=imgsz, half=False, opset=12)

# Warmup
dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
for _ in range(10):
    _ = model.predict(dummy_input, verbose=False)

# Benchmark
n_runs = 100
latencies = []
for _ in range(n_runs):
    start = time.perf_counter()
    _ = model.predict(dummy_input, verbose=False)
    latencies.append((time.perf_counter() - start) * 1000)

avg_latency = float(np.mean(latencies))
std_latency = float(np.std(latencies))

# Memory estimation (rough)
import psutil
process = psutil.Process()
mem_mb = process.memory_info().rss / (1024 * 1024)

output = {{
    "status": "success",
    "metrics": {{
        "Latency": avg_latency,
        "Latency_std": std_latency,
        "Memory_mb": mem_mb,
        "throughput_fps": 1000.0 / avg_latency if avg_latency > 0 else 0,
    }},
}}
print(json.dumps(output))
'''

ULTRALYTICS_CLASSIFICATION_TRAIN_TEMPLATE = '''#!/usr/bin/env python3
"""Training script for YOLO classification model."""
import json
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Model configuration — use workspace weights/ dir (shared cache symlink)
weights_dir = Path(os.getenv("EDGECRAFT_MODEL_CACHE_DIR", "weights"))
weights_dir.mkdir(parents=True, exist_ok=True)
model_weights = weights_dir / "{model_name}-cls.pt"
if not model_weights.exists():
    _ = YOLO(str(model_weights))
model = YOLO(str(model_weights))

# Train the model
results = model.train(
    data="config/data.yaml",
    epochs={epochs},
    imgsz={imgsz},
    batch={batch_size},
    lr0={lr},
    device=0,
    verbose=True,
    project="outputs",
    name="train",
    exist_ok=True,
)

# Copy best weights
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
        "accuracy_top1": float(metrics_dict.get("metrics/accuracy_top1", 0)),
        "accuracy_top5": float(metrics_dict.get("metrics/accuracy_top5", 0)),
    }},
}}
print(json.dumps(output))
'''

ULTRALYTICS_SEGMENTATION_TRAIN_TEMPLATE = '''#!/usr/bin/env python3
"""Training script for YOLO segmentation model."""
import json
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Model configuration — use workspace weights/ dir (shared cache symlink)
weights_dir = Path(os.getenv("EDGECRAFT_MODEL_CACHE_DIR", "weights"))
weights_dir.mkdir(parents=True, exist_ok=True)
model_weights = weights_dir / "{model_name}-seg.pt"
if not model_weights.exists():
    _ = YOLO(str(model_weights))
model = YOLO(str(model_weights))

# Train the model
results = model.train(
    data="config/data.yaml",
    epochs={epochs},
    imgsz={imgsz},
    batch={batch_size},
    lr0={lr},
    device=0,
    verbose=True,
    project="outputs",
    name="train",
    exist_ok=True,
)

# Copy best weights
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
        "mAP_mask": float(metrics_dict.get("metrics/mAP50-95(M)", 0)),
    }},
}}
print(json.dumps(output))
'''


def register_all() -> None:
    """Register all Ultralytics models and templates."""
    # Register model cards
    for card in YOLO11_DETECTION_MODELS:
        ModelRegistry.register(card)

    # Register code templates (keys must match TaskType.value)
    register_train_template("ultralytics", "object_detection", ULTRALYTICS_DETECTION_TRAIN_TEMPLATE)
    register_train_template("ultralytics", "classification", ULTRALYTICS_CLASSIFICATION_TRAIN_TEMPLATE)
    register_train_template("ultralytics", "segmentation", ULTRALYTICS_SEGMENTATION_TRAIN_TEMPLATE)

    register_infer_template("ultralytics", "object_detection", ULTRALYTICS_DETECTION_INFER_TEMPLATE)
    register_infer_template("ultralytics", "classification", ULTRALYTICS_DETECTION_INFER_TEMPLATE)
    register_infer_template("ultralytics", "segmentation", ULTRALYTICS_DETECTION_INFER_TEMPLATE)
