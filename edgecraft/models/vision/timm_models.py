"""Timm model definitions and code templates for image classification."""
from edgecraft.core.modality import Modality, TaskType
from edgecraft.models.registry import (
    ModelCard,
    ModelRegistry,
    register_train_template,
    register_infer_template,
)


# ---------------------------------------------------------------------------
# Edge-compatible Classification Models
# ---------------------------------------------------------------------------

TIMM_EDGE_MODELS = [
    ModelCard(
        name="mobilenetv3_small_100",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=2.5,
        default_input_shape=[3, 224, 224],
        default_hyperparams={"epochs": 50, "batch_size": 64, "lr": 0.001, "imgsz": 224},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="mobilenetv3_large_100",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=5.4,
        default_input_shape=[3, 224, 224],
        default_hyperparams={"epochs": 50, "batch_size": 64, "lr": 0.001, "imgsz": 224},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="efficientnet_b0",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=5.3,
        default_input_shape=[3, 224, 224],
        default_hyperparams={"epochs": 50, "batch_size": 32, "lr": 0.001, "imgsz": 224},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="efficientnet_b1",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=7.8,
        default_input_shape=[3, 240, 240],
        default_hyperparams={"epochs": 50, "batch_size": 32, "lr": 0.001, "imgsz": 240},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=True,
    ),
    ModelCard(
        name="resnet18",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=11.7,
        default_input_shape=[3, 224, 224],
        default_hyperparams={"epochs": 50, "batch_size": 64, "lr": 0.01, "imgsz": 224},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=True,
    ),
]


# ---------------------------------------------------------------------------
# Larger Classification Models (cloud training, may not be edge-deployable)
# ---------------------------------------------------------------------------

TIMM_LARGE_MODELS = [
    ModelCard(
        name="resnet50",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=25.6,
        default_input_shape=[3, 224, 224],
        default_hyperparams={"epochs": 100, "batch_size": 32, "lr": 0.01, "imgsz": 224},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=False,
    ),
    ModelCard(
        name="efficientnet_b2",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=9.1,
        default_input_shape=[3, 260, 260],
        default_hyperparams={"epochs": 100, "batch_size": 32, "lr": 0.001, "imgsz": 260},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=False,
    ),
    ModelCard(
        name="efficientnet_b3",
        family="timm",
        modality=Modality.VISION,
        supported_tasks=[TaskType.CLASSIFICATION],
        params_m=12.0,
        default_input_shape=[3, 300, 300],
        default_hyperparams={"epochs": 100, "batch_size": 24, "lr": 0.001, "imgsz": 300},
        pip_packages=["timm>=0.9.0", "torch", "torchvision"],
        supported_export_formats=["onnx", "pt"],
        edge_compatible=False,
    ),
]


# ---------------------------------------------------------------------------
# Code Templates for Timm Classification
# ---------------------------------------------------------------------------

TIMM_CLASSIFICATION_TRAIN_TEMPLATE = '''#!/usr/bin/env python3
"""Training script for timm classification model."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import yaml

# Configuration
model_name = "{model_name}"
epochs = {epochs}
batch_size = {batch_size}
lr = {lr}
imgsz = {imgsz}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(imgsz),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(int(imgsz * 1.14)),
    transforms.CenterCrop(imgsz),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _load_imagefolder_contract():
    cfg_path = Path("config/data.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {{}}
    cfg = cfg or {{}}
    root = Path(cfg.get("root") or ".").expanduser()
    train_dir = Path(cfg.get("train") or (root / "train")).expanduser()
    val_dir = Path(cfg.get("val") or (root / "val")).expanduser()
    if not val_dir.exists():
        val_dir = train_dir
    return train_dir, val_dir, cfg


train_dir, val_dir, data_cfg = _load_imagefolder_contract()
train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
num_classes = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Create model
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training loop
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save({{
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "class_names": train_dataset.classes,
            "model_name": model_name,
        }}, "outputs/best.pt")

# Final output
output = {{
    "status": "success",
    "model_path": "outputs/best.pt",
    "metrics": {{
        "accuracy": best_acc,
        "accuracy_top1": best_acc,
    }},
}}
print(json.dumps(output))
'''

TIMM_CLASSIFICATION_INFER_TEMPLATE = '''#!/usr/bin/env python3
"""Inference benchmark script for timm classification model."""
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import timm
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = Path("outputs/best.pt")
if not model_path.exists():
    print(json.dumps({{"status": "error", "error": "Model file not found"}}))
    sys.exit(1)

# Configuration
model_name = "{model_name}"
imgsz = {imgsz}
cfg_path = Path("config/data.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {{}}
cfg = cfg or {{}}
num_classes = int(cfg.get("num_classes") or cfg.get("nc") or 10)

# Create model and load weights
checkpoint = torch.load(str(model_path), map_location=device)
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    num_classes = int(checkpoint.get("num_classes") or num_classes)
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# Export to ONNX if requested
export_format = "{export_format}"
if export_format == "onnx":
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    torch.onnx.export(
        model, dummy, "outputs/best.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch"}}}},
    )

# Warmup
dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

# Benchmark
n_runs = 100
latencies = []
with torch.no_grad():
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

avg_latency = float(np.mean(latencies))
std_latency = float(np.std(latencies))

# Memory estimation
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


def register_all() -> None:
    """Register all timm models and templates."""
    # Register model cards
    for card in TIMM_EDGE_MODELS:
        ModelRegistry.register(card)
    for card in TIMM_LARGE_MODELS:
        ModelRegistry.register(card)

    # Register code templates
    register_train_template("timm", "classification", TIMM_CLASSIFICATION_TRAIN_TEMPLATE)
    register_infer_template("timm", "classification", TIMM_CLASSIFICATION_INFER_TEMPLATE)
