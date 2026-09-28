"""Normal-only visual anomaly detection family (reconstruction-based).

Trains only on train/good; image-level anomaly score = mean squared reconstruction error.
Evaluates image-level AUROC vs. test set (good vs. defective subfolders).

This plugin generates train.py / infer.py; it does not run training inside EdgeCraft.
"""
from __future__ import annotations

from typing import List

from edgecraft.core.modality import Modality, TaskType
from edgecraft.models.specs import (
    BaseFamilyPlugin,
    DeviceClass,
    ExportFormat,
    ModelSpec,
    RuntimeId,
    TemplateContext,
)
from edgecraft.models._tegrastats_code import wrap_with_tegrastats


class AnomalyDetectionPlugin(BaseFamilyPlugin):
    """Reconstruction-based unsupervised anomaly detection for category layouts."""

    @property
    def family_id(self) -> str:
        return "anomaly_detection"

    @property
    def display_name(self) -> str:
        return "Normal-only Visual Anomaly Detection"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.VISION]

    def get_model_specs(self) -> List[ModelSpec]:
        return [
            ModelSpec(
                name="mvtec_conv_ae",
                family_id="anomaly_detection",
                modality=Modality.VISION,
                supported_tasks=[TaskType.ANOMALY_DETECTION],
                params_m=2.0,
                default_input_shape=[3, 128, 128],
                default_hyperparams={
                    "epochs": 40,
                    "batch_size": 16,
                    "lr": 0.001,
                    "imgsz": 128,
                },
                pip_packages=["torch", "torchvision", "pyyaml", "scikit-learn"],
                supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [128, 256], "batch_size": [1]},
                source_url="https://www.mvtec.com/company/research/datasets/mvtec-ad",
                pretrained_weights=None,
                neighbor_models=["mvtec_resnet18_ae"],
                lighter_alternative=None,
                heavier_alternative="mvtec_resnet18_ae",
            ),
            ModelSpec(
                name="mvtec_resnet18_ae",
                family_id="anomaly_detection",
                modality=Modality.VISION,
                supported_tasks=[TaskType.ANOMALY_DETECTION],
                params_m=25.0,
                default_input_shape=[3, 128, 128],
                default_hyperparams={
                    "epochs": 40,
                    "batch_size": 8,
                    "lr": 0.0005,
                    "imgsz": 128,
                },
                pip_packages=["torch", "torchvision", "pyyaml", "scikit-learn"],
                supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [128, 256], "batch_size": [1]},
                source_url="https://www.mvtec.com/company/research/datasets/mvtec-ad",
                pretrained_weights="encoder_imagenet",
                neighbor_models=["mvtec_conv_ae"],
                lighter_alternative="mvtec_conv_ae",
                heavier_alternative=None,
            ),
        ]

    def render_train_script(self, context: TemplateContext) -> str:
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        epochs = int(hp.get("epochs", 40))
        batch_size = int(hp.get("batch_size", 16))
        lr = float(hp.get("lr", 0.001))
        imgsz = int(hp.get("imgsz", 128))
        model_name = spec.name
        use_pretrained_enc = model_name == "mvtec_resnet18_ae"

        return f'''#!/usr/bin/env python3
"""Normal-only category anomaly detection."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

CFG = Path("config/data.yaml")
with open(CFG) as f:
    cfg = yaml.safe_load(f)
task = cfg.get("task")
aliases = set(cfg.get("deprecated_task_aliases") or [])
valid_tasks = {{"category_structured_visual_anomaly", "mvtec_anomaly"}} | aliases
assert task in valid_tasks, "config/data.yaml must describe a normal-only category anomaly layout"
root = Path(cfg["root"]) / cfg["category"]
imgsz = {imgsz}
epochs = {epochs}
batch_size = {batch_size}
lr = {lr}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pretrained_enc = {use_pretrained_enc}


class TrainDataset(Dataset):
    def __init__(self, good_dir: Path):
        self.paths = sorted(
            [p for p in good_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        self.tf = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


class ResNet18AE(nn.Module):
    """Encoder: ResNet18 trunk (no FC); decoder: mirrors spatial size back to RGB."""

    def __init__(self, pretrained: bool):
        super().__init__()
        w = ResNet18_Weights.DEFAULT if pretrained else None
        rn = resnet18(weights=w)
        self.enc = nn.Sequential(
            rn.conv1,
            rn.bn1,
            rn.relu,
            rn.maxpool,
            rn.layer1,
            rn.layer2,
            rn.layer3,
            rn.layer4,
        )
        # layer4 out: (B, 512, h/32, w/32) for imgsz multiple of 32
        # 4x4 -> 8 -> 16 -> 32 -> 64 -> 128 (5 stride-2 upsamples; input imgsz must be 128)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


def collect_test_samples(category_root: Path):
    test_dir = category_root / "test"
    xs, ys = [], []
    tf = transforms.Compose(
        [transforms.Resize((imgsz, imgsz)), transforms.ToTensor()]
    )
    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for p in sub.glob("*"):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = Image.open(p).convert("RGB")
            xs.append(tf(img))
            ys.append(label)
    return torch.stack(xs), np.array(ys, dtype=np.int64)


good_dir = root / "train" / "good"
assert good_dir.is_dir(), f"missing {{good_dir}}"

if "{model_name}" == "mvtec_conv_ae":
    model = ConvAE().to(device)
elif "{model_name}" == "mvtec_resnet18_ae":
    model = ResNet18AE(pretrained=use_pretrained_enc).to(device)
else:
    model = ConvAE().to(device)

loader = DataLoader(
    TrainDataset(good_dir), batch_size=batch_size, shuffle=True, num_workers=2
)
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

best_loss = float("inf")
for ep in range(epochs):
    model.train()
    ep_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        opt.step()
        ep_loss += loss.item() * batch.size(0)
        n += batch.size(0)
    ep_loss /= max(n, 1)
    if ep_loss < best_loss:
        best_loss = ep_loss
        torch.save(model.state_dict(), "outputs/best.pt")

# AUROC on test split
model.eval()
X, y = collect_test_samples(root)
scores = []
with torch.no_grad():
    for i in range(0, len(X), batch_size):
        chunk = X[i : i + batch_size].to(device)
        out = model(chunk)
        err = (chunk - out).pow(2).mean(dim=(1, 2, 3))
        scores.append(err.cpu().numpy())
scores = np.concatenate(scores)
try:
    auroc = float(roc_auc_score(y, scores))
except Exception:
    auroc = float("nan")

out = {{
    "status": "success",
    "model_path": "outputs/best.pt",
    "metrics": {{
        "best_reconstruction_loss": float(best_loss),
        "AUROC": auroc,
        "n_test": int(len(y)),
    }},
}}
print(json.dumps(out))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        imgsz = int(hp.get("imgsz", 128))
        model_name = spec.name
        use_pretrained_enc = model_name == "mvtec_resnet18_ae"
        return f'''#!/usr/bin/env python3
"""Latency micro-benchmark for anomaly detection autoencoder."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgsz = {imgsz}
use_pretrained_enc = {use_pretrained_enc}


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class ResNet18AE(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        w = ResNet18_Weights.DEFAULT if pretrained else None
        rn = resnet18(weights=w)
        self.enc = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3, rn.layer4,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


if "{model_name}" == "mvtec_conv_ae":
    model = ConvAE().to(device)
else:
    model = ResNet18AE(pretrained=use_pretrained_enc).to(device)
wp = __import__("pathlib").Path("outputs/best.pt")
if wp.exists():
    model.load_state_dict(torch.load(str(wp), map_location=device))
model.eval()

dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy)
lat = []
with torch.no_grad():
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
avg = float(np.mean(lat))
print(json.dumps({{
    "status": "success",
    "metrics": {{"Latency": avg, "throughput_fps": 1000.0 / avg if avg > 0 else 0}},
}}))
'''

    def get_supported_profile_runtimes(self, model_spec: ModelSpec) -> List[RuntimeId]:
        """ORT profiling only (export random-weight model → ONNX → benchmark)."""
        supported: List[RuntimeId] = []
        if RuntimeId.ONNXRUNTIME in model_spec.supported_runtimes:
            supported.append(RuntimeId.ONNXRUNTIME)
        return supported

    def render_profile_benchmark_script(
        self,
        model_spec: ModelSpec,
        runtime_id: RuntimeId,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        if runtime_id != RuntimeId.ONNXRUNTIME:
            return ""
        ih = int(model_spec.default_hyperparams.get("imgsz", imgsz))
        name = model_spec.name
        return wrap_with_tegrastats(f'''#!/usr/bin/env python3
"""Offline profile: anomaly_detection family → ONNXRuntime (random weights)."""
import json
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    import psutil
    from torchvision.models import resnet18
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

MODEL_NAME = "{name}"
IMGSZ = {ih}
WARMUP = {warmup}
ITERATIONS = {iterations}


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class ResNet18AE(nn.Module):
    def __init__(self):
        super().__init__()
        rn = resnet18(weights=None)
        self.enc = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3, rn.layer4,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


try:
    device = torch.device("cpu")
    if MODEL_NAME == "mvtec_conv_ae":
        model = ConvAE().eval().cpu()
    else:
        model = ResNet18AE().eval().cpu()
    dummy = torch.randn(1, 3, IMGSZ, IMGSZ)
    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch"}}}},
        opset_version=13,
    )
    sess = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

dummy_np = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
for _ in range(WARMUP):
    _ = sess.run(None, {{input_name: dummy_np}})

latencies = []
for _ in range(ITERATIONS):
    t0 = time.perf_counter()
    _ = sess.run(None, {{input_name: dummy_np}})
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
''')


anomaly_detection_plugin = AnomalyDetectionPlugin()
