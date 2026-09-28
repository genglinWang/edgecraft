"""Point-annotation crowd counting (density map regression).

Reads images + GT .mat (image_info.location). Trains a CNN to predict a fixed-stride
density map; reports MAE between predicted count (sum of map) and head count.

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


class CrowdCountingPlugin(BaseFamilyPlugin):
    """Density-map crowd counting for point-annotation layouts."""

    @property
    def family_id(self) -> str:
        return "crowd_counting"

    @property
    def display_name(self) -> str:
        return "Crowd Counting (Point Annotation)"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.VISION]

    def get_model_specs(self) -> List[ModelSpec]:
        return [
            ModelSpec(
                name="shanghai_mcnn",
                family_id="crowd_counting",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CROWD_COUNTING],
                params_m=4.0,
                default_input_shape=[3, 448, 448],
                default_hyperparams={
                    "epochs": 30,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "imgsz": 448,
                },
                pip_packages=["torch", "torchvision", "pyyaml", "scipy", "numpy"],
                supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [384, 448], "batch_size": [1]},
                source_url="https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map",
                pretrained_weights=None,
                neighbor_models=["shanghai_csr_tiny"],
                lighter_alternative=None,
                heavier_alternative="shanghai_csr_tiny",
            ),
            ModelSpec(
                name="shanghai_csr_tiny",
                family_id="crowd_counting",
                modality=Modality.VISION,
                supported_tasks=[TaskType.CROWD_COUNTING],
                params_m=12.0,
                default_input_shape=[3, 448, 448],
                default_hyperparams={
                    "epochs": 30,
                    "batch_size": 2,
                    "lr": 5e-5,
                    "imgsz": 448,
                },
                pip_packages=["torch", "torchvision", "pyyaml", "scipy", "numpy"],
                supported_export_formats=[ExportFormat.PT, ExportFormat.ONNX],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={"imgsz": [384, 448], "batch_size": [1]},
                source_url="https://github.com/leeyeehoo/CSRNet-pytorch",
                pretrained_weights="vgg16_bn_backbone",
                neighbor_models=["shanghai_mcnn"],
                lighter_alternative="shanghai_mcnn",
                heavier_alternative=None,
            ),
        ]

    def render_train_script(self, context: TemplateContext) -> str:
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        epochs = int(hp.get("epochs", 30))
        batch_size = int(hp.get("batch_size", 4))
        lr = float(hp.get("lr", 1e-4))
        imgsz = int(hp.get("imgsz", 448))
        model_name = spec.name
        use_csr = model_name == "shanghai_csr_tiny"

        return f'''#!/usr/bin/env python3
"""Point-annotation crowd counting — density map regression."""
import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CFG = Path("config/data.yaml")
with open(CFG) as f:
    cfg = yaml.safe_load(f)
task = cfg.get("task")
aliases = set(cfg.get("deprecated_task_aliases") or [])
valid_tasks = {{"point_annotation_crowd_counting", "shanghai_tech"}} | aliases
assert task in valid_tasks, "expected point-annotation crowd-counting task in data.yaml"
root = Path(cfg["root"])
part = cfg.get("part", "part_A")
imgsz = {imgsz}
epochs = {epochs}
batch_size = {batch_size}
lr = {lr}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_csr = {use_csr}


def load_points(mat_path: Path):
    m = sio.loadmat(str(mat_path))
    if "image_info" not in m:
        return np.zeros((0, 2), dtype=np.float32)
    ii = m["image_info"][0, 0]
    row = ii[0, 0]
    loc = row["location"]
    if loc.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(loc, dtype=np.float32)


def make_density(oh: int, ow: int, pts: np.ndarray, ih: int, iw: int, sigma: float = 4.0):
    dens = np.zeros((oh, ow), dtype=np.float32)
    yy, xx = np.indices((oh, ow))
    for pt in pts:
        px = float(pt[0]) * ow / max(iw, 1)
        py = float(pt[1]) * oh / max(ih, 1)
        g = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma**2))
        dens += g
    return dens


class PointAnnotationCrowdDataset(Dataset):
    def __init__(self, img_dir: Path, gt_dir: Path, train: bool = True, val_frac: float = 0.1):
        self.paths = sorted(
            [p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        n = len(self.paths)
        if train:
            self.paths = self.paths[: int(n * (1 - val_frac))]
        else:
            self.paths = self.paths[int(n * (1 - val_frac)) :]
        self.gt_dir = gt_dir
        self.imgsz = imgsz
        # output map: 3x maxpool -> /8
        self.out_hw = imgsz // 8

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ip = self.paths[idx]
        im = Image.open(ip).convert("RGB")
        iw, ih = im.size
        im = im.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        x = torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        stem = re.sub(r"[.](jpg|jpeg|png)$", "", ip.name, flags=re.I)
        gt_name = "GT_" + stem + ".mat"
        mat_p = self.gt_dir / gt_name
        pts = load_points(mat_p) if mat_p.exists() else np.zeros((0, 2), dtype=np.float32)
        target = make_density(self.out_hw, self.out_hw, pts, ih, iw)
        y = torch.from_numpy(target).unsqueeze(0)
        return x, y, torch.tensor(float(len(pts)))


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class CSRTiny(nn.Module):
    """VGG16-BN features (first 10 layers) + dilated backend (light)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import vgg16_bn, VGG16_BN_Weights

        w = VGG16_BN_Weights.DEFAULT if pretrained else None
        vgg = vgg16_bn(weights=w).features
        self.front = nn.Sequential(*list(vgg.children())[:23])
        self.back = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        z = self.front(x)
        return self.back(z)


part_dir = root / part
train_img = part_dir / "train_data" / "images"
train_gt = part_dir / "train_data" / "ground-truth"
test_img = part_dir / "test_data" / "images"
test_gt = part_dir / "test_data" / "ground-truth"

assert train_img.is_dir() and train_gt.is_dir(), f"missing {{train_img}} or {{train_gt}}"

ds_tr = PointAnnotationCrowdDataset(train_img, train_gt, train=True)
ds_va = PointAnnotationCrowdDataset(train_img, train_gt, train=False)
ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=None)
ld_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=0)

if use_csr:
    model = CSRTiny(pretrained=True).to(device)
else:
    model = MCNN().to(device)

opt = optim.Adam(model.parameters(), lr=lr)
crit = nn.MSELoss()

best_mae = float("inf")
for ep in range(epochs):
    model.train()
    for xb, yb, _ in ld_tr:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        if pred.shape[-2:] != yb.shape[-2:]:
            pred = F.interpolate(pred, size=yb.shape[-2:], mode="bilinear", align_corners=False)
        loss = crit(pred, yb)
        loss.backward()
        opt.step()
    # val MAE (count)
    model.eval()
    maes = []
    with torch.no_grad():
        for xb, yb, cnt in ld_va:
            xb = xb.to(device)
            pred = model(xb)
            if pred.shape[-2:] != yb.shape[-2:]:
                pred = F.interpolate(pred, size=yb.shape[-2:], mode="bilinear", align_corners=False)
            pc = float(pred.sum().cpu())
            gc = float(cnt.item())
            maes.append(abs(pc - gc))
    mae = float(np.mean(maes)) if maes else float("nan")
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), "outputs/best.pt")

# Optional test-set MAE
test_mae = float("nan")
if test_img.is_dir() and test_gt.is_dir():
    maes_t = []
    model.load_state_dict(torch.load("outputs/best.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        for p in sorted(test_img.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            im = Image.open(p).convert("RGB")
            iw, ih = im.size
            imr = im.resize((imgsz, imgsz), Image.BILINEAR)
            x = torch.from_numpy(np.array(imr, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(x)
            stem = re.sub(r"[.](jpg|jpeg|png)$", "", p.name, flags=re.I)
            mat_p = test_gt / ("GT_" + stem + ".mat")
            pts = load_points(mat_p) if mat_p.exists() else np.zeros((0, 2))
            gc = float(len(pts))
            pc = float(pred.sum().cpu())
            maes_t.append(abs(pc - gc))
    test_mae = float(np.mean(maes_t)) if maes_t else float("nan")

print(
    json.dumps(
        {{
            "status": "success",
            "model_path": "outputs/best.pt",
            "metrics": {{
                "best_val_MAE": float(best_mae),
                "test_MAE": test_mae,
            }},
        }}
    )
)
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        imgsz = int(hp.get("imgsz", 448))
        model_name = spec.name
        use_csr = model_name == "shanghai_csr_tiny"
        return f'''#!/usr/bin/env python3
"""Latency benchmark for crowd counting network."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgsz = {imgsz}
use_csr = {use_csr}


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class CSRTiny(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        w = VGG16_BN_Weights.DEFAULT if pretrained else None
        vgg = vgg16_bn(weights=w).features
        self.front = nn.Sequential(*list(vgg.children())[:23])
        self.back = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        return self.back(self.front(x))


if use_csr:
    model = CSRTiny(pretrained=False).to(device)
else:
    model = MCNN().to(device)
from pathlib import Path
wp = Path("outputs/best.pt")
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
print(json.dumps({{"status": "success", "metrics": {{"Latency": avg, "throughput_fps": 1000.0 / avg if avg > 0 else 0}}}}))
'''

    def get_supported_profile_runtimes(self, model_spec: ModelSpec) -> List[RuntimeId]:
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
        is_csr = name == "shanghai_csr_tiny"
        return wrap_with_tegrastats(f'''#!/usr/bin/env python3
"""Offline profile: crowd_counting family → ONNXRuntime (random weights)."""
import json
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    import psutil
    from torchvision.models import vgg16_bn
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

MODEL_NAME = "{name}"
IMGSZ = {ih}
WARMUP = {warmup}
ITERATIONS = {iterations}
IS_CSR = {is_csr}


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class CSRTiny(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16_bn(weights=None).features
        self.front = nn.Sequential(*list(vgg.children())[:23])
        self.back = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        return self.back(self.front(x))


try:
    if IS_CSR:
        model = CSRTiny().eval().cpu()
    else:
        model = MCNN().eval().cpu()
    dummy = torch.randn(1, 3, IMGSZ, IMGSZ)
    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch", 2: "h", 3: "w"}}}},
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


crowd_counting_plugin = CrowdCountingPlugin()
