"""Transformers family plugin for the EdgeCraft model zoo.

This plugin provides model specs and script generation for Hugging Face
Transformers models, focusing on edge-deployable text models.

Key principle: This plugin generates SCRIPTS, it does NOT execute training.
"""
from __future__ import annotations

from typing import List, Literal

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


class TransformersPlugin(BaseFamilyPlugin):
    """Family plugin for Hugging Face Transformers models."""

    @staticmethod
    def _required_pretrained_weights(spec: ModelSpec) -> str:
        model_name = (spec.pretrained_weights or "").strip()
        if not model_name:
            raise ValueError(
                f"ModelSpec.pretrained_weights is required for transformers model '{spec.name}'"
            )
        return model_name

    @property
    def family_id(self) -> str:
        return "transformers"

    @property
    def display_name(self) -> str:
        return "Hugging Face Transformers"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.TEXT]

    def get_model_specs(self) -> List[ModelSpec]:
        """Return all Transformers model specs (edge-focused)."""
        specs = [
            # DistilBERT (lightweight BERT)
            ModelSpec(
                name="distilbert-base-uncased",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[
                    TaskType.TEXT_CLASSIFICATION,
                ],
                params_m=66.0,
                default_input_shape=[1, 128],  # [batch, seq_len]
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 32,
                    "lr": 2e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "max_length": [64, 128, 256],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/distilbert-base-uncased",
                pretrained_weights="distilbert-base-uncased",
                neighbor_models=["mobilebert-uncased", "albert-base-v2"],
                lighter_alternative="mobilebert-uncased",
                heavier_alternative="bert-base-uncased",
            ),
            # MobileBERT (mobile-optimized)
            ModelSpec(
                name="mobilebert-uncased",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[
                    TaskType.TEXT_CLASSIFICATION,
                ],
                params_m=25.3,
                default_input_shape=[1, 128],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 32,
                    "lr": 3e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.X86_CPU,
                ],
                benchmark_input_signature={
                    "max_length": [64, 128, 256],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/google/mobilebert-uncased",
                pretrained_weights="google/mobilebert-uncased",
                neighbor_models=["distilbert-base-uncased", "albert-base-v2"],
                lighter_alternative=None,
                heavier_alternative="distilbert-base-uncased",
            ),
            # ALBERT (parameter-efficient)
            ModelSpec(
                name="albert-base-v2",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[
                    TaskType.TEXT_CLASSIFICATION,
                ],
                params_m=11.8,
                default_input_shape=[1, 128],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 32,
                    "lr": 2e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.X86_CPU,
                ],
                benchmark_input_signature={
                    "max_length": [64, 128, 256],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/albert-base-v2",
                pretrained_weights="albert-base-v2",
                neighbor_models=["mobilebert-uncased", "distilbert-base-uncased"],
                lighter_alternative="mobilebert-uncased",
                heavier_alternative="distilbert-base-uncased",
            ),
            # TinyBERT
            ModelSpec(
                name="tinybert-general-4l-312d",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[
                    TaskType.TEXT_CLASSIFICATION,
                ],
                params_m=14.5,
                default_input_shape=[1, 128],
                default_hyperparams={
                    "epochs": 5,
                    "batch_size": 32,
                    "lr": 3e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.ANDROID,
                    DeviceClass.RASPBERRY_PI,
                ],
                benchmark_input_signature={
                    "max_length": [64, 128],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D",
                pretrained_weights="huawei-noah/TinyBERT_General_4L_312D",
                neighbor_models=["albert-base-v2", "mobilebert-uncased"],
                lighter_alternative=None,
                heavier_alternative="albert-base-v2",
            ),
            # Full BERT (reference, not edge-compatible)
            ModelSpec(
                name="bert-base-uncased",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[
                    TaskType.TEXT_CLASSIFICATION,
                ],
                params_m=110.0,
                default_input_shape=[1, 128],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 16,
                    "lr": 2e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "max_length": [128, 256, 512],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/bert-base-uncased",
                pretrained_weights="bert-base-uncased",
                neighbor_models=["distilbert-base-uncased"],
                lighter_alternative="distilbert-base-uncased",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="bert-tiny-uncased",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[TaskType.TEXT_CLASSIFICATION],
                params_m=4.4,
                default_input_shape=[1, 128],
                default_hyperparams={
                    "epochs": 5,
                    "batch_size": 64,
                    "lr": 3e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.RASPBERRY_PI,
                    DeviceClass.X86_CPU,
                ],
                benchmark_input_signature={
                    "max_length": [64, 128],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/prajjwal1/bert-tiny",
                pretrained_weights="prajjwal1/bert-tiny",
                neighbor_models=["tinybert-general-4l-312d", "mobilebert-uncased"],
                lighter_alternative=None,
                heavier_alternative="mobilebert-uncased",
            ),
            ModelSpec(
                name="distilroberta-base",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[TaskType.TEXT_CLASSIFICATION],
                params_m=82.0,
                default_input_shape=[1, 128],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 32,
                    "lr": 2e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "max_length": [64, 128, 256],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/distilroberta-base",
                pretrained_weights="distilroberta-base",
                neighbor_models=["distilbert-base-uncased", "bert-base-uncased"],
                lighter_alternative="distilbert-base-uncased",
                heavier_alternative="bert-base-uncased",
            ),
            ModelSpec(
                name="deberta-v3-small",
                family_id="transformers",
                modality=Modality.TEXT,
                supported_tasks=[TaskType.TEXT_CLASSIFICATION],
                params_m=44.0,
                default_input_shape=[1, 128],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 16,
                    "lr": 2e-5,
                    "max_length": 128,
                },
                pip_packages=["transformers>=4.30.0", "torch", "datasets", "sentencepiece"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "max_length": [64, 128, 256],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/microsoft/deberta-v3-small",
                pretrained_weights="microsoft/deberta-v3-small",
                neighbor_models=["mobilebert-uncased", "distilbert-base-uncased"],
                lighter_alternative="mobilebert-uncased",
                heavier_alternative="distilbert-base-uncased",
            ),
        ]
        specs.extend(self._build_extended_specs())
        for spec in specs:
            spec.edge_compatible = True
            if RuntimeId.TENSORRT not in spec.supported_runtimes:
                spec.supported_runtimes.append(RuntimeId.TENSORRT)
        return specs

    def _build_extended_specs(self) -> List[ModelSpec]:
        """Build additional non-generative NLP classification models."""
        entries = [
            ("bert-mini-uncased", "prajjwal1/bert-mini", 11.3),
            ("bert-small-uncased", "prajjwal1/bert-small", 29.1),
            ("roberta-base", "roberta-base", 125.0),
            ("xlm-roberta-base", "xlm-roberta-base", 125.0),
            ("electra-small-discriminator", "google/electra-small-discriminator", 14.0),
            ("electra-base-discriminator", "google/electra-base-discriminator", 110.0),
            ("deberta-v3-base", "microsoft/deberta-v3-base", 86.0),
            ("modernbert-base", "answerdotai/ModernBERT-base", 149.0),
            ("bert-base-chinese", "bert-base-chinese", 102.0),
            ("distilbert-base-multilingual-cased", "distilbert-base-multilingual-cased", 135.0),
        ]
        specs: List[ModelSpec] = []
        for name, weights, params in entries:
            specs.append(
                ModelSpec(
                    name=name,
                    family_id="transformers",
                    modality=Modality.TEXT,
                    supported_tasks=[TaskType.TEXT_CLASSIFICATION],
                    params_m=params,
                    default_input_shape=[1, 128],
                    default_hyperparams={"epochs": 3, "batch_size": 16, "lr": 2e-5, "max_length": 128},
                    pip_packages=["transformers>=4.30.0", "torch", "datasets"],
                    supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                    supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                    edge_compatible=True,
                    recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU, DeviceClass.X86_CPU],
                    benchmark_input_signature={"max_length": [64, 128, 256], "batch_size": [1]},
                    source_url="https://huggingface.co/models",
                    pretrained_weights=weights,
                    neighbor_models=[],
                    lighter_alternative=None,
                    heavier_alternative=None,
                )
            )
        return specs

    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py for Transformers models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        model_name = self._required_pretrained_weights(spec)

        return f'''#!/usr/bin/env python3
"""Training script for {spec.name} text classification."""
import json
import sys
import inspect
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Configuration
model_name = "{model_name}"
max_length = {hp.get("max_length", 128)}
epochs = {hp.get("epochs", 3)}
batch_size = {hp.get("batch_size", 32)}
lr = {hp.get("lr", 2e-5)}

def fail(error_code, error_message, detail=None):
    payload = {{
        "status": "error",
        "error_code": error_code,
        "error": error_message,
    }}
    if detail:
        payload["detail"] = str(detail)
    print(json.dumps(payload))
    sys.exit(1)

def _version_tuple(v):
    parts = []
    for token in str(v).split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def check_accelerate():
    try:
        import accelerate  # type: ignore
    except Exception as exc:
        fail(
            "missing_dependency",
            "accelerate>=0.26.0 is required for transformers Trainer.",
            exc,
        )
    if _version_tuple(getattr(accelerate, "__version__", "0.0.0")) < (0, 26, 0):
        fail(
            "incompatible_dependency",
            f"accelerate>=0.26.0 required, found {{getattr(accelerate, '__version__', 'unknown')}}",
        )

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load and preprocess dataset
# Expects dataset in HuggingFace format or CSV
dataset_path = "{context.dataset_path}"
if Path(dataset_path).suffix == ".csv":
    dataset = load_dataset("csv", data_files=dataset_path)
else:
    dataset = load_dataset(dataset_path)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
check_accelerate()
ta_kwargs = {{
    "output_dir": "outputs/checkpoints",
    "num_train_epochs": epochs,
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "learning_rate": lr,
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
}}
ta_params = inspect.signature(TrainingArguments.__init__).parameters
if "evaluation_strategy" in ta_params:
    ta_kwargs["evaluation_strategy"] = "epoch"
elif "eval_strategy" in ta_params:
    ta_kwargs["eval_strategy"] = "epoch"
training_args = TrainingArguments(**ta_kwargs)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation") or tokenized_dataset.get("test"),
)

# Train
trainer.train()

# Save best model
trainer.save_model("outputs/best")

# Evaluate
eval_results = trainer.evaluate()

output = {{
    "status": "success",
    "model_path": "outputs/best",
    "metrics": {{
        "accuracy": eval_results.get("eval_accuracy", 0),
        "loss": eval_results.get("eval_loss", 0),
    }},
}}
print(json.dumps(output))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py for Transformers models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        max_length = hp.get("max_length", 128)

        return f'''#!/usr/bin/env python3
"""Edge fused eval script for {spec.name} (edge_eval_v1)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = {max_length}

def _safe_text(sample: dict) -> str:
    for k in ("text", "sentence", "content"):
        if k in sample and sample[k]:
            return str(sample[k])
    title = str(sample.get("title", "") or "")
    desc = str(sample.get("description", "") or "")
    merged = (title + " " + desc).strip()
    return merged if merged else ""

def _safe_label(sample: dict):
    for k in ("label", "labels", "target", "y"):
        if k in sample:
            return sample[k]
    return None

def _load_eval_dataset(cfg_path: Path):
    if not cfg_path.exists():
        return None, "none"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {{}}
    ds_path = str((cfg.get("dataset_path") or cfg.get("path") or os.getenv("EDGE_DATASET_DIR", "")).strip())
    if not ds_path:
        return None, "none"
    p = Path(ds_path)
    try:
        if p.is_dir():
            for name in ("test.jsonl", "test.json", "test.csv", "validation.jsonl", "validation.json", "validation.csv", "val.jsonl", "val.json", "val.csv"):
                q = p / name
                if q.exists():
                    if q.suffix == ".csv":
                        ds = load_dataset("csv", data_files={{"eval": str(q)}})["eval"]
                    else:
                        ds = load_dataset("json", data_files={{"eval": str(q)}})["eval"]
                    split = "test" if "test" in name else "validation"
                    return ds, split
            ds_obj = load_dataset(ds_path)
            split = "test" if "test" in ds_obj else ("validation" if "validation" in ds_obj else ("val" if "val" in ds_obj else "train"))
            return ds_obj[split], split
        if p.is_file():
            if p.suffix == ".csv":
                ds = load_dataset("csv", data_files={{"eval": str(p)}})["eval"]
            else:
                ds = load_dataset("json", data_files={{"eval": str(p)}})["eval"]
            return ds, "eval_file"
        ds_obj = load_dataset(ds_path)
        split = "test" if "test" in ds_obj else ("validation" if "validation" in ds_obj else ("val" if "val" in ds_obj else "train"))
        return ds_obj[split], split
    except Exception:
        return None, "none"

def _macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1s = []
    for lbl in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lbl and b == lbl)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lbl and b == lbl)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lbl and b != lbl)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
    return float(sum(f1s) / len(f1s))

model_path = Path("outputs/best")
if not model_path.exists():
    print(json.dumps({{"status": "error", "error": "Model not found"}}))
    raise SystemExit(1)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

metrics = {{}}
if os.getenv("EDGE_EVAL_DATASET", "0") == "1":
    eval_ds, split = _load_eval_dataset(Path(os.getenv("EDGE_DATASET_CONFIG", "config/data.yaml")))
    if eval_ds is not None:
        max_samples = int(os.getenv("EDGE_EVAL_MAX_SAMPLES", "512"))
        ys, ps = [], []
        with torch.no_grad():
            for i, sample in enumerate(eval_ds):
                if i >= max_samples:
                    break
                text = _safe_text(sample)
                label = _safe_label(sample)
                if not text or label is None:
                    continue
                toks = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
                toks = {{k: v.to(device) for k, v in toks.items()}}
                pred = int(torch.argmax(model(**toks).logits, dim=-1).item())
                ys.append(int(label))
                ps.append(pred)
        if ys:
            acc = float(sum(int(a == b) for a, b in zip(ys, ps)) / len(ys))
            metrics["Accuracy"] = acc
            metrics["F1Score"] = _macro_f1(ys, ps)
            metrics["edge_eval_samples"] = float(len(ys))
            metrics["edge_eval_split"] = split

dummy_tokens = tokenizer(
    "This is a sample text for benchmarking inference latency.",
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)
dummy_tokens = {{k: v.to(device) for k, v in dummy_tokens.items()}}

with torch.no_grad():
    for _ in range(5):
        _ = model(**dummy_tokens)

latencies = []
with torch.no_grad():
    for _ in range(40):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(**dummy_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

avg_latency = float(np.mean(latencies)) if latencies else 0.0
std_latency = float(np.std(latencies)) if latencies else 0.0
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
metrics.update({{
    "Latency": avg_latency,
    "Latency_std": std_latency,
    "Memory_mb": float(mem_mb),
    "Throughput": float(1000.0 / avg_latency) if avg_latency > 0 else 0.0,
}})

print(json.dumps({{
    "status": "success",
    "schema_version": "edge_eval_v1",
    "metrics": metrics,
}}))
'''

    def get_supported_profile_runtimes(self, model_spec: ModelSpec) -> List[RuntimeId]:
        supported: List[RuntimeId] = []
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
        _ = imgsz
        max_length = int(model_spec.default_hyperparams.get("max_length", 128))
        model_name = self._required_pretrained_weights(model_spec)
        if runtime_id == RuntimeId.ONNXRUNTIME:
            return wrap_with_tegrastats(self._render_profile_onnxruntime(model_name, max_length, warmup, iterations))
        if runtime_id == RuntimeId.TENSORRT:
            return wrap_with_tegrastats(self._render_profile_tensorrt(model_name, max_length, iterations))
        return ""

    def _render_profile_onnxruntime(
        self,
        model_name: str,
        max_length: int,
        warmup: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import time
import numpy as np

try:
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

model_name = "{model_name}"
max_length = {max_length}
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().cpu()
    input_ids = torch.ones((1, max_length), dtype=torch.long)
    attention_mask = torch.ones((1, max_length), dtype=torch.long)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={{
            "input_ids": {{0: "batch", 1: "seq"}},
            "attention_mask": {{0: "batch", 1: "seq"}},
            "logits": {{0: "batch"}},
        }},
        opset_version=13,
    )
    sess = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

feeds = {{
    "input_ids": np.ones((1, max_length), dtype=np.int64),
    "attention_mask": np.ones((1, max_length), dtype=np.int64),
}}
for _ in range({warmup}):
    _ = sess.run(None, feeds)

latencies = []
for _ in range({iterations}):
    t0 = time.perf_counter()
    _ = sess.run(None, feeds)
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
        self,
        model_name: str,
        max_length: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import re
import shutil
import subprocess

try:
    import torch
    from transformers import AutoModelForSequenceClassification
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

trtexec = shutil.which("trtexec")
if not trtexec:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":"trtexec not found in PATH"}}))
    raise SystemExit(1)

model_name = "{model_name}"
max_length = {max_length}
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().cpu()
    input_ids = torch.ones((1, max_length), dtype=torch.long)
    attention_mask = torch.ones((1, max_length), dtype=torch.long)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={{
            "input_ids": {{0: "batch", 1: "seq"}},
            "attention_mask": {{0: "batch", 1: "seq"}},
            "logits": {{0: "batch"}},
        }},
        opset_version=13,
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export","error_message":str(e)}}))
    raise SystemExit(1)

shape = f"1x{{max_length}}"
cmd = [
    trtexec,
    "--onnx=model.onnx",
    "--saveEngine=model.engine",
    "--explicitBatch",
    "--fp16",
    "--useSpinWait",
    "--warmUp=0",
    "--duration=0",
    "--iterations={iterations}",
    "--avgRuns=1",
    f"--minShapes=input_ids:{{shape}},attention_mask:{{shape}}",
    f"--optShapes=input_ids:{{shape}},attention_mask:{{shape}}",
    f"--maxShapes=input_ids:{{shape}},attention_mask:{{shape}}",
]
result = subprocess.run(cmd, capture_output=True, text=True)
combined = (result.stdout or "") + "\\n" + (result.stderr or "")
if result.returncode != 0:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":combined[-4000:]}}))
    raise SystemExit(1)

mean_match = re.search(r"mean\\s*=\\s*([0-9.]+)\\s*ms", combined)
p95_match = re.search(r"percentile\\(95%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
p99_match = re.search(r"percentile\\(99%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
if not mean_match:
    print(json.dumps({{"status":"error","error_type":"parse","error_message":"Could not parse trtexec latency output"}}))
    raise SystemExit(1)

lat_mean = float(mean_match.group(1))
lat_p95 = float(p95_match.group(1)) if p95_match else lat_mean
lat_p99 = float(p99_match.group(1)) if p99_match else lat_p95
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

print(json.dumps({{
    "status":"success",
    "latency_avg_ms": lat_mean,
    "latency_p50_ms": lat_mean,
    "latency_p95_ms": lat_p95,
    "latency_p99_ms": lat_p99,
    "latency_min_ms": lat_mean,
    "latency_max_ms": lat_mean,
    "latency_std_ms": 0.0,
    "throughput_fps": float(1000.0 / lat_mean) if lat_mean > 0 else 0.0,
    "peak_memory_mb": float(mem_mb),
    "raw_latencies": [],
}}))
'''


# Module-level instance
transformers_plugin = TransformersPlugin()
