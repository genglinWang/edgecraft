"""Audio family plugins for the EdgeCraft model zoo.

This module provides plugins for audio processing models:
- WhisperPlugin: OpenAI Whisper ASR models
- SpeechBrainPlugin: SpeechBrain speech processing models

Key principle: These plugins generate SCRIPTS, they do NOT execute training.
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


# ---------------------------------------------------------------------------
# WhisperPlugin: OpenAI Whisper ASR models
# ---------------------------------------------------------------------------

class WhisperPlugin(BaseFamilyPlugin):
    """Family plugin for OpenAI Whisper models."""

    @property
    def family_id(self) -> str:
        return "whisper"

    @property
    def display_name(self) -> str:
        return "OpenAI Whisper"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.AUDIO]

    def get_model_specs(self) -> List[ModelSpec]:
        """Return Whisper model specs."""
        specs = [
            ModelSpec(
                name="whisper-tiny",
                family_id="whisper",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.SPEECH_RECOGNITION],
                params_m=39.0,
                default_input_shape=[1, 80, 3000],  # [batch, mel_bins, time_steps]
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 8,
                    "lr": 1e-5,
                    "sample_rate": 16000,
                },
                pip_packages=["openai-whisper", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.RASPBERRY_PI,
                    DeviceClass.X86_CPU,
                ],
                benchmark_input_signature={
                    "audio_duration_s": [5, 10, 30],
                    "batch_size": [1],
                },
                source_url="https://github.com/openai/whisper",
                pretrained_weights="tiny",
                neighbor_models=["whisper-base"],
                lighter_alternative=None,
                heavier_alternative="whisper-base",
            ),
            ModelSpec(
                name="whisper-base",
                family_id="whisper",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.SPEECH_RECOGNITION],
                params_m=74.0,
                default_input_shape=[1, 80, 3000],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 8,
                    "lr": 1e-5,
                    "sample_rate": 16000,
                },
                pip_packages=["openai-whisper", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "audio_duration_s": [5, 10, 30],
                    "batch_size": [1],
                },
                source_url="https://github.com/openai/whisper",
                pretrained_weights="base",
                neighbor_models=["whisper-tiny", "whisper-small"],
                lighter_alternative="whisper-tiny",
                heavier_alternative="whisper-small",
            ),
            ModelSpec(
                name="whisper-small",
                family_id="whisper",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.SPEECH_RECOGNITION],
                params_m=244.0,
                default_input_shape=[1, 80, 3000],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 4,
                    "lr": 1e-5,
                    "sample_rate": 16000,
                },
                pip_packages=["openai-whisper", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "audio_duration_s": [5, 10, 30],
                    "batch_size": [1],
                },
                source_url="https://github.com/openai/whisper",
                pretrained_weights="small",
                neighbor_models=["whisper-base"],
                lighter_alternative="whisper-base",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="whisper-tiny-en",
                family_id="whisper",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.SPEECH_RECOGNITION],
                params_m=39.0,
                default_input_shape=[1, 80, 3000],
                default_hyperparams={
                    "epochs": 3,
                    "batch_size": 8,
                    "lr": 1e-5,
                    "sample_rate": 16000,
                },
                pip_packages=["openai-whisper", "transformers>=4.30.0", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.RASPBERRY_PI,
                    DeviceClass.X86_CPU,
                ],
                benchmark_input_signature={
                    "audio_duration_s": [5, 10, 30],
                    "batch_size": [1],
                },
                source_url="https://github.com/openai/whisper",
                pretrained_weights="tiny.en",
                neighbor_models=["whisper-tiny", "whisper-base"],
                lighter_alternative=None,
                heavier_alternative="whisper-base",
            ),
            ModelSpec(
                name="whisper-medium",
                family_id="whisper",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.SPEECH_RECOGNITION],
                params_m=769.0,
                default_input_shape=[1, 80, 3000],
                default_hyperparams={
                    "epochs": 2,
                    "batch_size": 1,
                    "lr": 1e-5,
                    "sample_rate": 16000,
                },
                pip_packages=["openai-whisper", "transformers>=4.30.0", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "audio_duration_s": [5, 10],
                    "batch_size": [1],
                },
                source_url="https://github.com/openai/whisper",
                pretrained_weights="medium",
                neighbor_models=["whisper-small"],
                lighter_alternative="whisper-small",
                heavier_alternative=None,
            ),
        ]
        specs.extend(self._build_extended_specs())
        for spec in specs:
            spec.edge_compatible = True
            if (
                TaskType.SPEECH_RECOGNITION not in spec.supported_tasks
                and RuntimeId.TENSORRT not in spec.supported_runtimes
            ):
                spec.supported_runtimes.append(RuntimeId.TENSORRT)
        return specs

    def _build_extended_specs(self) -> List[ModelSpec]:
        entries = [
            ("whisper-large-v2", "large-v2", 1550.0),
            ("whisper-large-v3", "large-v3", 1550.0),
            ("whisper-large-v3-turbo", "large-v3-turbo", 809.0),
        ]
        specs: List[ModelSpec] = []
        for name, weights, params in entries:
            specs.append(
                ModelSpec(
                    name=name,
                    family_id="whisper",
                    modality=Modality.AUDIO,
                    supported_tasks=[TaskType.SPEECH_RECOGNITION],
                    params_m=params,
                    default_input_shape=[1, 80, 3000],
                    default_hyperparams={"epochs": 2, "batch_size": 1, "lr": 1e-5, "sample_rate": 16000},
                    pip_packages=["openai-whisper", "transformers>=4.30.0", "torch", "torchaudio"],
                    supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                    supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                    edge_compatible=True,
                    recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                    benchmark_input_signature={"audio_duration_s": [5, 10], "batch_size": [1]},
                    source_url="https://docs.ultralytics.com/zh/models/",
                    pretrained_weights=weights,
                    neighbor_models=[],
                    lighter_alternative="whisper-medium",
                    heavier_alternative=None,
                )
            )
        return specs

    @staticmethod
    def _required_pretrained_weights(spec: ModelSpec) -> str:
        """Use registry-grounded pretrained_weights only (no name guessing)."""
        model_ref = (spec.pretrained_weights or "").strip()
        if not model_ref:
            raise ValueError(
                f"ModelSpec.pretrained_weights is required for whisper model '{spec.name}'"
            )
        return model_ref

    @staticmethod
    def _hf_whisper_repo_id(model_ref: str) -> str:
        if "/" in model_ref:
            return model_ref
        return f"openai/whisper-{model_ref}"

    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py for Whisper fine-tuning."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        model_ref = self._required_pretrained_weights(spec)
        model_name = self._hf_whisper_repo_id(model_ref)

        return f'''#!/usr/bin/env python3
"""Fine-tuning script for {spec.name} ASR model.

Note: Whisper fine-tuning typically uses the Hugging Face transformers library.
This script provides a basic template - customize for your dataset.
"""
import json
import sys
import inspect
import csv
from pathlib import Path

import torch
import yaml
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio
from torch.utils.data import Dataset

# Configuration
model_name = "{model_name}"
epochs = {hp.get("epochs", 3)}
batch_size = {hp.get("batch_size", 8)}
lr = {hp.get("lr", 1e-5)}
sample_rate = {hp.get("sample_rate", 16000)}

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
            "accelerate>=0.26.0 is required for Seq2SeqTrainer.",
            exc,
        )
    if _version_tuple(getattr(accelerate, "__version__", "0.0.0")) < (0, 26, 0):
        fail(
            "incompatible_dependency",
            f"accelerate>=0.26.0 required, found {{getattr(accelerate, '__version__', 'unknown')}}",
        )

def resolve_audio_root(dataset_spec: str) -> Path:
    p = Path(dataset_spec)
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {{}}
        except Exception as exc:
            fail("dataset_contract_mismatch", f"Failed to parse dataset config file: {{p}}", exc)
        value = cfg.get("audio_root") or cfg.get("dataset_path")
        if not value:
            fail(
                "dataset_contract_mismatch",
                "config/data.yaml must contain `audio_root` or `dataset_path` for audio datasets.",
            )
        p = Path(str(value))

    p = p.expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()

    # If metadata file is passed, use its parent directory as audio root.
    if p.is_file():
        if p.suffix.lower() in (".csv", ".tsv", ".json", ".jsonl"):
            p = p.parent
        else:
            fail(
                "dataset_contract_mismatch",
                f"Unsupported audio dataset spec file: {{p.name}}; expected yaml/csv/tsv/json/jsonl.",
            )

    if not p.exists() or not p.is_dir():
        fail("dataset_not_found", f"Resolved audio root does not exist or is not a directory: {{p}}")
    return p

# Load processor and model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load dataset from config/data.yaml (workspace contract) and normalize to audio root
dataset_spec = "{context.dataset_path}"
audio_root = resolve_audio_root(dataset_spec)

def load_manifest_rows(split_dir: Path):
    meta = split_dir / "metadata.csv"
    if not meta.exists():
        return []
    rows = []
    with meta.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            file_name = (r.get("file_name") or "").strip()
            text = (r.get("text") or "").strip()
            if not file_name or not text:
                continue
            wav_path = split_dir / file_name
            if wav_path.exists():
                rows.append({{"audio_path": str(wav_path), "text": text}})
    return rows

class ManifestASRDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        item = self.rows[idx]
        wav, sr = sf.read(item["audio_path"], dtype="float32", always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            wav = resample_poly(wav, sample_rate, sr).astype("float32")
        input_features = self.processor(
            np.asarray(wav, dtype="float32"),
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(item["text"]).input_ids
        return {{"input_features": input_features, "labels": labels}}

def collate_fn(batch):
    feats = [{{"input_features": b["input_features"]}} for b in batch]
    labels = [{{"input_ids": b["labels"]}} for b in batch]
    feat_batch = processor.feature_extractor.pad(feats, return_tensors="pt")
    label_batch = processor.tokenizer.pad(labels, return_tensors="pt")
    feat_batch["labels"] = label_batch["input_ids"]
    return feat_batch

train_ds = None
eval_ds = None
try:
    # Fast path: HuggingFace audiofolder
    dataset = load_dataset("audiofolder", data_dir=str(audio_root))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        if "text" not in batch:
            fail("dataset_contract_mismatch", "Audio sample is missing `text` transcript field.")
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch
    remove_cols = dataset.column_names["train"] if "train" in dataset.column_names else next(iter(dataset.column_names.values()))
    dataset = dataset.map(prepare_dataset, remove_columns=remove_cols)
    train_ds = dataset["train"]
    eval_ds = dataset.get("validation") or dataset.get("test")
except Exception:
    # Fallback path: torchaudio + CSV manifest (no audiofolder/torchcodec dependency)
    train_rows = load_manifest_rows(audio_root / "train")
    val_rows = load_manifest_rows(audio_root / "validation") or load_manifest_rows(audio_root / "val")
    if not train_rows:
        fail(
            "dataset_contract_mismatch",
            "Failed to build audio dataset. Need either audiofolder-compatible structure "
            "or train/metadata.csv with (file_name,text).",
        )
    train_ds = ManifestASRDataset(train_rows, processor)
    eval_ds = ManifestASRDataset(val_rows, processor) if val_rows else None

# Training arguments
check_accelerate()
ta_kwargs = {{
    "output_dir": "outputs/checkpoints",
    "per_device_train_batch_size": batch_size,
    "learning_rate": lr,
    "num_train_epochs": epochs,
    "save_strategy": "epoch",
    "predict_with_generate": True,
    "generation_max_length": 225,
}}
ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
if eval_ds is not None:
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
else:
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "no"
training_args = Seq2SeqTrainingArguments(**ta_kwargs)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=processor.feature_extractor,
    data_collator=collate_fn,
)

# Train
trainer.train()

# Save model
trainer.save_model("outputs/best")
processor.save_pretrained("outputs/best")

# Output
output = {{
    "status": "success",
    "model_path": "outputs/best",
    "metrics": {{}},
}}
print(json.dumps(output))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py for Whisper inference."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        model_ref = self._required_pretrained_weights(spec)

        return f'''#!/usr/bin/env python3
"""Edge fused eval script for {spec.name} (edge_eval_v1)."""
import json
import csv
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = Path("outputs/best")
use_pretrained = not model_path.exists()

if use_pretrained:
    import whisper
    model = whisper.load_model("{model_ref}")
else:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

model = model.to(device)
model.eval()

sample_rate = {hp.get("sample_rate", 16000)}

def _read_eval_manifest(cfg_path: Path):
    if not cfg_path.exists():
        return [], "none"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {{}}
    root = Path(str(cfg.get("audio_root") or cfg.get("dataset_path") or os.getenv("EDGE_DATASET_DIR", "dataset")))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    split = "test"
    meta = root / "test" / "metadata.csv"
    if not meta.exists():
        split = "validation"
        meta = root / "validation" / "metadata.csv"
    if not meta.exists():
        split = "val"
        meta = root / "val" / "metadata.csv"
    if not meta.exists():
        return [], "none"
    rows = []
    with meta.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = (r.get("file_name") or "").strip()
            tx = (r.get("text") or "").strip()
            if not fn or not tx:
                continue
            wav = meta.parent / fn
            if wav.exists():
                rows.append((str(wav), tx))
    return rows, split

def _wer(ref: str, hyp: str) -> float:
    r = ref.strip().lower().split()
    h = hyp.strip().lower().split()
    if not r:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return float(dp[len(r)][len(h)] / max(1, len(r)))

def _transcribe_file(wav_path: str) -> str:
    if use_pretrained:
        out = model.transcribe(wav_path)
        return str((out or {{}}).get("text", "")).strip()
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    feats = processor(np.asarray(audio, dtype="float32"), sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        gen = model.generate(feats)
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

def _bench_once(dummy_audio):
    t0 = time.perf_counter()
    if use_pretrained:
        _ = model.transcribe(dummy_audio)
    else:
        feats = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            _ = model.generate(feats)
    return (time.perf_counter() - t0) * 1000.0

metrics = {{}}
if os.getenv("EDGE_EVAL_DATASET", "0") == "1":
    rows, split = _read_eval_manifest(Path(os.getenv("EDGE_DATASET_CONFIG", "config/data.yaml")))
    if rows:
        max_samples = int(os.getenv("EDGE_EVAL_MAX_SAMPLES", "64"))
        wers = []
        for wav_path, ref in rows[:max_samples]:
            hyp = _transcribe_file(wav_path)
            wers.append(_wer(ref, hyp))
        metrics["WER"] = float(sum(wers) / len(wers)) if wers else 1.0
        metrics["edge_eval_samples"] = float(len(wers))
        metrics["edge_eval_split"] = split

duration_s = 10
dummy_audio = np.random.randn(sample_rate * duration_s).astype(np.float32)
for _ in range(3):
    _ = _bench_once(dummy_audio)

latencies = [_bench_once(dummy_audio) for _ in range(20)]
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
        model_ref = self._required_pretrained_weights(model_spec)
        model_name = self._hf_whisper_repo_id(model_ref)
        if runtime_id == RuntimeId.ONNXRUNTIME:
            return wrap_with_tegrastats(self._render_profile_onnxruntime(model_name, warmup, iterations))
        if runtime_id == RuntimeId.TENSORRT:
            return wrap_with_tegrastats(self._render_profile_tensorrt(model_name, iterations))
        return ""

    def _render_profile_onnxruntime(self, model_name: str, warmup: int, iterations: int) -> str:
        return f'''#!/usr/bin/env python3
import json
import time
import numpy as np

try:
    import torch
    import onnxruntime as ort
    from transformers import WhisperForConditionalGeneration
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

model_name = "{model_name}"
try:
    model = WhisperForConditionalGeneration.from_pretrained(model_name).eval().cpu()
    input_features = torch.randn(1, 80, 3000, dtype=torch.float32)
    decoder_input_ids = torch.ones((1, 2), dtype=torch.long)

    class WhisperWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_features, decoder_input_ids):
            out = self.m(input_features=input_features, decoder_input_ids=decoder_input_ids)
            return out.logits

    wrapped = WhisperWrapper(model).eval()
    torch.onnx.export(
        wrapped,
        (input_features, decoder_input_ids),
        "model.onnx",
        input_names=["input_features", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={{
            "input_features": {{0: "batch"}},
            "decoder_input_ids": {{0: "batch", 1: "decoder_seq"}},
            "logits": {{0: "batch"}},
        }},
        opset_version=17,
    )
    sess = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

feeds = {{
    "input_features": np.random.randn(1, 80, 3000).astype(np.float32),
    "decoder_input_ids": np.ones((1, 2), dtype=np.int64),
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

    def _render_profile_tensorrt(self, model_name: str, iterations: int) -> str:
        return f'''#!/usr/bin/env python3
import json
import re
import shutil
import subprocess

try:
    import torch
    from transformers import WhisperForConditionalGeneration
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

trtexec = shutil.which("trtexec")
if not trtexec:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":"trtexec not found in PATH"}}))
    raise SystemExit(1)

model_name = "{model_name}"
try:
    model = WhisperForConditionalGeneration.from_pretrained(model_name).eval().cpu()
    input_features = torch.randn(1, 80, 3000, dtype=torch.float32)
    decoder_input_ids = torch.ones((1, 2), dtype=torch.long)

    class WhisperWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_features, decoder_input_ids):
            out = self.m(input_features=input_features, decoder_input_ids=decoder_input_ids)
            return out.logits

    wrapped = WhisperWrapper(model).eval()
    torch.onnx.export(
        wrapped,
        (input_features, decoder_input_ids),
        "model.onnx",
        input_names=["input_features", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={{
            "input_features": {{0: "batch"}},
            "decoder_input_ids": {{0: "batch", 1: "decoder_seq"}},
            "logits": {{0: "batch"}},
        }},
        opset_version=17,
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export","error_message":str(e)}}))
    raise SystemExit(1)

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
    "--minShapes=input_features:1x80x3000,decoder_input_ids:1x2",
    "--optShapes=input_features:1x80x3000,decoder_input_ids:1x2",
    "--maxShapes=input_features:1x80x3000,decoder_input_ids:1x2",
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


# ---------------------------------------------------------------------------
# SpeechBrainPlugin: SpeechBrain speech processing models
# ---------------------------------------------------------------------------

class SpeechBrainPlugin(BaseFamilyPlugin):
    """Family plugin for SpeechBrain models."""

    @property
    def family_id(self) -> str:
        return "speechbrain"

    @property
    def display_name(self) -> str:
        return "SpeechBrain"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.AUDIO]

    def get_model_specs(self) -> List[ModelSpec]:
        """Return SpeechBrain model specs."""
        specs = [
            ModelSpec(
                name="ecapa-tdnn-voxceleb",
                family_id="speechbrain",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.AUDIO_CLASSIFICATION],
                params_m=6.2,
                default_input_shape=[1, 16000],  # [batch, samples]
                default_hyperparams={
                    "epochs": 10,
                    "batch_size": 16,
                    "lr": 0.001,
                    "sample_rate": 16000,
                },
                pip_packages=["speechbrain", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "audio_duration_s": [1, 3, 5],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
                pretrained_weights="speechbrain/spkrec-ecapa-voxceleb",
                neighbor_models=["xvector-voxceleb"],
                lighter_alternative="xvector-voxceleb",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="xvector-voxceleb",
                family_id="speechbrain",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.AUDIO_CLASSIFICATION],
                params_m=4.2,
                default_input_shape=[1, 16000],
                default_hyperparams={
                    "epochs": 10,
                    "batch_size": 16,
                    "lr": 0.001,
                    "sample_rate": 16000,
                },
                pip_packages=["speechbrain", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.RASPBERRY_PI,
                    DeviceClass.X86_CPU,
                ],
                benchmark_input_signature={
                    "audio_duration_s": [1, 3, 5],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/speechbrain/spkrec-xvect-voxceleb",
                pretrained_weights="speechbrain/spkrec-xvect-voxceleb",
                neighbor_models=["ecapa-tdnn-voxceleb"],
                lighter_alternative=None,
                heavier_alternative="ecapa-tdnn-voxceleb",
            ),
            ModelSpec(
                name="crdnn-commands",
                family_id="speechbrain",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.AUDIO_CLASSIFICATION],
                params_m=2.1,
                default_input_shape=[1, 16000],
                default_hyperparams={
                    "epochs": 20,
                    "batch_size": 32,
                    "lr": 0.001,
                    "sample_rate": 16000,
                },
                pip_packages=["speechbrain", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.RASPBERRY_PI,
                    DeviceClass.ANDROID,
                ],
                benchmark_input_signature={
                    "audio_duration_s": [1],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/speechbrain/google_speech_command_xvector",
                pretrained_weights="speechbrain/google_speech_command_xvector",
                neighbor_models=["xvector-voxceleb"],
                lighter_alternative=None,
                heavier_alternative="xvector-voxceleb",
            ),
            ModelSpec(
                name="lang-id-ecapa",
                family_id="speechbrain",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.AUDIO_CLASSIFICATION],
                params_m=6.0,
                default_input_shape=[1, 16000],
                default_hyperparams={
                    "epochs": 10,
                    "batch_size": 16,
                    "lr": 0.001,
                    "sample_rate": 16000,
                },
                pip_packages=["speechbrain", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "audio_duration_s": [1, 3, 5],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa",
                pretrained_weights="speechbrain/lang-id-commonlanguage_ecapa",
                neighbor_models=["ecapa-tdnn-voxceleb"],
                lighter_alternative="xvector-voxceleb",
                heavier_alternative=None,
            ),
            ModelSpec(
                name="resnet-speaker",
                family_id="speechbrain",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.AUDIO_CLASSIFICATION],
                params_m=15.0,
                default_input_shape=[1, 16000],
                default_hyperparams={
                    "epochs": 10,
                    "batch_size": 8,
                    "lr": 0.001,
                    "sample_rate": 16000,
                },
                pip_packages=["speechbrain", "torch", "torchaudio"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                benchmark_input_signature={
                    "audio_duration_s": [1, 3, 5],
                    "batch_size": [1],
                },
                source_url="https://huggingface.co/speechbrain/spkrec-resnet-voxceleb",
                pretrained_weights="speechbrain/spkrec-resnet-voxceleb",
                neighbor_models=["ecapa-tdnn-voxceleb"],
                lighter_alternative="ecapa-tdnn-voxceleb",
                heavier_alternative=None,
            ),
        ]
        specs.extend(self._build_extended_specs())
        for spec in specs:
            spec.edge_compatible = True
            if RuntimeId.TENSORRT not in spec.supported_runtimes:
                spec.supported_runtimes.append(RuntimeId.TENSORRT)
        return specs

    def _build_extended_specs(self) -> List[ModelSpec]:
        entries = [
            # CRDNN English repo is unavailable; use the maintained CommonVoice EN ASR model.
            ("asr-crdnn-commonvoice", "speechbrain/asr-wav2vec2-commonvoice-en", 130.0),
            ("asr-transformer-transformerlm-librispeech", "speechbrain/asr-transformer-transformerlm-librispeech", 120.0),
            ("vad-crdnn-libriparty", "speechbrain/vad-crdnn-libriparty", 18.0),
        ]
        specs: List[ModelSpec] = []
        for name, weights, params in entries:
            task = TaskType.SPEECH_RECOGNITION if name.startswith("asr-") else TaskType.AUDIO_CLASSIFICATION
            specs.append(
                ModelSpec(
                    name=name,
                    family_id="speechbrain",
                    modality=Modality.AUDIO,
                    supported_tasks=[task],
                    params_m=params,
                    default_input_shape=[1, 16000],
                    default_hyperparams={"epochs": 10, "batch_size": 8, "lr": 0.001, "sample_rate": 16000},
                    pip_packages=["speechbrain", "torch", "torchaudio"],
                    supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                    supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                    edge_compatible=True,
                    recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_GPU],
                    benchmark_input_signature={"audio_duration_s": [1, 3, 5], "batch_size": [1]},
                    source_url="https://huggingface.co/speechbrain",
                    pretrained_weights=weights,
                    neighbor_models=[],
                    lighter_alternative=None,
                    heavier_alternative=None,
                )
            )
        return specs

    def render_train_script(self, context: TemplateContext) -> str:
        """Generate train.py for SpeechBrain models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}

        return f'''#!/usr/bin/env python3
"""Training script for {spec.name} audio model.

Note: SpeechBrain uses YAML-based recipes. This is a simplified template.
For full functionality, use the SpeechBrain recipe system.
"""
import json
import sys
from pathlib import Path

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Configuration
model_hub_id = "{spec.pretrained_weights}"
epochs = {hp.get("epochs", 10)}
batch_size = {hp.get("batch_size", 16)}
lr = {hp.get("lr", 0.001)}

# Load pretrained model for fine-tuning
classifier = EncoderClassifier.from_hparams(
    source=model_hub_id,
    savedir="outputs/pretrained",
)

# Note: Full fine-tuning requires SpeechBrain recipe YAML configuration
# This template shows the basic structure

# For demonstration, save the pretrained model
classifier.save("outputs/best")

output = {{
    "status": "success",
    "model_path": "outputs/best",
    "metrics": {{}},
}}
print(json.dumps(output))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        """Generate infer.py for SpeechBrain models."""
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}

        return f'''#!/usr/bin/env python3
"""Inference benchmark script for {spec.name}."""
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
from speechbrain.pretrained import EncoderClassifier

model_path = Path("outputs/best")
if model_path.exists():
    classifier = EncoderClassifier.from_hparams(
        source=str(model_path),
        run_opts={{"device": str(device)}},
    )
else:
    classifier = EncoderClassifier.from_hparams(
        source="{spec.pretrained_weights}",
        savedir="outputs/pretrained",
        run_opts={{"device": str(device)}},
    )

# Create dummy audio
sample_rate = {hp.get("sample_rate", 16000)}
duration_s = 3
dummy_audio = torch.randn(1, sample_rate * duration_s).to(device)

# Warmup
for _ in range(5):
    _ = classifier.encode_batch(dummy_audio)

# Benchmark
n_runs = 50
latencies = []
for _ in range(n_runs):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = classifier.encode_batch(dummy_audio)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)

avg_latency = float(np.mean(latencies))
std_latency = float(np.std(latencies))

# Memory
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

    def get_supported_profile_runtimes(self, model_spec: ModelSpec) -> List[RuntimeId]:
        # ASR recipes require task-specific decoding graphs and are not covered by
        # the generic encoder-export path yet, so skip profile runtimes for now.
        if TaskType.SPEECH_RECOGNITION in model_spec.supported_tasks:
            return []
        # VAD checkpoints need task-specific inference APIs instead of the
        # generic EncoderClassifier.encode_batch path.
        if model_spec.name.startswith("vad-"):
            return []
        # Stable default: profile SpeechBrain with PyTorch only.
        # ORT/TRT paths are currently too noisy due to frontend/export
        # incompatibilities (STFT complex path and dependency variance).
        if RuntimeId.PYTORCH in model_spec.supported_runtimes:
            return [RuntimeId.PYTORCH]
        return []

    def render_profile_benchmark_script(
        self,
        model_spec: ModelSpec,
        runtime_id: RuntimeId,
        imgsz: int,
        warmup: int,
        iterations: int,
    ) -> str:
        _ = imgsz
        model_name = model_spec.pretrained_weights or model_spec.name
        sample_rate = int(model_spec.default_hyperparams.get("sample_rate", 16000))
        sample_len = sample_rate * 3
        if runtime_id == RuntimeId.PYTORCH:
            return wrap_with_tegrastats(self._render_profile_pytorch(model_name, sample_len, warmup, iterations))
        if runtime_id == RuntimeId.ONNXRUNTIME:
            return wrap_with_tegrastats(self._render_profile_onnxruntime(model_name, sample_len, warmup, iterations))
        if runtime_id == RuntimeId.TENSORRT:
            return wrap_with_tegrastats(self._render_profile_tensorrt(model_name, sample_len, iterations))
        return ""

    def _render_profile_pytorch(
        self,
        model_name: str,
        sample_len: int,
        warmup: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import time
import numpy as np
import inspect
import os
import tempfile

try:
    import torch
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

# Compatibility shim:
# speechbrain may pass `use_auth_token` to hf_hub_download, while newer
# huggingface_hub versions expect `token`.
try:
    import huggingface_hub as _hf
    if "use_auth_token" not in inspect.signature(_hf.hf_hub_download).parameters:
        _orig_hf_hub_download = _hf.hf_hub_download
        def _hf_hub_download_compat(*args, **kwargs):
            filename = kwargs.get("filename", "")
            if not filename and len(args) >= 2:
                filename = str(args[1])
            if "use_auth_token" in kwargs:
                token = kwargs.pop("use_auth_token")
                if "token" not in kwargs:
                    kwargs["token"] = token
            try:
                return _orig_hf_hub_download(*args, **kwargs)
            except Exception as e:
                # Some SpeechBrain model repos do not include custom.py. Older
                # hub stacks tolerated this; newer ones raise hard 404.
                msg = str(e)
                if ("custom.py" in filename) and ("404" in msg or "Entry Not Found" in msg):
                    stub = os.path.join(tempfile.gettempdir(), "speechbrain_custom_stub.py")
                    if not os.path.exists(stub):
                        with open(stub, "w", encoding="utf-8") as f:
                            f.write("# Auto-generated fallback for missing custom.py\\n")
                    return stub
                raise
        _hf.hf_hub_download = _hf_hub_download_compat
        try:
            import huggingface_hub.file_download as _hf_file_download
            _hf_file_download.hf_hub_download = _hf_hub_download_compat
        except Exception:
            pass
except Exception:
    pass

try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "{model_name}"
sample_len = {sample_len}

try:
    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        savedir="outputs/pretrained",
        run_opts={{"device": str(device)}},
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"load","error_message":str(e)}}))
    raise SystemExit(1)

try:
    wavs = torch.randn(1, sample_len, dtype=torch.float32, device=device)
    for _ in range({warmup}):
        _ = classifier.encode_batch(wavs)

    latencies = []
    for _ in range({iterations}):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = classifier.encode_batch(wavs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies, dtype=np.float64)
    process = psutil.Process()
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)
    gpu_memory_mb = (
        torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else None
    )

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
        "peak_memory_mb": float(peak_memory_mb),
        "gpu_memory_mb": float(gpu_memory_mb) if gpu_memory_mb is not None else None,
        "raw_latencies": arr.tolist(),
    }}))
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":str(e)}}))
    raise SystemExit(1)
'''

    def _render_profile_onnxruntime(
        self,
        model_name: str,
        sample_len: int,
        warmup: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import time
import numpy as np
import inspect
import os
import tempfile

try:
    import torch
    import onnxruntime as ort
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

# Compatibility shim:
# speechbrain may pass `use_auth_token` to hf_hub_download, while newer
# huggingface_hub versions expect `token`.
try:
    import huggingface_hub as _hf
    if "use_auth_token" not in inspect.signature(_hf.hf_hub_download).parameters:
        _orig_hf_hub_download = _hf.hf_hub_download
        def _hf_hub_download_compat(*args, **kwargs):
            filename = kwargs.get("filename", "")
            if not filename and len(args) >= 2:
                filename = str(args[1])
            if "use_auth_token" in kwargs:
                token = kwargs.pop("use_auth_token")
                if "token" not in kwargs:
                    kwargs["token"] = token
            try:
                return _orig_hf_hub_download(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                if ("custom.py" in filename) and ("404" in msg or "Entry Not Found" in msg):
                    stub = os.path.join(tempfile.gettempdir(), "speechbrain_custom_stub.py")
                    if not os.path.exists(stub):
                        with open(stub, "w", encoding="utf-8") as f:
                            f.write("# Auto-generated fallback for missing custom.py\\n")
                    return stub
                raise
        _hf.hf_hub_download = _hf_hub_download_compat
        try:
            import huggingface_hub.file_download as _hf_file_download
            _hf_file_download.hf_hub_download = _hf_hub_download_compat
        except Exception:
            pass
except Exception:
    pass

try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

model_name = "{model_name}"
sample_len = {sample_len}
try:
    classifier = EncoderClassifier.from_hparams(source=model_name, savedir="outputs/pretrained")

    class SBWrapper(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, wavs):
            out = self.c.encode_batch(wavs)
            if isinstance(out, tuple):
                out = out[0]
            if hasattr(out, "dim") and out.dim() == 3 and out.shape[1] == 1:
                out = out[:, 0, :]
            return out

    wrapped = SBWrapper(classifier).eval()
    wavs = torch.randn(1, sample_len, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        wavs,
        "model.onnx",
        input_names=["wavs"],
        output_names=["embedding"],
        dynamic_axes={{"wavs": {{0: "batch"}}, "embedding": {{0: "batch"}}}},
        opset_version=17,
    )
    sess = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export_or_load","error_message":str(e)}}))
    raise SystemExit(1)

dummy = np.random.randn(1, sample_len).astype(np.float32)
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
        self,
        model_name: str,
        sample_len: int,
        iterations: int,
    ) -> str:
        return f'''#!/usr/bin/env python3
import json
import re
import shutil
import subprocess

try:
    import torch
    from speechbrain.pretrained import EncoderClassifier
    import psutil
except ImportError as e:
    print(json.dumps({{"status":"error","error_type":"import","error_message":str(e)}}))
    raise SystemExit(1)

trtexec = shutil.which("trtexec")
if not trtexec:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":"trtexec not found in PATH"}}))
    raise SystemExit(1)

model_name = "{model_name}"
sample_len = {sample_len}
try:
    classifier = EncoderClassifier.from_hparams(source=model_name, savedir="outputs/pretrained")

    class SBWrapper(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, wavs):
            out = self.c.encode_batch(wavs)
            if isinstance(out, tuple):
                out = out[0]
            if hasattr(out, "dim") and out.dim() == 3 and out.shape[1] == 1:
                out = out[:, 0, :]
            return out

    wrapped = SBWrapper(classifier).eval()
    wavs = torch.randn(1, sample_len, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        wavs,
        "model.onnx",
        input_names=["wavs"],
        output_names=["embedding"],
        dynamic_axes={{"wavs": {{0: "batch"}}, "embedding": {{0: "batch"}}}},
        opset_version=17,
    )
except Exception as e:
    print(json.dumps({{"status":"error","error_type":"export","error_message":str(e)}}))
    raise SystemExit(1)

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
    "--shapes=wavs:1x{sample_len}",
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


# ---------------------------------------------------------------------------
# TinyAudioASRPlugin: artifact-first ASR for tiny edge experiments
# ---------------------------------------------------------------------------

class TinyAudioASRPlugin(BaseFamilyPlugin):
    """Tiny supervised ASR baseline for edge-first artifact generation.

    This family intentionally avoids Whisper-sized seq2seq models.  It produces
    a compact character-CTC PyTorch model and ONNX export from manifest-style
    datasets such as audio-mini-asr.
    """

    @property
    def family_id(self) -> str:
        return "tiny_audio_asr"

    @property
    def display_name(self) -> str:
        return "Tiny Audio ASR"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.AUDIO]

    def get_model_specs(self) -> List[ModelSpec]:
        return [
            ModelSpec(
                name="tiny_char_ctc_asr",
                family_id="tiny_audio_asr",
                modality=Modality.AUDIO,
                supported_tasks=[TaskType.SPEECH_RECOGNITION],
                params_m=0.35,
                default_input_shape=[1, 16000],
                default_hyperparams={
                    "epochs": 30,
                    "batch_size": 2,
                    "lr": 1e-3,
                    "sample_rate": 16000,
                    "max_audio_seconds": 4.0,
                },
                pip_packages=["torch", "numpy", "soundfile", "pyyaml"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.PYTORCH, RuntimeId.ONNXRUNTIME],
                edge_compatible=True,
                recommended_devices=[
                    DeviceClass.JETSON,
                    DeviceClass.RASPBERRY_PI,
                    DeviceClass.X86_CPU,
                ],
                benchmark_input_signature={
                    "audio_duration_s": [1, 2, 4],
                    "batch_size": [1],
                },
                source_url="internal://edgecraft/tiny-char-ctc-asr",
                pretrained_weights=None,
                neighbor_models=[],
                lighter_alternative=None,
                heavier_alternative="whisper-tiny",
            )
        ]

    def render_train_script(self, context: TemplateContext) -> str:
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        return f'''#!/usr/bin/env python3
"""Artifact-first tiny character CTC ASR training script for {spec.name}."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sample_rate = {int(hp.get("sample_rate", 16000))}
epochs = {int(hp.get("epochs", 30))}
batch_size = {int(hp.get("batch_size", 2))}
lr = {float(hp.get("lr", 1e-3))}
max_audio_seconds = {float(hp.get("max_audio_seconds", 4.0))}
max_len = int(sample_rate * max_audio_seconds)

def fail(error_code, error, detail=None):
    payload = {{"status": "error", "error_code": error_code, "error": error}}
    if detail is not None:
        payload["detail"] = str(detail)
    print(json.dumps(payload))
    sys.exit(1)

def resolve_audio_root(dataset_spec):
    p = Path(dataset_spec)
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {{}}
        except Exception as exc:
            fail("dataset_contract_mismatch", f"Could not parse dataset config: {{p}}", exc)
        p = Path(str(cfg.get("audio_root") or cfg.get("dataset_path") or ""))
    if not p:
        fail("dataset_contract_mismatch", "Missing audio_root/dataset_path in config/data.yaml")
    p = p.expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if p.is_file():
        p = p.parent
    if p.name.lower() in {{"train", "validation", "val", "test"}}:
        p = p.parent
    if not p.exists():
        fail("dataset_not_found", f"Audio root not found: {{p}}")
    return p

def read_rows(root, split_names):
    rows = []
    for split in split_names:
        meta = root / split / "metadata.csv"
        if not meta.exists():
            continue
        with meta.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                file_name = (r.get("file_name") or "").strip()
                text = (r.get("text") or "").strip().lower()
                if not file_name or not text:
                    continue
                wav = meta.parent / file_name
                if wav.exists():
                    rows.append((str(wav), text))
    return rows

def read_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    if sr != sample_rate and len(wav) > 1:
        new_len = max(1, int(round(len(wav) * sample_rate / float(sr))))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
    if len(wav) > max_len:
        wav = wav[:max_len]
    if len(wav) < max_len:
        wav = np.pad(wav, (0, max_len - len(wav)))
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1e-6:
        wav = wav / peak
    return wav.astype(np.float32)

class TinyASRDataset(Dataset):
    def __init__(self, rows, char_to_id):
        self.rows = rows
        self.char_to_id = char_to_id
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        path, text = self.rows[idx]
        wav = torch.from_numpy(read_audio(path))
        labels = torch.tensor([self.char_to_id[c] for c in text if c in self.char_to_id], dtype=torch.long)
        if labels.numel() == 0:
            labels = torch.tensor([self.char_to_id[" "]], dtype=torch.long)
        return wav, labels, text

def collate(batch):
    wavs, labels, texts = zip(*batch)
    wavs = torch.stack(list(wavs), dim=0)
    label_lengths = torch.tensor([len(x) for x in labels], dtype=torch.long)
    targets = torch.cat(list(labels), dim=0)
    return wavs, targets, label_lengths, texts

class TinyCharCTC(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(64, 96, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(192, vocab_size)
    def forward(self, wav):
        x = wav.unsqueeze(1)
        x = self.conv(x).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.fc(x)

def greedy_decode(logits, id_to_char):
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    outs = []
    for seq in pred:
        prev = 0
        chars = []
        for tok in seq:
            if tok != 0 and tok != prev:
                chars.append(id_to_char.get(int(tok), ""))
            prev = tok
        outs.append("".join(chars).strip())
    return outs

def wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / max(1, len(r))

audio_root = resolve_audio_root("{context.dataset_path}")
train_rows = read_rows(audio_root, ["train", "training"])
val_rows = read_rows(audio_root, ["validation", "val", "test"])
if not train_rows:
    fail("dataset_contract_mismatch", "Need train/metadata.csv with file_name,text for supervised tiny ASR.")

alphabet = sorted(set(" ".join(text for _, text in train_rows + val_rows)))
if " " not in alphabet:
    alphabet.append(" ")
id_to_char = {{0: ""}}
for idx, ch in enumerate(alphabet, start=1):
    id_to_char[idx] = ch
char_to_id = {{ch: idx for idx, ch in id_to_char.items() if idx != 0}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCharCTC(vocab_size=len(id_to_char)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
loader = DataLoader(TinyASRDataset(train_rows, char_to_id), batch_size=batch_size, shuffle=True, collate_fn=collate)

model.train()
last_loss = math.nan
for _epoch in range(max(1, epochs)):
    for wavs, targets, target_lengths, _texts in loader:
        wavs = wavs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        logits = model(wavs)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full((wavs.shape[0],), logits.shape[1], dtype=torch.long, device=device)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())

metrics = {{"Loss": last_loss if math.isfinite(last_loss) else 0.0}}
if val_rows:
    model.eval()
    eval_loader = DataLoader(TinyASRDataset(val_rows, char_to_id), batch_size=1, shuffle=False, collate_fn=collate)
    wers = []
    with torch.no_grad():
        for wavs, _targets, _target_lengths, texts in eval_loader:
            logits = model(wavs.to(device))
            hyp = greedy_decode(logits, id_to_char)[0]
            wers.append(wer(texts[0], hyp))
    metrics["WER"] = float(sum(wers) / len(wers)) if wers else 1.0

Path("outputs").mkdir(exist_ok=True)
payload = {{
    "state_dict": model.cpu().state_dict(),
    "id_to_char": id_to_char,
    "sample_rate": sample_rate,
    "max_len": max_len,
    "model_name": "{spec.name}",
}}
torch.save(payload, "outputs/best.pt")
model.eval()
dummy = torch.zeros(1, max_len, dtype=torch.float32)
try:
    torch.onnx.export(
        model,
        dummy,
        "outputs/best.onnx",
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={{"audio": {{0: "batch", 1: "samples"}}, "logits": {{0: "batch", 1: "frames"}}}},
        opset_version=17,
        dynamo=False,
    )
except Exception as exc:
    metrics["onnx_export_error"] = str(exc)

print(json.dumps({{"status": "success", "model_path": "outputs/best.pt", "metrics": metrics}}))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        return f'''#!/usr/bin/env python3
"""Edge evaluation script for {spec.name} tiny character CTC ASR."""
import csv
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import yaml

sample_rate = {int(hp.get("sample_rate", 16000))}

class TinyCharCTC(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(64, 96, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(192, vocab_size)
    def forward(self, wav):
        x = wav.unsqueeze(1)
        x = self.conv(x).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.fc(x)

def read_audio(path, max_len):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sample_width == 1:
        wav = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        wav = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        wav = np.zeros(max_len, dtype=np.float32)
    if channels > 1 and wav.size:
        wav = wav.reshape(-1, channels).mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    if sr != sample_rate and len(wav) > 1:
        new_len = max(1, int(round(len(wav) * sample_rate / float(sr))))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
    if len(wav) > max_len:
        wav = wav[:max_len]
    if len(wav) < max_len:
        wav = np.pad(wav, (0, max_len - len(wav)))
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1e-6:
        wav = wav / peak
    return wav.astype(np.float32)

def greedy_decode(logits, id_to_char):
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    outs = []
    for seq in pred:
        prev = 0
        chars = []
        for tok in seq:
            if tok != 0 and tok != prev:
                chars.append(id_to_char.get(int(tok), ""))
            prev = tok
        outs.append("".join(chars).strip())
    return outs

def wer(ref, hyp):
    r = ref.strip().lower().split()
    h = hyp.strip().lower().split()
    if not r:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / max(1, len(r))

def read_eval_rows():
    cfg_path = Path(os.getenv("EDGE_DATASET_CONFIG", "config/data.yaml"))
    if not cfg_path.exists():
        return []
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {{}}
    root = Path(str(cfg.get("audio_root") or cfg.get("dataset_path") or os.getenv("EDGE_DATASET_DIR", "dataset")))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    rows = []
    for split in ("test", "validation", "val", "train"):
        meta = root / split / "metadata.csv"
        if not meta.exists():
            continue
        with meta.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fn = (r.get("file_name") or "").strip()
                tx = (r.get("text") or "").strip().lower()
                wav = meta.parent / fn
                if fn and tx and wav.exists():
                    rows.append((str(wav), tx))
        if rows:
            break
    return rows

checkpoint_path = Path("outputs") / ("best" + ".pt")
if checkpoint_path.suffix != ".pt" or not checkpoint_path.exists():
    # The edge runner may choose an ONNX artifact for deployment, but this
    # PyTorch evaluator still needs the sidecar training checkpoint because it
    # stores the character vocabulary and CTC state_dict.
    checkpoint_path = Path(os.getenv("EDGECRAFT_CHECKPOINT_PATH", str(Path("outputs") / ("best" + ".pt"))))
if checkpoint_path.suffix != ".pt" or not checkpoint_path.exists():
    checkpoint_path = Path("outputs") / ("best" + ".pt")
ckpt = torch.load(checkpoint_path, map_location="cpu")
id_to_char = {{int(k): v for k, v in ckpt["id_to_char"].items()}}
max_len = int(ckpt.get("max_len", sample_rate * 4))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCharCTC(vocab_size=len(id_to_char))
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()

latencies = []
dummy = torch.zeros(1, max_len, dtype=torch.float32, device=device)
with torch.no_grad():
    for _ in range(5):
        _ = model(dummy)
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

metrics = {{}}
if os.getenv("EDGE_EVAL_DATASET", "0") == "1":
    rows = read_eval_rows()
    max_samples = int(os.getenv("EDGE_EVAL_MAX_SAMPLES", "32"))
    wers = []
    with torch.no_grad():
        for wav_path, ref in rows[:max_samples]:
            wav = torch.from_numpy(read_audio(wav_path, max_len)).unsqueeze(0).to(device)
            hyp = greedy_decode(model(wav), id_to_char)[0]
            wers.append(wer(ref, hyp))
    if wers:
        metrics["WER"] = float(sum(wers) / len(wers))
        metrics["edge_eval_samples"] = float(len(wers))

arr = np.array(latencies, dtype=np.float64)
metrics.update({{
    "Latency": float(arr.mean()) if arr.size else 0.0,
    "Memory_mb": float(psutil.Process().memory_info().rss / (1024 * 1024)),
    "latency_p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
    "latency_p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
}})
print(json.dumps({{"status": "success", "metrics": metrics}}))
'''


# Module-level instances
whisper_plugin = WhisperPlugin()
speechbrain_plugin = SpeechBrainPlugin()
tiny_audio_asr_plugin = TinyAudioASRPlugin()
