"""Structured/log anomaly detection baselines.

This family targets supervised block/session-level log anomaly datasets.
It intentionally refuses to hallucinate labels: parquet/csv/json rows must
contain an explicit label column or config/data.yaml must expose label_files
that can be joined through label_join_key, for example BlockId/session_id/trace_id.
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


class StructuredLogPlugin(BaseFamilyPlugin):
    @property
    def family_id(self) -> str:
        return "structured_log"

    @property
    def display_name(self) -> str:
        return "Structured Log Baselines"

    @property
    def modalities(self) -> List[Modality]:
        return [Modality.STRUCTURED, Modality.TEXT, Modality.TIME_SERIES]

    def get_model_specs(self) -> List[ModelSpec]:
        return [
            ModelSpec(
                name="structured_log_baseline",
                family_id="structured_log",
                modality=Modality.STRUCTURED,
                supported_tasks=[TaskType.ANOMALY_DETECTION, TaskType.TEXT_CLASSIFICATION],
                params_m=0.1,
                default_input_shape=[1, 4096],
                default_hyperparams={
                    "max_rows": 200000,
                    "n_features": 262144,
                    "test_size": 0.2,
                },
                pip_packages=["pandas", "pyarrow", "scikit-learn", "skl2onnx", "onnxruntime"],
                supported_export_formats=[ExportFormat.ONNX, ExportFormat.PT],
                supported_runtimes=[RuntimeId.ONNXRUNTIME, RuntimeId.PYTORCH],
                edge_compatible=True,
                recommended_devices=[DeviceClass.JETSON, DeviceClass.X86_CPU, DeviceClass.X86_GPU],
                benchmark_input_signature={"text": ["single log message"], "batch_size": [1]},
                source_url="https://github.com/logpai/loghub",
            )
        ]

    def render_train_script(self, context: TemplateContext) -> str:
        spec = context.model_spec
        hp = {**spec.default_hyperparams, **context.hyperparams}
        max_rows = int(hp.get("max_rows", 200000))
        n_features = int(hp.get("n_features", 262144))
        test_size = float(hp.get("test_size", 0.2))
        return f'''#!/usr/bin/env python3
"""Supervised structured/log anomaly baseline.

Reads config/data.yaml. For block/session-level logs, labels must be present in
the table or joined from an external label file through label_join_key.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

MAX_ROWS = {max_rows}
N_FEATURES = {n_features}
TEST_SIZE = {test_size}


def emit_error(error_code, message, **extra):
    Path("outputs").mkdir(exist_ok=True)
    payload = {{"status": "error", "error_code": error_code, "error": message, **extra}}
    Path("outputs/dataset_contract.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    raise SystemExit(1)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".txt", ".log"):
        try:
            return pd.read_csv(p, sep=None, engine="python")
        except Exception:
            rows = p.read_text(encoding="utf-8", errors="replace").splitlines()[:MAX_ROWS]
            return pd.DataFrame({{"Content": rows}})
    if suffix in (".jsonl", ".json"):
        return pd.read_json(p, lines=suffix == ".jsonl")
    emit_error("dataset_contract_mismatch", f"Unsupported table file format: {{p}}")


def normalize_label(value):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "anomaly", "abnormal", "anomalous"):
        return 1
    if text in ("0", "false", "no", "normal", "-"):
        return 0
    try:
        return int(float(text) != 0.0)
    except Exception:
        return 0


def build_text_series(frame: pd.DataFrame) -> pd.Series:
    text_parts = []
    for name in ("EventId", "EventTemplate", text_col, "Content"):
        if name in frame.columns and name not in text_parts:
            text_parts.append(name)
    if not text_parts:
        emit_error("dataset_contract_mismatch", f"text_column {{text_col!r}} not found in loaded table")
    return frame[text_parts].fillna("").astype(str).agg(" ".join, axis=1)


cfg = yaml.safe_load(Path("config/data.yaml").read_text(encoding="utf-8")) or {{}}
files = cfg.get("files") or (cfg.get("schema") or {{}}).get("files") or []
if not files:
    emit_error("dataset_contract_mismatch", "config/data.yaml must expose schema.files for structured log training")

text_col = cfg.get("text_column") or "Content"
label_col = cfg.get("label_column")
join_key = cfg.get("label_join_key")
label_files = cfg.get("label_files") or []

frames = []
remaining = MAX_ROWS
for f in files:
    df = read_table(f)
    if remaining > 0 and len(df) > remaining:
        df = df.head(remaining)
    frames.append(df)
    remaining -= len(df)
    if remaining <= 0:
        break
data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if data.empty:
    emit_error("dataset_contract_mismatch", "No rows loaded from schema.files")

if label_col and label_col in data.columns:
    labels = data[label_col].map(normalize_label).astype(int)
elif label_files and join_key and join_key in data.columns:
    label_frames = [read_table(f) for f in label_files]
    label_df = pd.concat(label_frames, ignore_index=True)
    candidate_cols = [c for c in label_df.columns if c.lower() in ("label", "labels", "anomaly", "is_anomaly", "target", "y")]
    if join_key not in label_df.columns or not candidate_cols:
        emit_error(
            "dataset_contract_mismatch",
            "External label file must contain label_join_key and a label/anomaly column",
            label_join_key=join_key,
            label_files=label_files,
        )
    label_name = candidate_cols[0]
    label_map = label_df[[join_key, label_name]].drop_duplicates(join_key)
    data = data.merge(label_map, on=join_key, how="inner")
    if data.empty:
        emit_error("dataset_contract_mismatch", "Label join produced zero rows", label_join_key=join_key)
    labels = data[label_name].map(normalize_label).astype(int)
else:
    labels = None

if text_col not in data.columns:
    emit_error("dataset_contract_mismatch", f"text_column {{text_col!r}} not found in loaded table")

raw_rows = int(len(data))
if join_key and join_key in data.columns:
    data = data.copy()
    data["_edgecraft_text"] = build_text_series(data)
    data["_edgecraft_label"] = labels.to_numpy()
    data = (
        data.groupby(join_key, sort=False)
        .agg(
            _edgecraft_text=("_edgecraft_text", lambda parts: " ".join(str(p) for p in parts if str(p))),
            _edgecraft_label=("_edgecraft_label", "max"),
        )
        .reset_index()
    )
    texts = data["_edgecraft_text"].fillna("").astype(str)
    labels = data["_edgecraft_label"].astype(int)
else:
    texts = build_text_series(data).fillna("").astype(str)

if labels is None:
    texts = build_text_series(data).fillna("").astype(str)
    max_normal_rows = int(os.environ.get("EDGECRAFT_LOG_NORMAL_MAX_ROWS", "5000"))
    if len(texts) > max_normal_rows:
        texts = texts.sample(n=max_normal_rows, random_state=42)
    vectorizer = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm="l2")
    X = vectorizer.transform(texts).astype("float32")
    if X.shape[0] == 0:
        emit_error("dataset_contract_mismatch", "No log rows available for normal-only anomaly baseline")
    centroid = np.asarray(X.mean(axis=0)).ravel().astype("float32")
    centroid_norm = float(np.dot(centroid, centroid))
    row_norms = np.asarray(X.multiply(X).sum(axis=1)).ravel().astype("float32")
    dot = np.asarray(X.dot(centroid)).ravel().astype("float32")
    distances = np.sqrt(np.maximum(row_norms - 2.0 * dot + centroid_norm, 0.0))
    threshold = float(np.percentile(distances, 95)) if len(distances) else 0.0
    Path("outputs").mkdir(exist_ok=True)
    payload = {{
        "format": "edgecraft_hash_centroid_v1",
        "n_features": int(N_FEATURES),
        "centroid": centroid,
        "threshold": threshold,
        "token_pattern": r"(?u)\\b\\w\\w+\\b",
        "lowercase": True,
        "norm": "l2",
    }}
    with open("outputs/best.pt", "wb") as f:
        pickle.dump(payload, f)
    Path("outputs/train_summary.json").write_text(json.dumps({{
        "raw_rows": int(len(data)),
        "training_examples": int(len(texts)),
        "artifact_format": payload["format"],
        "label_strategy": "normal_only_centroid",
        "threshold": threshold,
    }}, indent=2), encoding="utf-8")
    print(json.dumps({{
        "status": "success",
        "model_path": "outputs/best.pt",
        "training_examples": int(len(texts)),
        "metrics": {{"F1Score": 0.0}},
        "label_strategy": "normal_only_centroid",
    }}))
    raise SystemExit(0)

if labels.nunique(dropna=False) < 2:
    emit_error("dataset_contract_mismatch", "Supervised labels must contain at least two classes")

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=labels,
)
model = make_pipeline(
    HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm="l2"),
    SGDClassifier(loss="log_loss", max_iter=20, tol=1e-3, random_state=42),
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
f1 = float(f1_score(y_test, pred, zero_division=0))

Path("outputs").mkdir(exist_ok=True)
vectorizer = model.named_steps["hashingvectorizer"]
classifier = model.named_steps["sgdclassifier"]
payload = {{
    "format": "edgecraft_hash_sgd_v1",
    "n_features": int(N_FEATURES),
    "classes": classifier.classes_.astype(int).tolist(),
    "coef": classifier.coef_.astype("float32"),
    "intercept": classifier.intercept_.astype("float32"),
    "token_pattern": r"(?u)\\b\\w\\w+\\b",
    "lowercase": True,
    "norm": "l2",
}}
with open("outputs/best.pt", "wb") as f:
    pickle.dump(payload, f)
Path("outputs/train_summary.json").write_text(json.dumps({{
    "raw_rows": raw_rows,
    "training_examples": int(len(texts)),
    "grouped_by": join_key if join_key and join_key in data.columns else None,
    "positive_examples": int(labels.sum()),
    "negative_examples": int((labels == 0).sum()),
    "artifact_format": payload["format"],
}}, indent=2), encoding="utf-8")

export_warning = ""
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    onx = convert_sklearn(model, initial_types=[("input", StringTensorType([None, 1]))])
    Path("outputs/best.onnx").write_bytes(onx.SerializeToString())
except Exception as exc:
    export_warning = str(exc)

print(json.dumps({{
    "status": "success",
    "model_path": "outputs/best.pt",
    "training_examples": int(len(texts)),
    "metrics": {{"F1Score": f1}},
    "export_warning": export_warning,
}}))
'''

    def render_infer_script(self, context: TemplateContext) -> str:
        return '''#!/usr/bin/env python3
"""Edge latency benchmark for structured log baseline.

The saved best.pt is a lightweight hash-SGD payload, not a sklearn pickle, so
edge containers do not need scikit-learn installed.
"""
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import psutil

model_path = Path("outputs/best.pt")
if not model_path.exists():
    print(json.dumps({"status": "error", "error": "outputs/best.pt not found"}))
    raise SystemExit(1)

with model_path.open("rb") as f:
    payload = pickle.load(f)

if not isinstance(payload, dict) or payload.get("format") not in {"edgecraft_hash_sgd_v1", "edgecraft_hash_centroid_v1"}:
    print(json.dumps({"status": "error", "error": "Unsupported structured_log artifact payload"}))
    raise SystemExit(1)

N_FEATURES = int(payload["n_features"])
FORMAT = payload.get("format")
if FORMAT == "edgecraft_hash_sgd_v1":
    COEF = np.asarray(payload["coef"], dtype=np.float32)
    INTERCEPT = np.asarray(payload["intercept"], dtype=np.float32)
    CLASSES = np.asarray(payload["classes"], dtype=np.int64)
else:
    CENTROID = np.asarray(payload["centroid"], dtype=np.float32)
    THRESHOLD = float(payload.get("threshold", 0.0))
TOKEN_RE = re.compile(payload.get("token_pattern") or r"(?u)\\b\\w\\w+\\b")


def murmurhash3_32(data, seed=0):
    data = bytearray(data.encode("utf-8"))
    length = len(data)
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    h1 = seed & 0xFFFFFFFF
    rounded_end = length & 0xFFFFFFFC
    for i in range(0, rounded_end, 4):
        k1 = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF
    k1 = 0
    tail = length & 3
    if tail == 3:
        k1 ^= data[rounded_end + 2] << 16
    if tail >= 2:
        k1 ^= data[rounded_end + 1] << 8
    if tail >= 1:
        k1 ^= data[rounded_end]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    return h1 & 0xFFFFFFFF


def vectorize(text):
    if payload.get("lowercase", True):
        text = text.lower()
    x = np.zeros(N_FEATURES, dtype=np.float32)
    for token in TOKEN_RE.findall(text):
        x[murmurhash3_32(token) % N_FEATURES] += 1.0
    norm = float(np.linalg.norm(x))
    if norm > 0:
        x /= norm
    return x


def predict(texts):
    outputs = []
    for text in texts:
        x = vectorize(str(text))
        if FORMAT == "edgecraft_hash_centroid_v1":
            dist = float(np.linalg.norm(x - CENTROID))
            outputs.append(int(dist > THRESHOLD))
        elif COEF.shape[0] == 1:
            score = float(np.dot(COEF[0], x) + INTERCEPT[0])
            outputs.append(int(CLASSES[1] if score > 0 and len(CLASSES) > 1 else CLASSES[0]))
        else:
            scores = COEF.dot(x) + INTERCEPT
            outputs.append(int(CLASSES[int(np.argmax(scores))]))
    return outputs

sample = ["blk_-1608999687919862906 PacketResponder 1 for block terminating"]
for _ in range(5):
    _ = predict(sample)
latencies = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = predict(sample)
    latencies.append((time.perf_counter() - t0) * 1000.0)

avg = float(np.mean(latencies))
mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
print(json.dumps({
    "status": "success",
    "schema_version": "edge_eval_v1",
    "runtime_used": "hash-sgd-numpy",
    "runtime_provider": "CPUExecutionProvider",
    "artifact_used": "outputs/best.pt",
    "metrics": {
        "Latency": avg,
        "Latency_std": float(np.std(latencies)),
        "Memory_mb": float(mem_mb),
        "Throughput": float(1000.0 / avg) if avg > 0 else 0.0,
    },
}))
'''


structured_log_plugin = StructuredLogPlugin()
