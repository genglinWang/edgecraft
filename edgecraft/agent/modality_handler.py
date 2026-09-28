"""Modality-specific handler registry for EdgeCraft.

In the code-centric architecture, ModalityHandlers have a reduced role:
- materialize_config(): writes config/data.yaml for the dataset
- validate_variant(): validates SolutionVariant fields

Latency estimation, search space description, and trainer selection
are now handled by ModelRegistry.
"""
from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from edgecraft.core.modality import Modality, TaskType


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModalityHandler(ABC):
    """Contract that every modality handler must fulfil."""

    modality: Modality

    @abstractmethod
    def materialize_config(
        self,
        variant: Any,
        config_dir: Path,
        dataset_path: str,
    ) -> None:
        """Write dataset config files into config_dir.

        In code-centric architecture, this only writes data.yaml.
        Training hyperparameters are in train_code.
        """

    def validate_variant(self, variant: Any) -> List[str]:
        """Validate SolutionVariant fields. Returns list of error messages."""
        return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModalityHandlerRegistry:
    _handlers: Dict[Modality, ModalityHandler] = {}

    @classmethod
    def register(cls, handler: ModalityHandler) -> None:
        cls._handlers[handler.modality] = handler

    @classmethod
    def get(cls, modality: Modality) -> ModalityHandler:
        handler = cls._handlers.get(modality)
        if handler is None:
            raise NotImplementedError(
                f"No ModalityHandler registered for modality '{modality.value}'. "
                f"Registered: {[m.value for m in cls._handlers]}"
            )
        return handler

    @classmethod
    def is_supported(cls, modality: Modality) -> bool:
        return modality in cls._handlers

    @classmethod
    def supported_modalities(cls) -> List[Modality]:
        return list(cls._handlers.keys())


def _resolve_category_anomaly_layout(path: Path) -> tuple[str, str]:
    """Return (dataset_root, category_name) for normal-only anomaly layouts.

    The signature is structural: a category contains ``train/good`` and
    optional defect subfolders under test.  This deliberately avoids depending
    on any benchmark name.
    """
    path = path.resolve()
    if (path / "train" / "good").is_dir():
        return str(path.parent), path.name
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "train" / "good").is_dir():
            return str(path), child.name
    return str(path), "bottle"


def _resolve_point_counting_layout(path: Path) -> tuple[str, str]:
    """Return (root, split/part) for point-annotation crowd-counting layouts."""
    path = path.resolve()
    if (path / "part_A").is_dir() or (path / "part_B").is_dir():
        return str(path), "part_A" if (path / "part_A").is_dir() else "part_B"
    if (
        path.name.startswith("part_")
        and (path / "train_data" / "images").is_dir()
    ):
        return str(path.parent), path.name
    return str(path), "part_A"


def _find_split_dir(root: Path, names: tuple[str, ...]) -> Optional[Path]:
    """Find a split directory case-insensitively."""
    if not root.is_dir():
        return None
    lower_names = {n.lower() for n in names}
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in lower_names:
            return child
    return None


def _looks_like_imagefolder_split(path: Optional[Path]) -> bool:
    if not path or not path.is_dir():
        return False
    class_dirs = [p for p in path.iterdir() if p.is_dir()]
    if not class_dirs:
        return False
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}
    return any(
        any(f.suffix.lower() in image_suffixes for f in cls.rglob("*") if f.is_file())
        for cls in class_dirs[:20]
    )


def _imagefolder_root(path: Path) -> Path:
    """Resolve the dataset root for classification imagefolder layouts."""
    root = path.resolve()
    if root.is_file() or root.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl"}:
        root = root.parent
    if not root.exists() and root.suffix:
        root = root.parent
    return root


def _infer_table_schema(path: Path) -> Dict[str, Any]:
    """Infer generic text/structured columns from tabular files.

    This is deliberately heuristic and dataset-agnostic.  It gives templates a
    stable contract without hardcoding known benchmark names.
    """
    files: List[Path] = []
    table_suffixes = {".parquet", ".csv", ".jsonl", ".json", ".log", ".txt"}
    path_text = str(path)
    if any(ch in path_text for ch in "*?[]"):
        files = [
            Path(p)
            for p in sorted(glob.glob(path_text))
            if Path(p).suffix.lower() in table_suffixes
        ]
        if files:
            path = files[0].parent
    if path.is_file():
        if path.suffix.lower() in table_suffixes:
            files = [path]
        else:
            path = path.parent
    elif path.is_dir():
        pass
    if path.is_dir() and not files:
        for suffix in ("*.parquet", "*.csv", "*.jsonl", "*.json"):
            files.extend(sorted(path.rglob(suffix))[:8])
            if files:
                break
    if not files:
        return {}

    columns: List[str] = []
    first = files[0]
    try:
        if first.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as _pq

                columns = list(_pq.read_schema(first).names)
            except Exception:
                import pandas as _pd

                columns = list(_pd.read_parquet(first, engine="pyarrow").head(1).columns)
        elif first.suffix.lower() == ".csv":
            import csv as _csv

            with first.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = _csv.DictReader(f)
                columns = list(reader.fieldnames or [])
        elif first.suffix.lower() == ".jsonl":
            import json as _json

            with first.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        obj = _json.loads(line)
                        if isinstance(obj, dict):
                            columns = list(obj.keys())
                        break
        elif first.suffix.lower() == ".json":
            import json as _json

            obj = _json.loads(first.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                columns = list(obj[0].keys())
            elif isinstance(obj, dict):
                columns = list(obj.keys())
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"table schema inference failed for {first}: {exc}")
        columns = []

    def _pick(candidates: tuple[str, ...]) -> Optional[str]:
        lowered = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        for col in columns:
            col_l = col.lower()
            if any(len(c) > 2 and c in col_l for c in candidates):
                return col
        return None

    if first.suffix.lower() in {".log", ".txt"} and not columns:
        columns = ["Content"]

    return {
        "files": [str(f) for f in files],
        "columns": columns,
        "text_column": _pick(("text", "sentence", "content", "message", "log", "event", "template", "description")),
        "label_column": _pick((
            "label",
            "labels",
            "target",
            "class",
            "gt",
            "ground_truth",
            "activity",
            "anomaly",
            "is_anomaly",
            "defect",
            "defects",
            "bug",
            "buggy",
            "yield",
            "y",
        )),
        "label_join_key": _pick(("BlockId", "block_id", "blockid", "session_id", "trace_id", "window_id")),
        "split_column": _pick(("split", "fold", "partition")),
    }


def _common_data_root(dataset_path: Path, files: Optional[List[str]] = None) -> str:
    """Return the most useful absolute root for generated loaders.

    ``dataset_path`` remains the source of truth, but LLM-generated loaders often
    use conventional keys such as root/dataset_root/data_root.  For nested shard
    layouts, the common parent of observed files is the least surprising value.
    """
    candidates = [Path(f).resolve().parent for f in (files or []) if f]
    candidates = [p for p in candidates if p.exists()]
    if candidates:
        try:
            root = Path(os.path.commonpath([str(p) for p in candidates])).resolve()
        except Exception:
            root = candidates[0]
        dataset_root = dataset_path.resolve()
        if dataset_root.is_file():
            dataset_root = dataset_root.parent.resolve()
        if root == dataset_root:
            return str(root)
        # If schema sampling only saw one activity/session folder, expose its
        # parent as root so generated loaders can discover sibling classes.
        if "_" in root.name and root.parent.exists():
            sibling_dirs = [p for p in root.parent.iterdir() if p.is_dir()]
            if len(sibling_dirs) > 1 and any("_" in p.name for p in sibling_dirs):
                return str(root.parent.resolve())
        return str(root)
    ds = dataset_path.resolve()
    if ds.is_file():
        root = ds.parent
        if root.name.lower() in {"code", "codes", "notebook", "notebooks", "doc", "docs", "script", "scripts"}:
            root = root.parent
        data_dir = root / "data"
        if data_dir.exists() and data_dir.is_dir():
            return str(data_dir.resolve())
        return str(root.resolve())
    return str(ds)


def _add_root_aliases(payload: Dict[str, Any], root: str) -> None:
    """Expose common root keys without changing the canonical dataset_path."""
    payload.setdefault("root", root)
    payload.setdefault("dataset_root", root)
    payload.setdefault("data_root", root)


def _find_label_files(path: Path) -> List[str]:
    root = path if path.is_dir() else path.parent
    if not root.exists():
        return []
    candidates: List[Path] = []
    for pattern in (
        "*label*.csv",
        "*Label*.csv",
        "*anomaly*.csv",
        "*Anomaly*.csv",
        "*label*.txt",
        "*Label*.txt",
        "*anomaly*.txt",
        "*Anomaly*.txt",
    ):
        candidates.extend(sorted(root.rglob(pattern))[:8])
    seen: set[str] = set()
    out: List[str] = []
    for p in candidates:
        if not p.is_file():
            continue
        resolved = str(p.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _resolve_audio_data_root(path: Path) -> Path:
    """Normalize an audio dataset path without crossing dataset boundaries."""
    root = path.resolve()
    if root.is_file():
        return root.parent
    return root


def _audio_manifest_summary(path: Path) -> Dict[str, Any]:
    root = path.resolve()
    if root.is_file():
        root = root.parent
    manifests: List[str] = []
    for name in ("metadata.csv", "metadata.tsv", "metadata.jsonl", "validation_list.txt", "testing_list.txt"):
        manifests.extend(str(p.resolve()) for p in sorted(root.rglob(name))[:8] if p.is_file())
    audio_files = [
        p for suffix in ("*.wav", "*.flac", "*.mp3", "*.ogg")
        for p in sorted(root.rglob(suffix))[:32]
    ]
    return {
        "manifest_files": manifests,
        "audio_file_count_sampled": len(audio_files),
        "audio_root": str(root),
    }


# ---------------------------------------------------------------------------
# Vision handler
# ---------------------------------------------------------------------------

class VisionModalityHandler(ModalityHandler):
    """Handler for all vision modalities."""

    modality = Modality.VISION

    def materialize_config(
        self,
        variant: Any,
        config_dir: Path,
        dataset_path: str,
    ) -> None:
        """Write data.yaml for YOLO/timm training."""
        import yaml as _yaml

        config_dir.mkdir(parents=True, exist_ok=True)

        data_yaml_dst = config_dir / "data.yaml"
        src = Path(dataset_path)
        tt = getattr(variant, "task_type", None)

        # Dedicated layouts first (avoid picking a stray *.yaml in the dataset dir)
        if tt == TaskType.CLASSIFICATION:
            root = _imagefolder_root(Path(dataset_path))
            train_dir = _find_split_dir(root, ("train", "training"))
            val_dir = _find_split_dir(root, ("val", "valid", "validation", "test"))
            if _looks_like_imagefolder_split(train_dir):
                class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
                payload = {
                    "task": "image_classification",
                    "format": "imagefolder",
                    "root": str(root),
                    "train": str(train_dir),
                    "val": str(val_dir) if val_dir and _looks_like_imagefolder_split(val_dir) else str(train_dir),
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "notes": (
                        "Generic ImageFolder-style classification contract. "
                        "Use timm/torchvision public APIs; do not load this with YOLO(). "
                        "Do not assume Train.csv/test.csv when format=imagefolder."
                    ),
                }
                data_yaml_dst.write_text(
                    _yaml.dump(payload, default_flow_style=False, allow_unicode=True)
                )
                logger.debug(
                    f"materialize_config: image classification data.yaml → "
                    f"train={train_dir} classes={len(class_names)}"
                )
                return

        if tt == TaskType.ANOMALY_DETECTION:
            root = Path(dataset_path).resolve()
            anomaly_root, category = _resolve_category_anomaly_layout(root)
            payload = {
                "task": "category_structured_visual_anomaly",
                "deprecated_task_aliases": ["mvtec_anomaly"],
                "layout_contract": "normal_only_category_anomaly",
                "root": anomaly_root,
                "category": category,
                "imgsz": 128,
            }
            data_yaml_dst.write_text(
                _yaml.dump(payload, default_flow_style=False, allow_unicode=True)
            )
            logger.debug(
                f"materialize_config: category anomaly data.yaml → root={anomaly_root} category={category}"
            )
            return
        if tt == TaskType.CROWD_COUNTING:
            root = Path(dataset_path).resolve()
            count_root, part = _resolve_point_counting_layout(root)
            payload = {
                "task": "point_annotation_crowd_counting",
                "deprecated_task_aliases": ["shanghai_tech"],
                "layout_contract": "point_annotation_density_map",
                "root": count_root,
                "part": part,
                "imgsz": 448,
            }
            data_yaml_dst.write_text(
                _yaml.dump(payload, default_flow_style=False, allow_unicode=True)
            )
            logger.debug(
                f"materialize_config: point-counting data.yaml → root={count_root} part={part}"
            )
            return

        # If a directory is given, search for a YAML config inside it
        if src.is_dir():
            candidates = sorted(src.glob("*.yaml")) + sorted(src.glob("*.yml"))
            if candidates:
                preferred = [p for p in candidates if p.name.lower() == "data.yaml"]
                src = preferred[0] if preferred else candidates[0]
                logger.debug(f"materialize_config: resolved dataset dir → {src}")

        if src.is_file() and src.suffix in (".yaml", ".yml"):
            try:
                raw = _yaml.safe_load(src.read_text()) or {}
            except Exception:
                raw = {}

            # Patch 'path' to absolute so training resolves images correctly
            raw_path = raw.get("path", "")
            if raw_path:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    resolved = (src.parent / raw_path).resolve()
                    if not resolved.exists():
                        resolved = src.parent.resolve()
                    raw["path"] = str(resolved)
                    logger.debug(f"data.yaml 'path' patched: '{raw_path}' → '{raw['path']}'")
            else:
                raw["path"] = str(src.parent.resolve())

            raw.setdefault("data", "config/data.yaml")
            data_yaml_dst.write_text(
                _yaml.dump(raw, default_flow_style=False, allow_unicode=True)
            )
        else:
            # Write a skeleton data.yaml
            abs_dataset = str(Path(dataset_path).resolve())
            data_yaml_dst.write_text(
                f"path: {abs_dataset}\n"
                f"root: {abs_dataset}\n"
                f"dataset_root: {abs_dataset}\n"
                f"data_root: {abs_dataset}\n"
                "train: images/train\n"
                "val: images/val\n"
                "nc: 0\n"
                "names: []\n"
            )
            logger.warning(
                "dataset_path does not point to a YOLO yaml; wrote a skeleton "
                "data.yaml – you may need to fill in 'nc' and 'names' manually."
            )

    def validate_variant(self, variant: Any) -> List[str]:
        """Validate vision-specific fields."""
        errors = []
        if not variant.model_name:
            errors.append("model_name is required")
        if variant.quant_mode not in ("fp32", "fp16", "int8"):
            errors.append(f"Invalid quant_mode: {variant.quant_mode}")
        if variant.export_format not in ("onnx", "engine", "tflite", "pt"):
            errors.append(f"Invalid export_format: {variant.export_format}")
        return errors


# ---------------------------------------------------------------------------
# Generic handlers for non-vision modalities
# ---------------------------------------------------------------------------

class _GenericDatasetConfigHandler(ModalityHandler):
    """Minimal dataset-config materializer for non-vision modalities.

    This keeps the synthesis loop executable while modality-specific loaders
    evolve. The generated `config/data.yaml` provides a stable contract:
    - dataset_path: absolute path to dataset root/file
    - modality: modality id
    - task_type: inferred task type
    """

    modality: Modality

    def materialize_config(
        self,
        variant: Any,
        config_dir: Path,
        dataset_path: str,
    ) -> None:
        import yaml as _yaml

        config_dir.mkdir(parents=True, exist_ok=True)
        data_yaml_dst = config_dir / "data.yaml"
        ds_path = Path(dataset_path).resolve()
        schema = _infer_table_schema(ds_path)
        payload = {
            "dataset_path": str(ds_path),
            "modality": self.modality.value,
            "task_type": getattr(getattr(variant, "task_type", None), "value", "unknown"),
            "format": "tabular_or_text" if schema else "custom",
            "notes": (
                "Generated by generic modality handler. "
                "Model-family templates should consume dataset_path directly."
            ),
        }
        _add_root_aliases(payload, _common_data_root(ds_path, schema.get("files") if schema else None))
        canonical_dataset_root = ds_path.parent if ds_path.is_file() else ds_path
        payload["dataset_root"] = str(canonical_dataset_root.resolve())
        if schema:
            payload["schema"] = schema
            payload["files"] = schema.get("files", [])
            payload["text_column"] = schema.get("text_column")
            payload["label_column"] = schema.get("label_column")
            payload["split_column"] = schema.get("split_column")
            label_files = _find_label_files(ds_path)
            if label_files:
                payload["label_files"] = label_files
            label_join_key = schema.get("label_join_key")
            if not label_join_key:
                lowered_cols = {str(c).lower(): c for c in (schema.get("columns") or [])}
                label_join_key = (
                    lowered_cols.get("blockid")
                    or lowered_cols.get("block_id")
                )
            if label_join_key:
                payload["label_join_key"] = label_join_key
                payload["supervised_label_hint"] = (
                    "Block/session-level supervised log anomaly detection expects an "
                    "in-table anomaly column or external trace-level label file. "
                    "External files are often named anomaly_label.csv. "
                    "Join labels to table rows, blocks, sessions, or traces by the inferred "
                    f"join key {label_join_key!r} before supervised training."
                )
            payload["file_format"] = (
                Path(schema["files"][0]).suffix.lower().lstrip(".")
                if schema.get("files")
                else "unknown"
            )
            payload["label_required"] = bool(schema.get("label_column") or label_files)
            if not schema.get("label_column") and not label_files:
                payload["no_label_strategy"] = (
                    "No in-table label column or external label file was inferred. "
                    "For supervised anomaly_detection, fail with "
                    "dataset_contract_mismatch and explain that a block/session-level label "
                    "file is required; do not hallucinate labels or assume train.csv."
                )
            payload["notes"] = (
                "Generic text/structured contract with inferred schema. "
                "Templates must load schema.files directly and must not assume "
                "dataset/train.csv or dataset/val.csv exists."
            )
        data_yaml_dst.write_text(
            _yaml.dump(payload, default_flow_style=False, allow_unicode=True)
        )
        logger.debug(
            f"materialize_config: generic data.yaml for modality={self.modality.value} "
            f"dataset_path={payload['dataset_path']}"
        )

    def validate_variant(self, variant: Any) -> List[str]:
        errors = []
        if not getattr(variant, "model_name", ""):
            errors.append("model_name is required")
        if getattr(variant, "quant_mode", "") not in ("fp32", "fp16", "int8"):
            errors.append(f"Invalid quant_mode: {getattr(variant, 'quant_mode', '')}")
        if getattr(variant, "export_format", "") not in ("onnx", "engine", "tflite", "pt"):
            errors.append(f"Invalid export_format: {getattr(variant, 'export_format', '')}")
        return errors


class AudioModalityHandler(_GenericDatasetConfigHandler):
    modality = Modality.AUDIO

    def materialize_config(
        self,
        variant: Any,
        config_dir: Path,
        dataset_path: str,
    ) -> None:
        import yaml as _yaml

        config_dir.mkdir(parents=True, exist_ok=True)
        data_yaml_dst = config_dir / "data.yaml"
        ds_path = Path(dataset_path).resolve()

        # Audio contract: if dataset_path points to metadata (csv/tsv/json),
        # normalize to its parent directory for `load_dataset("audiofolder")`.
        audio_root = ds_path
        if ds_path.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".txt", ".lst"}:
            # Analyzer may point to a manifest path that is missing in a partial
            # download or nested mirror; treat manifest-like paths structurally
            # instead of requiring is_file().
            audio_root = ds_path.parent
            # If metadata is under a split folder (train/validation/test), promote
            # to dataset root so `audiofolder` can discover multiple splits.
            if audio_root.name.lower() in {"train", "training", "val", "validation", "test"}:
                parent = audio_root.parent
                if parent.exists() and parent.is_dir():
                    audio_root = parent
        audio_root = _resolve_audio_data_root(audio_root)

        payload = {
            "dataset_path": str(ds_path),
            "audio_root": str(audio_root),
            "audio_contract": _audio_manifest_summary(audio_root),
            "modality": self.modality.value,
            "task_type": getattr(getattr(variant, "task_type", None), "value", "unknown"),
            "target_model_scale": "tiny_edge",
            "max_recommended_params": 10_000_000,
            "notes": (
                "Audio contract: use audio_root for audiofolder loader; "
                "dataset_path preserves original source path. For tiny edge ASR, "
                "prefer a compact feature-based or small neural baseline that saves "
                "outputs/best.pt and attempts ONNX; avoid Whisper-sized models unless "
                "the user explicitly asks for them."
            ),
        }
        _add_root_aliases(payload, str(audio_root))
        data_yaml_dst.write_text(
            _yaml.dump(payload, default_flow_style=False, allow_unicode=True)
        )
        logger.debug(
            f"materialize_config: audio data.yaml dataset_path={payload['dataset_path']} "
            f"audio_root={payload['audio_root']}"
        )


class TextModalityHandler(_GenericDatasetConfigHandler):
    modality = Modality.TEXT


class TimeSeriesModalityHandler(_GenericDatasetConfigHandler):
    modality = Modality.TIME_SERIES


class StructuredModalityHandler(_GenericDatasetConfigHandler):
    modality = Modality.STRUCTURED


class MultimodalModalityHandler(_GenericDatasetConfigHandler):
    modality = Modality.MULTIMODAL


# ---------------------------------------------------------------------------
# Auto-register supported handlers at import time.
# Vision has a specialized handler. Other modalities currently use a generic
# handler that keeps the end-to-end loop runnable until dedicated handlers are
# introduced.
# ---------------------------------------------------------------------------

ModalityHandlerRegistry.register(VisionModalityHandler())
ModalityHandlerRegistry.register(AudioModalityHandler())
ModalityHandlerRegistry.register(TextModalityHandler())
ModalityHandlerRegistry.register(TimeSeriesModalityHandler())
ModalityHandlerRegistry.register(StructuredModalityHandler())
ModalityHandlerRegistry.register(MultimodalModalityHandler())
