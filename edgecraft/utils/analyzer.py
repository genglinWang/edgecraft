import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from loguru import logger

from edgecraft.agent.prompts.system import DATASET_EXPLORATION_SYSTEM_PROMPT, DATASET_JSON_FROM_REPORT_PROMPT
from edgecraft.utils.llm import create_chat_llm


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(getattr(block, "text", str(block)))
        return "".join(parts)
    return str(content or "")


def _normalize_tool_call(tc: Any) -> Dict[str, Any]:
    if isinstance(tc, dict):
        return {
            "id": str(tc.get("id") or ""),
            "name": str(tc.get("name") or ""),
            "args": tc.get("args") if isinstance(tc.get("args"), dict) else {},
        }
    return {
        "id": str(getattr(tc, "id", "") or ""),
        "name": str(getattr(tc, "name", "") or ""),
        "args": getattr(tc, "args", None) or {},
    }


def _analysis_failure_payload(
    *,
    error: str,
    dataset_path: str,
    sample_observation: Dict[str, Any],
    path_facts: Dict[str, Any],
    exploration_report: str = "",
) -> Dict[str, Any]:
    """Preserve deterministic dataset evidence when LLM exploration fails."""
    return {
        "status": "error",
        "error": error,
        "dataset_path": dataset_path,
        "sample_observation": sample_observation,
        "path_facts": path_facts,
        "exploration_report": exploration_report,
    }


_LABEL_KEYS = ("label", "labels", "target", "class", "category", "gt", "activity", "action", "intent", "anomaly", "is_anomaly", "sentiment", "y")
_TEXT_KEYS = ("text", "sentence", "message", "log", "content", "review", "transcript", "utterance", "title")
_SENSOR_KEYS = ("x", "y", "z", "accel", "accelerometer", "gyro", "gyroscope", "mag", "sensor")
_MEDIA_KEYS = ("audio", "image", "video", "file", "filepath", "path", "filename")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TABULAR_SUFFIXES = {".csv", ".tsv", ".jsonl", ".json", ".parquet"}
_CONFIG_SUFFIXES = {".yaml", ".yml"}
_MANIFEST_NAMES = {
    "train.txt",
    "train_list.txt",
    "training_list.txt",
    "val.txt",
    "valid.txt",
    "validation.txt",
    "validation_list.txt",
    "test.txt",
    "testing.txt",
    "testing_list.txt",
    "manifest.txt",
    "metadata.csv",
    "metadata.tsv",
}


def _timeseries_text_observation(path: Path, root: Path) -> Dict[str, Any]:
    """Expose a bounded raw-format fact for ARFF/UEA text time series."""
    directives: List[str] = []
    first_record = ""
    in_data = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                elif line.startswith("@") and len(directives) < 24:
                    directives.append(line[:240])
                continue
            first_record = line
            break

    class_values: List[str] = []
    for directive in directives:
        if directive.lower().startswith("@classlabel"):
            fields = directive.split()
            class_values = fields[2:] if len(fields) > 2 else []
            break
    return {
        "type": "timeseries_text_record",
        "source_file": str(path.relative_to(root)),
        "metadata_directives": directives,
        "record_prefix": first_record[:320],
        "record_suffix": first_record[-120:] if first_record else "",
        "separator_counts": {
            "colon": first_record.count(":"),
            "tab": first_record.count("\t"),
            "comma": first_record.count(","),
        },
        "class_values": class_values,
    }


def _safe_short(value: Any, limit: int = 180) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes len={len(value)}>"
    text = str(value)
    return text[:limit]


def _summarize_value(value: Any) -> Dict[str, Any]:
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": [int(x) for x in list(getattr(value, "shape", []))[:8]],
            "dtype": str(getattr(value, "dtype", "")),
        }
    if isinstance(value, dict):
        return {"type": "dict", "keys": list(value.keys())[:12], "items": {str(k): _safe_short(v) for k, v in list(value.items())[:4]}}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value), "items": [_safe_short(x) for x in list(value)[:4]]}
    return {"type": type(value).__name__, "repr": _safe_short(value)}


def _choose_label_key(keys: List[str]) -> str:
    low = {str(k).lower(): k for k in keys}
    for cand in _LABEL_KEYS:
        if cand in low:
            return low[cand]
    return ""


def _choose_input_keys(keys: List[str], label_key: str = "") -> List[str]:
    selected: List[str] = []
    low_pairs = [(str(k).lower(), k) for k in keys if k != label_key]
    for cand in _MEDIA_KEYS:
        for low, orig in low_pairs:
            if cand == low or cand in low:
                selected.append(orig)
                break
    for cand in _TEXT_KEYS:
        for low, orig in low_pairs:
            if orig in selected:
                continue
            if cand == low or cand in low:
                selected.append(orig)
                break
    if selected:
        return selected[:3]
    sensor_selected: List[str] = []
    for low, orig in low_pairs:
        if low in {"x", "y", "z"} or any(cand in low for cand in _SENSOR_KEYS[3:]):
            sensor_selected.append(orig)
    if sensor_selected:
        return sensor_selected[:6]
    return [k for k in keys if k != label_key][:3]


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:  # noqa: BLE001
        return False


def _column_value_samples(rows: List[Dict[str, Any]], *, max_columns: int = 12, max_values: int = 8) -> Dict[str, List[str]]:
    samples: Dict[str, List[str]] = {}
    keys = sorted({str(k) for row in rows for k in row.keys()})
    for key in keys:
        values: List[str] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            text = str(value)
            if not text or len(text) > 80:
                continue
            if text not in values:
                values.append(text)
            if len(values) >= max_values:
                break
        if values and any(not _looks_numeric(value) for value in values):
            samples[key] = values
        if len(samples) >= max_columns:
            break
    return samples


def _rows_to_sample_observation(
    rows: List[Dict[str, Any]],
    *,
    source_file: str = "",
    column_dtypes: Dict[str, str] | None = None,
    table_shape: List[int] | None = None,
) -> Dict[str, Any]:
    keys = sorted({str(k) for row in rows for k in row.keys()})
    label_key = _choose_label_key(keys)
    input_keys = _choose_input_keys(keys, label_key)
    labels = [row.get(label_key) for row in rows if label_key and label_key in row]
    label_counts: Dict[str, int] = {}
    for label in labels:
        key = str(label)
        label_counts[key] = label_counts.get(key, 0) + 1
    examples = []
    for row in rows[:5]:
        x = {k: row.get(k) for k in input_keys if k in row}
        y = row.get(label_key) if label_key else None
        examples.append({"input": _summarize_value(x), "label": _summarize_value(y) if y is not None else None})
    return {
        "status": "success" if rows else "empty",
        "source_file": source_file,
        "num_observed_samples": len(rows),
        "input_keys": input_keys,
        "label_key": label_key,
        "input_summary": examples[0]["input"] if examples else {},
        "column_dtypes": column_dtypes or {},
        "column_value_samples": _column_value_samples(rows),
        "table_shape": table_shape or [],
        "label_summary": {
            "num_labels_observed": len(labels),
            "unique_labels_observed": len(label_counts),
            "label_counts": dict(sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
        },
        "examples": examples,
    }


def _sample_csv_dataframe(path: Path, *, sep: str, max_rows: int):
    """Return compact but less head-biased rows from a CSV/TSV file."""
    import pandas as pd

    head = pd.read_csv(path, sep=sep, nrows=max_rows)
    keys = [str(c) for c in head.columns]
    label_key = _choose_label_key(keys)
    if not label_key or label_key not in head.columns:
        return head
    try:
        head_unique = int(head[label_key].dropna().astype(str).nunique())
    except Exception:  # noqa: BLE001
        return head
    if head_unique > 1:
        return head

    seen = set()
    rows: List[Dict[str, Any]] = []
    chunk_rows = max(10000, max_rows * 512)
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(path, sep=sep, chunksize=chunk_rows)):
            if label_key not in chunk.columns:
                break
            for _, row in chunk.drop_duplicates(subset=[label_key], keep="first").iterrows():
                value = str(row.get(label_key))
                if value and value not in seen:
                    seen.add(value)
                    rows.append(row.to_dict())
            if len(rows) >= max_rows or len(seen) >= 12 or chunk_idx >= 19:
                break
    except Exception:  # noqa: BLE001
        return head
    if len(seen) <= head_unique:
        return head
    sampled = pd.DataFrame(rows)
    return pd.concat([sampled, head], ignore_index=True).head(max_rows)


def build_sample_observation(dataset_root: Path, *, max_rows: int = 32) -> Dict[str, Any]:
    """Best-effort sample-level dataset understanding used by preflight.

    This is intentionally generic and read-only. It never edits dataset files and
    returns compact metadata only: a few input/label examples, label counts, split
    counts, and input shape/path summaries.
    """
    root = dataset_root.resolve()
    result: Dict[str, Any] = {
        "status": "empty",
        "root": str(root),
        "split_counts": {},
        "examples": [],
    }
    if not root.exists():
        return {"status": "error", "error": f"path not found: {root}", "root": str(root)}
    files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith("._")]
    for split in ("train", "training", "val", "valid", "validation", "dev", "test"):
        count = sum(1 for p in files if split in [part.lower() for part in p.relative_to(root).parts])
        if count:
            result["split_counts"][split] = count

    # HuggingFace Dataset/DatasetDict saved to disk.
    if (root / "dataset_dict.json").exists() or (root / "dataset_info.json").exists():
        try:
            from datasets import Audio, Image, load_from_disk
            ds = load_from_disk(str(root))
            split_name = "train" if hasattr(ds, "keys") and "train" in ds else None
            if split_name is None and hasattr(ds, "keys"):
                keys = list(ds.keys())
                split_name = keys[0] if keys else None
            subset = ds[split_name] if split_name else ds
            # Analyzer should observe samples without requiring heavyweight media codecs.
            # HuggingFace Audio/Image features are still real dataset evidence when decoded=False:
            # they expose path/bytes metadata without loading waveform/image tensors.
            try:
                for col, feature in (getattr(subset, "features", {}) or {}).items():
                    feature_name = type(feature).__name__.lower()
                    if feature_name == "audio":
                        subset = subset.cast_column(col, Audio(decode=False))
                    elif feature_name == "image":
                        subset = subset.cast_column(col, Image(decode=False))
            except Exception:
                pass
            rows = [dict(subset[i]) for i in range(min(max_rows, len(subset)))]
            feature_dtypes = {str(k): type(v).__name__ for k, v in (getattr(subset, "features", {}) or {}).items()}
            obs = _rows_to_sample_observation(
                rows,
                source_file=str(root),
                column_dtypes=feature_dtypes,
                table_shape=[int(len(subset)), int(len(getattr(subset, "column_names", []) or []))],
            )
            obs["split_counts"] = {str(k): int(len(v)) for k, v in ds.items()} if hasattr(ds, "items") else {"data": int(len(ds))}
            obs["format"] = "huggingface_dataset"
            return obs
        except Exception as exc:
            result["huggingface_error"] = f"{type(exc).__name__}: {exc}"

    # Tabular/text files.
    metadata_names = {"state.json", "dataset_info.json", "dataset_dict.json"}
    tabular_files = sorted([p for p in files if p.suffix.lower() in _TABULAR_SUFFIXES and p.name not in metadata_names], key=lambda p: str(p.relative_to(root)))
    for path in tabular_files[:8]:
        rows: List[Dict[str, Any]] = []
        try:
            suf = path.suffix.lower()
            if suf in {".csv", ".tsv"}:
                df = _sample_csv_dataframe(path, sep="\t" if suf == ".tsv" else ",", max_rows=max_rows)
                rows = df.to_dict(orient="records")
            elif suf == ".jsonl":
                with path.open(encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                rows.append(obj)
                        if len(rows) >= max_rows:
                            break
            elif suf == ".json":
                obj = json.loads(path.read_text(encoding="utf-8", errors="replace")[: 4 * 1024 * 1024])
                if isinstance(obj, list):
                    rows = [x for x in obj[:max_rows] if isinstance(x, dict)]
                elif isinstance(obj, dict):
                    for value in obj.values():
                        if isinstance(value, list) and value and isinstance(value[0], dict):
                            rows = value[:max_rows]
                            break
            elif suf == ".parquet":
                import pandas as pd
                df = pd.read_parquet(path)
                rows = df.head(max_rows).to_dict(orient="records")
            if rows:
                column_dtypes = {}
                table_shape = []
                if "df" in locals():
                    column_dtypes = {str(k): str(v) for k, v in df.dtypes.items()}
                    table_shape = [int(df.shape[0]), int(df.shape[1])]
                obs = _rows_to_sample_observation(
                    rows,
                    source_file=str(path.relative_to(root)),
                    column_dtypes=column_dtypes,
                    table_shape=table_shape,
                )
                related = []
                for related_path in tabular_files[:8]:
                    if related_path == path or related_path.suffix.lower() not in {".csv", ".tsv"}:
                        continue
                    try:
                        import pandas as pd

                        related_df = pd.read_csv(
                            related_path,
                            sep="\t" if related_path.suffix.lower() == ".tsv" else ",",
                            nrows=max_rows,
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    related.append(
                        {
                            "source_file": str(related_path.relative_to(root)),
                            "column_value_samples": _column_value_samples(
                                related_df.to_dict(orient="records")
                            ),
                        }
                    )
                if related:
                    obs["related_file_observations"] = related
                obs["split_counts"] = result.get("split_counts", {})
                obs["format"] = suf.lstrip(".")
                return obs
        except Exception as exc:
            result.setdefault("tabular_errors", []).append({"file": str(path.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"})

    # Standard text time-series files need raw delimiter evidence before image
    # files are considered.  Otherwise a documentation image can accidentally
    # become the dataset sample for a valid UEA/UCR dataset.
    timeseries_files = sorted(
        [p for p in files if p.suffix.lower() in {".ts", ".arff"}],
        key=lambda p: (
            p.suffix.lower() != ".ts",
            "train" not in p.name.lower(),
            str(p.relative_to(root)),
        ),
    )
    if timeseries_files:
        try:
            record = _timeseries_text_observation(timeseries_files[0], root)
            return {
                "status": "success",
                "source_file": record["source_file"],
                "num_observed_samples": int(bool(record["record_prefix"])),
                "input_summary": record,
                "label_summary": {
                    "num_labels_observed": 0,
                    "unique_labels_observed": len(record["class_values"]),
                    "label_counts": {},
                    "declared_classes": record["class_values"],
                },
                "split_counts": result.get("split_counts", {}),
                "examples": [{"input": record, "label": None}],
                "format": timeseries_files[0].suffix.lower().lstrip("."),
            }
        except Exception as exc:  # noqa: BLE001
            result.setdefault("timeseries_errors", []).append(
                {
                    "file": str(timeseries_files[0].relative_to(root)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    # Vision/image sample with sidecar YOLO labels when available.
    image_files = sorted([p for p in files if p.suffix.lower() in _IMAGE_SUFFIXES], key=lambda p: str(p.relative_to(root)))
    if image_files:
        examples = []
        label_counts: Dict[str, int] = {}
        for img in image_files[:8]:
            info: Dict[str, Any] = {"path": str(img.relative_to(root))}
            try:
                from PIL import Image
                with Image.open(img) as im:
                    info.update({"size": list(im.size), "mode": im.mode})
            except Exception as exc:
                info["image_error"] = f"{type(exc).__name__}: {exc}"
            label_path = None
            stem = img.stem + ".txt"
            candidates = [img.with_suffix(".txt"), root / "labels" / stem]
            rel_parts = list(img.relative_to(root).parts)
            if rel_parts and rel_parts[0].lower() == "images":
                candidates.append(root.joinpath("labels", *rel_parts[1:]).with_suffix(".txt"))
            for cand in candidates:
                if cand.exists():
                    label_path = cand
                    break
            label_summary = None
            if label_path:
                lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                classes = [ln.split()[0] for ln in lines if ln.split()]
                for cls in classes:
                    label_counts[cls] = label_counts.get(cls, 0) + 1
                label_summary = {"path": str(label_path.relative_to(root)), "num_items": len(lines), "classes": classes[:8]}
            examples.append({"input": {"type": "image", **info}, "label": label_summary})
        return {
            "status": "success",
            "format": "image_files",
            "num_observed_samples": len(examples),
            "input_summary": examples[0]["input"] if examples else {},
            "label_summary": {"label_counts": label_counts, "unique_labels_observed": len(label_counts)},
            "split_counts": result.get("split_counts", {}),
            "examples": examples[:5],
        }

    try:
        from edgecraft.data.observations import additional_observation

        extra = additional_observation(root, files, max_rows)
        if extra is not None:
            extra.setdefault("split_counts", result.get("split_counts", {}))
            return extra
    except Exception as exc:
        result["raw_format_error"] = f"{type(exc).__name__}: {exc}"

    result["status"] = "no_supported_sample_format"
    result["file_count"] = len(files)
    result["suffix_counts"] = {}
    for p in files[:500]:
        key = p.suffix.lower() or "<none>"
        result["suffix_counts"][key] = result["suffix_counts"].get(key, 0) + 1
    return result


def _path_exists_kind(path: Path) -> str:
    if path.is_file():
        return "file"
    if path.is_dir():
        return "dir"
    return "missing"


def _resolve_from(base: Path, raw: Any) -> Path:
    value = Path(str(raw))
    return value if value.is_absolute() else base / value


def _yaml_path_entries(config_path: Path, root: Path) -> Dict[str, Any]:
    try:
        import yaml as _yaml

        data = _yaml.safe_load(config_path.read_text(encoding="utf-8", errors="replace")) or {}
    except Exception as exc:
        return {
            "path": str(config_path.relative_to(root)),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(data, dict):
        return {"path": str(config_path.relative_to(root)), "kind": type(data).__name__}

    base = config_path.parent
    raw_path = data.get("path", "")
    path_base = _resolve_from(base, raw_path).resolve() if raw_path else base.resolve()
    item: Dict[str, Any] = {
        "path": str(config_path.relative_to(root)),
        "raw_path": raw_path,
        "resolved_path": str(path_base),
        "resolved_kind": _path_exists_kind(path_base),
        "splits": {},
    }
    for split in ("train", "val", "valid", "validation", "test"):
        raw_split = data.get(split)
        if raw_split is None:
            continue
        raw_items = raw_split if isinstance(raw_split, list) else [raw_split]
        split_entries = []
        for raw_entry in raw_items[:6]:
            resolved = _resolve_from(path_base, raw_entry).resolve()
            entry = {
                "raw": str(raw_entry),
                "resolved": str(resolved),
                "kind": _path_exists_kind(resolved),
            }
            if path_base != base.resolve():
                parent_resolved = _resolve_from(base, raw_entry).resolve()
                entry["parent_relative_resolved"] = str(parent_resolved)
                entry["parent_relative_kind"] = _path_exists_kind(parent_resolved)
            split_entries.append(entry)
        item["splits"][split] = split_entries
    return item


def _manifest_entries(path: Path, root: Path, *, max_lines: int = 5) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "kind": _path_exists_kind(path),
    }
    if not path.is_file():
        return item
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
        return item
    item["sample_lines"] = lines[:max_lines]
    path_checks = []
    for line in lines[:max_lines]:
        token = line.split()[0].split(",")[0]
        if not token or "://" in token:
            continue
        resolved = _resolve_from(path.parent, token).resolve()
        path_checks.append({
            "raw": token,
            "resolved": str(resolved),
            "kind": _path_exists_kind(resolved),
        })
    if path_checks:
        item["sample_path_checks"] = path_checks
    return item


def build_path_facts(dataset_root: Path, *, max_files: int = 4000) -> Dict[str, Any]:
    """Return compact read-only path/manifest facts for code generation.

    This is evidence, not policy. It helps the LLM see whether a config or
    manifest resolves to real files before it writes loader/train code.
    """
    root = dataset_root.resolve()
    if not root.exists():
        return {"status": "error", "root": str(root), "error": "path not found"}
    files: List[Path] = []
    dirs: List[Path] = []
    for idx, p in enumerate(root.rglob("*")):
        if idx >= max_files:
            break
        if p.name.startswith("._"):
            continue
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            dirs.append(p)

    configs = [
        _yaml_path_entries(p, root)
        for p in sorted(files, key=lambda x: (x.name.lower() != "data.yaml", str(x.relative_to(root))))
        if p.suffix.lower() in _CONFIG_SUFFIXES
    ][:8]
    manifest_paths = [
        p
        for p in sorted(files + dirs, key=lambda x: str(x.relative_to(root)))
        if p.name.lower() in _MANIFEST_NAMES or "manifest" in p.name.lower()
    ][:12]
    manifests = [_manifest_entries(p, root) for p in manifest_paths]
    directory_index = [
        str(path.relative_to(root))
        for path in sorted(
            dirs,
            key=lambda path: (
                len(path.relative_to(root).parts),
                str(path.relative_to(root)).lower(),
            ),
        )[:120]
    ]
    return {
        "status": "success",
        "root": str(root),
        "path_roles": _path_role_facts(root, files, configs, manifests),
        "directory_index": directory_index,
        "config_files": configs,
        "manifest_files": manifests,
        "scanned_files": len(files),
        "truncated": len(files) >= max_files,
    }


def _path_role_facts(
    root: Path,
    files: List[Path],
    configs: List[Dict[str, Any]],
    manifests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Expose path roles without deciding how the loader must use them.

    The old single ``dataset_path`` field mixed root directories, manifest
    files, split files, and library load roots.  This read-only view separates
    those meanings so the LLM can make a better dataset-specific loading plan.
    """
    anchor_files: List[Dict[str, Any]] = []
    candidate_load_roots: List[Dict[str, Any]] = [
        {"path": str(root), "kind": "dir", "reason": "dataset_root"}
    ]

    def add_anchor(path: Path, role: str) -> None:
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        item = {
            "path": str(path),
            "relative_path": rel,
            "role": role,
            "kind": _path_exists_kind(path),
        }
        if item not in anchor_files:
            anchor_files.append(item)

    def add_load_root(path: Path, reason: str) -> None:
        path = path.resolve()
        item = {"path": str(path), "kind": _path_exists_kind(path), "reason": reason}
        if item not in candidate_load_roots:
            candidate_load_roots.append(item)

    by_name = {p.name.lower(): p for p in files}
    for name in ("dataset_dict.json", "dataset_info.json", "state.json"):
        marker = by_name.get(name)
        if marker:
            add_anchor(marker, "library_dataset_marker")
            add_load_root(marker.parent, f"load_from_disk_root_for_{name}")

    for cfg in configs[:6]:
        path = root / str(cfg.get("path", ""))
        if path.exists():
            add_anchor(path, "config_file")
        resolved = cfg.get("resolved_path")
        if resolved:
            add_load_root(Path(str(resolved)), "config_resolved_path")

    for manifest in manifests[:8]:
        path = root / str(manifest.get("path", ""))
        if path.exists():
            add_anchor(path, "manifest_or_split_file")
            add_load_root(path.parent, "manifest_parent")

    split_like_suffixes = {".ts", ".arff", ".csv", ".tsv", ".jsonl", ".parquet"}
    split_tokens = ("train", "test", "val", "valid", "dev")
    for path in sorted(files, key=lambda x: str(x.relative_to(root)))[: max(200, min(len(files), 800))]:
        name = path.name.lower()
        if path.suffix.lower() in split_like_suffixes and any(tok in name for tok in split_tokens):
            add_anchor(path, "split_data_file")
            add_load_root(path.parent, "split_file_parent")
            if len(anchor_files) >= 18:
                break

    return {
        "dataset_root": str(root),
        "anchor_files": anchor_files[:18],
        "candidate_load_roots": candidate_load_roots[:12],
    }


def build_dataset_exploration_tools(
    root: Path,
    *,
    list_dir_cap: int = 80,
    read_line_cap: int = 160,
    read_char_cap: int = 8_000,
    max_file_bytes: int = 4 * 1024 * 1024,
) -> List[BaseTool]:
    """Create sandboxed list/read tools rooted at ``root`` (absolute, resolved)."""
    from langchain_core.tools import tool

    r = root.resolve()

    def _safe_resolve(relative_path: str) -> Path:
        rel = (relative_path or ".").strip() or "."
        rel = rel.replace("\\", "/").lstrip("/")
        candidate = (r / rel).resolve()
        try:
            candidate.relative_to(r)
        except ValueError:
            raise ValueError(f"path escapes dataset root: {relative_path!r}") from None
        return candidate

    @tool
    def dataset_list_dir(relative_path: str = ".", max_entries: int = 80) -> str:
        """List files and subdirectories at a path relative to the dataset root. Use '.' for the root."""
        try:
            cap = max(1, min(int(max_entries), list_dir_cap))
            target = _safe_resolve(relative_path)
        except Exception as exc:
            return f"Error: {exc}"
        if not target.is_dir():
            return f"Error: not a directory: {relative_path!r}"
        try:
            items = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            return f"Error: cannot list directory: {exc}"
        total = len(items)
        items = items[:cap]
        rel_display = "." if target == r else str(target.relative_to(r))
        lines = [f"Listing: {rel_display}  (showing {len(items)} of {total} entries)"]
        for p in items:
            suffix = "/" if p.is_dir() else ""
            try:
                sz = p.stat().st_size if p.is_file() else 0
            except OSError:
                sz = -1
            extra = f"  size={sz}" if p.is_file() and sz >= 0 else ""
            lines.append(f"- {p.name}{suffix}{extra}")
        if total > cap:
            lines.append(f"... truncated ({total - cap} more entries not shown)")
        return "\n".join(lines)

    allowed_suffixes = {
        ".txt",
        ".md",
        ".markdown",
        ".yaml",
        ".yml",
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".xml",
        ".cfg",
        ".ini",
    }
    basename_ok = {"readme", "copying", "license", "makefile", "dockerfile"}

    @tool
    def dataset_read_file(relative_path: str, start_line: int = 1, max_lines: int = 120) -> str:
        """Read bounded text under the dataset root; both line and character caps apply."""
        try:
            ml = max(1, min(int(max_lines), read_line_cap))
            sl = max(1, int(start_line))
            target = _safe_resolve(relative_path)
        except Exception as exc:
            return f"Error: {exc}"
        if not target.is_file():
            return f"Error: not a file: {relative_path!r}"
        name_low = target.name.lower()
        suf = target.suffix.lower()
        if name_low not in basename_ok and suf not in allowed_suffixes:
            return (
                "Error: file type not allowed. "
                f"Allowed suffixes: {sorted(allowed_suffixes)}; "
                f"allowed basenames: {sorted(basename_ok)}."
            )
        try:
            sz = target.stat().st_size
        except OSError as exc:
            return f"Error: stat failed: {exc}"
        if sz > max_file_bytes:
            return f"Error: file too large ({sz} bytes > limit {max_file_bytes})"
        try:
            probe = target.read_bytes()[:8192]
            if b"\x00" in probe:
                return "Error: file appears binary (null byte in first 8KiB)"
        except OSError as exc:
            return f"Error: read failed: {exc}"

        lines_out: List[str] = []
        n = 0
        line_no = 0
        emitted_chars = 0
        char_truncated = False
        try:
            with target.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_no += 1
                    if line_no < sl:
                        continue
                    clean_line = line.rstrip("\r\n")
                    rendered = f"L{line_no}: {clean_line}"
                    remaining = max(0, int(read_char_cap) - emitted_chars)
                    if len(rendered) > remaining:
                        if remaining:
                            lines_out.append(rendered[:remaining])
                            emitted_chars += remaining
                            n += 1
                        char_truncated = True
                        break
                    lines_out.append(rendered)
                    emitted_chars += len(rendered) + 1
                    n += 1
                    if n >= ml:
                        break
        except OSError as exc:
            return f"Error: cannot read file: {exc}"

        rel = str(target.relative_to(r))
        header = f"File: {rel}  (lines {sl}-{sl + n - 1}, showing up to {ml} lines)"
        if n == 0:
            return header + "\n(no lines in range; file may be shorter than start_line)"
        if char_truncated:
            more = (
                f"\n... (output truncated at {read_char_cap} characters; "
                "inspect schema/keys instead of requesting a giant line)"
            )
        else:
            more = "\n... (file continues; increase start_line to read further)" if n >= ml else ""
        return header + "\n" + "\n".join(lines_out) + more

    @tool
    def dataset_file_summary(relative_path: str = ".", max_files: int = 40) -> str:
        """Summarize dataset file formats, table headers, and loader hints under a path."""
        try:
            cap = max(1, min(int(max_files), 200))
            target = _safe_resolve(relative_path)
        except Exception as exc:
            return f"Error: {exc}"
        files: List[Path]
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = []
            for p in target.rglob("*"):
                if p.is_file() and not p.name.startswith("._"):
                    files.append(p)
                    if len(files) >= cap:
                        break
        else:
            return f"Error: not found: {relative_path!r}"

        suffix_counts: Dict[str, int] = {}
        for p in files:
            suffix_counts[p.suffix.lower() or "<none>"] = suffix_counts.get(p.suffix.lower() or "<none>", 0) + 1

        lines = [f"File summary for {relative_path or '.'}: sampled {len(files)} file(s)"]
        lines.append("Suffix counts: " + json.dumps(dict(sorted(suffix_counts.items())), ensure_ascii=False))
        for p in files[: min(len(files), 20)]:
            rel = str(p.relative_to(r))
            suf = p.suffix.lower()
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            lines.append(f"- {rel}  suffix={suf or '<none>'} size={size}")
            try:
                if suf in {".csv", ".tsv"}:
                    with p.open(newline="", encoding="utf-8", errors="replace") as f:
                        dialect = "\t" if suf == ".tsv" else ","
                        header = next(csv.reader(f, delimiter=dialect), [])
                    lines.append(f"  table_header={header[:40]}")
                elif suf == ".jsonl":
                    with p.open(encoding="utf-8", errors="replace") as f:
                        for line in f:
                            if line.strip():
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    lines.append(f"  jsonl_keys={list(obj.keys())[:40]}")
                                break
                elif suf == ".json":
                    text = p.read_text(encoding="utf-8", errors="replace")[:20000]
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        lines.append(f"  json_keys={list(obj.keys())[:40]}")
                    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        lines.append(f"  json_list_keys={list(obj[0].keys())[:40]}")
                elif suf == ".parquet":
                    try:
                        import pyarrow.parquet as pq

                        lines.append(f"  parquet_columns={list(pq.read_schema(p).names)[:40]}")
                    except Exception as exc:
                        lines.append(f"  parquet_schema_error={exc}")
                elif suf in {".arff", ".ts"}:
                    head = []
                    with p.open(encoding="utf-8", errors="replace") as f:
                        for _, line in zip(range(12), f):
                            head.append(line.rstrip())
                    lines.append("  time_series_header=" + " | ".join(head[:12])[:800])
                elif suf == ".arrow":
                    lines.append("  arrow_hint=HuggingFace/Arrow dataset shard; inspect dataset_info.json/state.json and load with datasets.load_from_disk when possible.")
            except Exception as exc:
                lines.append(f"  summary_error={type(exc).__name__}: {exc}")
        return "\n".join(lines)

    def _iter_sample_files(target: Path, cap: int, suffixes: set[str] | None = None) -> List[Path]:
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = [p for p in target.rglob("*") if p.is_file() and not p.name.startswith("._")]
        else:
            return []
        if suffixes:
            files = [p for p in files if p.suffix.lower() in suffixes]
        return sorted(files, key=lambda p: str(p.relative_to(r)))[:cap]

    @tool
    def dataset_profile_schema(relative_path: str = ".", max_files: int = 12, max_rows: int = 20) -> str:
        """Profile tabular/text/time-series files and suggest text/target/split columns."""
        try:
            target = _safe_resolve(relative_path)
            files = _iter_sample_files(
                target,
                max(1, min(int(max_files), 50)),
                {".csv", ".tsv", ".jsonl", ".json", ".parquet", ".arff", ".ts"},
            )
            row_cap = max(1, min(int(max_rows), 100))
        except Exception as exc:
            return f"Error: {exc}"
        lines = [f"Schema profile for {relative_path or '.'}: {len(files)} file(s)"]
        for p in files:
            rel = str(p.relative_to(r))
            suf = p.suffix.lower()
            lines.append(f"- file={rel} suffix={suf}")
            try:
                if suf in {".csv", ".tsv"}:
                    import pandas as pd

                    df = pd.read_csv(p, sep="\t" if suf == ".tsv" else ",", nrows=row_cap)
                    cols = list(df.columns)
                    lines.append(f"  columns={cols[:60]}")
                    lines.append(f"  dtypes={{{', '.join(f'{c}: {str(df[c].dtype)}' for c in cols[:20])}}}")
                    nunique = {c: int(df[c].nunique(dropna=True)) for c in cols[:20]}
                    lines.append(f"  sampled_unique_counts={nunique}")
                elif suf == ".jsonl":
                    rows = []
                    with p.open(encoding="utf-8", errors="replace") as f:
                        for line in f:
                            if line.strip():
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    rows.append(obj)
                                if len(rows) >= row_cap:
                                    break
                    keys = sorted({k for row in rows for k in row.keys()})
                    lines.append(f"  jsonl_keys={keys[:60]}")
                elif suf == ".json":
                    obj = json.loads(p.read_text(encoding="utf-8", errors="replace")[:max_file_bytes])
                    sample = obj[0] if isinstance(obj, list) and obj else obj
                    if isinstance(sample, dict):
                        lines.append(f"  json_keys={list(sample.keys())[:60]}")
                elif suf == ".parquet":
                    import pyarrow.parquet as pq

                    schema = pq.read_schema(p)
                    lines.append(f"  parquet_columns={list(schema.names)[:60]}")
                elif suf in {".arff", ".ts"}:
                    head = []
                    with p.open(encoding="utf-8", errors="replace") as f:
                        for _, line in zip(range(40), f):
                            head.append(line.rstrip())
                    lines.append("  header_excerpt=" + " | ".join(head)[:1600])
            except Exception as exc:
                lines.append(f"  profile_error={type(exc).__name__}: {exc}")
        joined = "\n".join(lines)
        lower = joined.lower()
        candidates = []
        for token in ("label", "target", "class", "category", "anomaly", "is_anomaly", "sentiment", "split", "fold", "text", "sentence", "message", "log"):
            if token in lower:
                candidates.append(token)
        if candidates:
            lines.append("candidate_semantic_columns=" + ", ".join(sorted(set(candidates))))
        return "\n".join(lines)

    @tool
    def dataset_detect_splits(relative_path: str = ".", max_entries: int = 200) -> str:
        """Detect train/dev/val/test/fold split files or directories."""
        try:
            target = _safe_resolve(relative_path)
            cap = max(1, min(int(max_entries), 1000))
        except Exception as exc:
            return f"Error: {exc}"
        names = {"train", "training", "dev", "valid", "validation", "val", "test", "fold", "folds"}
        matches = []
        for p in _iter_sample_files(target, cap) + [d for d in target.rglob("*") if d.is_dir()][:cap]:
            rel = str(p.relative_to(r))
            parts = [part.lower() for part in p.parts]
            if any(n in parts or n in p.name.lower() for n in names):
                matches.append(rel + ("/" if p.is_dir() else ""))
        return "Split candidates:\n" + ("\n".join(f"- {m}" for m in matches[:cap]) if matches else "(none found)")

    @tool
    def dataset_detect_annotation_format(relative_path: str = ".", max_files: int = 80) -> str:
        """Detect common annotation contracts such as YOLO bbox/pose/seg, mask folders, COCO JSON, and point annotations."""
        try:
            target = _safe_resolve(relative_path)
            files = _iter_sample_files(target, max(1, min(int(max_files), 300)))
        except Exception as exc:
            return f"Error: {exc}"
        suffix_counts: Dict[str, int] = {}
        hints: List[str] = []
        for p in files:
            suffix_counts[p.suffix.lower() or "<none>"] = suffix_counts.get(p.suffix.lower() or "<none>", 0) + 1
            rel_low = str(p.relative_to(r)).lower()
            if p.suffix.lower() == ".txt" and "label" in rel_low:
                try:
                    line = p.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
                    nums = line.split()
                    if len(nums) == 5:
                        hints.append("yolo_bbox")
                    elif len(nums) > 5 and (len(nums) - 5) % 3 == 0:
                        hints.append("yolo_pose")
                    elif len(nums) > 5 and (len(nums) - 1) % 2 == 0:
                        hints.append("yolo_seg_polygon")
                except Exception:
                    pass
            if p.suffix.lower() == ".json":
                try:
                    obj = json.loads(p.read_text(encoding="utf-8", errors="replace")[:max_file_bytes])
                    if isinstance(obj, dict) and {"images", "annotations"}.issubset(obj.keys()):
                        hints.append("coco_json")
                except Exception:
                    pass
            if p.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"} and any(x in rel_low for x in ("mask", "masks", "seg")):
                hints.append("mask_folder")
            if p.suffix.lower() in {".mat", ".npy", ".csv", ".json"} and any(x in rel_low for x in ("gt", "ground", "point", "density", "annot")):
                hints.append("point_or_density_annotation")
        lines = ["Annotation format detection:"]
        lines.append("suffix_counts=" + json.dumps(dict(sorted(suffix_counts.items())), ensure_ascii=False))
        lines.append("format_hints=" + (", ".join(sorted(set(hints))) if hints else "none"))
        return "\n".join(lines)

    @tool
    def dataset_validate_labels(relative_path: str = ".", max_files: int = 120) -> str:
        """Check label files for empty labels, class counts, and missing-looking image/label pairs."""
        try:
            target = _safe_resolve(relative_path)
            files = _iter_sample_files(target, max(1, min(int(max_files), 500)))
        except Exception as exc:
            return f"Error: {exc}"
        label_files = [p for p in files if "label" in str(p.relative_to(r)).lower() or p.suffix.lower() in {".csv", ".jsonl"}]
        empty = 0
        class_counts: Dict[str, int] = {}
        for p in label_files[: max(1, min(int(max_files), 500))]:
            try:
                if p.suffix.lower() == ".txt":
                    text = p.read_text(encoding="utf-8", errors="replace").strip()
                    if not text:
                        empty += 1
                    for line in text.splitlines():
                        parts = line.split()
                        if parts:
                            class_counts[parts[0]] = class_counts.get(parts[0], 0) + 1
            except Exception:
                continue
        image_files = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        return "\n".join([
            f"Label validation sampled_label_files={len(label_files)} sampled_image_files={len(image_files)}",
            f"empty_label_files={empty}",
            "class_counts_sample=" + json.dumps(dict(sorted(class_counts.items())[:40]), ensure_ascii=False),
        ])

    @tool
    def dataset_sample_media(relative_path: str = ".", max_files: int = 24) -> str:
        """Return media sample metadata such as image size/channel or audio file headers without returning media content."""
        try:
            target = _safe_resolve(relative_path)
            files = _iter_sample_files(target, max(1, min(int(max_files), 100)), {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".wav", ".flac", ".mp3", ".ogg"})
        except Exception as exc:
            return f"Error: {exc}"
        lines = [f"Media sample metadata: {len(files)} file(s)"]
        for p in files:
            rel = str(p.relative_to(r))
            try:
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    from PIL import Image

                    with Image.open(p) as img:
                        lines.append(f"- {rel}: image size={img.size} mode={img.mode}")
                elif p.suffix.lower() == ".wav":
                    import wave

                    with wave.open(str(p), "rb") as w:
                        lines.append(f"- {rel}: wav channels={w.getnchannels()} rate={w.getframerate()} frames={w.getnframes()}")
                else:
                    lines.append(f"- {rel}: media suffix={p.suffix.lower()} size={p.stat().st_size}")
            except Exception as exc:
                lines.append(f"- {rel}: media_error={type(exc).__name__}: {exc}")
        return "\n".join(lines)

    return [
        dataset_list_dir,
        dataset_read_file,
        dataset_file_summary,
        dataset_profile_schema,
        dataset_detect_splits,
        dataset_detect_annotation_format,
        dataset_validate_labels,
        dataset_sample_media,
    ]


class DatasetAnalyzer:
    """Dataset analyzer: LLM explores the tree via sandboxed tools, then emits prose + structured JSON."""

    def __init__(
        self,
        *,
        max_tool_rounds: int = 30,
        max_tool_calls: int = 16,
        list_dir_max_entries: int = 100,
        read_file_max_lines: int = 160,
        read_file_max_chars: int = 8_000,
        read_file_max_bytes: int = 2 * 1024 * 1024,
    ):
        self.llm = create_chat_llm(temperature=0.1, purpose="dataset_analyzer")
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.list_dir_max_entries = int(list_dir_max_entries)
        self.read_file_max_lines = int(read_file_max_lines)
        self.read_file_max_chars = int(read_file_max_chars)
        self.read_file_max_bytes = int(read_file_max_bytes)

    def run(
        self,
        dataset_path: str,
        *,
        requested_modality: str = "",
        requested_task_type: str = "",
    ) -> Dict[str, Any]:
        path = Path(dataset_path)
        if not path.exists():
            logger.warning("DatasetAnalyzer: path does not exist: {}", dataset_path)
            return {"status": "error", "error": f"Path {dataset_path} not found"}

        if path.is_file():
            root = path.resolve().parent
            entry_hint = (
                f"The user pointed to a **file** `{path.name}`; the dataset root for tools is its parent directory.\n"
                f"Absolute root: `{root}`\n"
            )
            user_path_abs = str(path.resolve())
        elif path.is_dir():
            root = path.resolve()
            entry_hint = f"Dataset root (absolute): `{root}`\n"
            user_path_abs = str(root)
        else:
            return {"status": "error", "error": f"Path {dataset_path} is neither file nor directory"}

        logger.info("DatasetAnalyzer: tool exploration root={}", root)

        tools = build_dataset_exploration_tools(
            root,
            list_dir_cap=self.list_dir_max_entries,
            read_line_cap=self.read_file_max_lines,
            read_char_cap=self.read_file_max_chars,
            max_file_bytes=self.read_file_max_bytes,
        )
        tool_map: Dict[str, BaseTool] = {t.name: t for t in tools}
        deterministic_evidence = []
        sample_observation = build_sample_observation(root)
        path_facts = build_path_facts(root)
        deterministic_evidence.append(
            "[dataset_sample_observation]\n"
            + json.dumps(sample_observation, ensure_ascii=False, indent=2)[:6000]
        )
        deterministic_evidence.append(
            "[dataset_path_facts]\n"
            + json.dumps(path_facts, ensure_ascii=False, indent=2)[:6000]
        )
        for tool_name, args in (
            ("dataset_file_summary", {"relative_path": ".", "max_files": 80}),
            ("dataset_profile_schema", {"relative_path": ".", "max_files": 24, "max_rows": 5}),
            ("dataset_detect_splits", {"relative_path": "."}),
        ):
            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                continue
            try:
                deterministic_evidence.append(f"[{tool_name}]\n{tool_fn.invoke(args)}")
            except Exception as exc:  # noqa: BLE001
                deterministic_evidence.append(f"[{tool_name}] error: {exc}")
        deterministic_evidence_text = "\n\n".join(deterministic_evidence)
        # OpenAI-style: force the first model turn to call dataset_list_dir (not plain text).
        _force_list_dir_choice: Dict[str, Any] = {
            "type": "function",
            "function": {"name": "dataset_list_dir"},
        }

        request_hint = ""
        if requested_modality or requested_task_type:
            request_hint = (
                "Requested synthesis context (use this to prioritize relevant real "
                "annotations and manifests; still report contradictions honestly):\n"
                f"- modality: {requested_modality or 'unspecified'}\n"
                f"- task: {requested_task_type or 'unspecified'}\n\n"
            )

        human_task = (
            entry_hint
            + "\n"
            + request_hint
            + "Explore this dataset using only the tools. When finished, **stop calling tools** and write a single "
            "detailed report (plain text or markdown, **not JSON**). The report should cover, as applicable:\n"
            "- Root location and any canonical entry manifests (yaml, jsonl, fold lists, etc.)\n"
            "- Overall directory/file structure and naming patterns\n"
            "- Train/validation/test splits, k-fold files, or how splits are defined\n"
            "- How raw media or text relates to labels or metadata (paths, columns, sidecars)\n"
            "- Modality, likely ML task, dataset format family, and practical notes for a dataloader or training job\n"
            "Use dataset_file_summary whenever modality is ambiguous. Use dataset_profile_schema for "
            "CSV/TSV/JSONL/parquet/ARFF/TS files, dataset_detect_splits for train/dev/test layouts, "
            "dataset_detect_annotation_format for vision labels, dataset_validate_labels for label "
            "consistency, and dataset_sample_media for image/audio shape metadata. Do not infer a "
            "loader until the relevant evidence is observed.\n"
            "\nInitial deterministic evidence from generic dataset tools (treat this as observed evidence; "
            "do not ignore nested files just because a root-level archive or metadata file is empty):\n"
            f"{deterministic_evidence_text}\n"
        )

        messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]] = [
            SystemMessage(content=DATASET_EXPLORATION_SYSTEM_PROMPT),
            HumanMessage(content=human_task),
        ]

        report = ""
        tool_calls_used = 0
        tool_cache: Dict[str, str] = {}
        try:
            for turn in range(self.max_tool_rounds):
                if turn == 0:
                    bound = self.llm.bind_tools(tools, tool_choice=_force_list_dir_choice)
                    logger.info("DatasetAnalyzer: round 0 — tool_choice forces dataset_list_dir")
                else:
                    bound = self.llm.bind_tools(tools)
                ai = bound.invoke(messages)
                messages.append(ai)
                tcalls = getattr(ai, "tool_calls", None) or []
                if not tcalls:
                    report = _stringify_message_content(ai.content).strip()
                    logger.info("DatasetAnalyzer: exploration finished after {} tool round(s)", turn)
                    break
                budget_exhausted = False
                for idx, tc in enumerate(tcalls):
                    norm = _normalize_tool_call(tc)
                    name = norm["name"]
                    args = norm["args"]
                    tid = norm["id"] or f"call_{turn}_{idx}"
                    logger.info("DatasetAnalyzer: tool_call name={} args={}", name, args)
                    signature = json.dumps(
                        {"name": name, "args": args},
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    if signature in tool_cache:
                        body = tool_cache[signature]
                    elif tool_calls_used >= self.max_tool_calls:
                        body = "Tool-call budget exhausted; use the evidence already observed."
                        budget_exhausted = True
                    else:
                        tool_calls_used += 1
                        tool_fn = tool_map.get(name)
                        if tool_fn is None:
                            body = f"Error: unknown tool {name!r}"
                        else:
                            try:
                                body = tool_fn.invoke(args)
                            except Exception as exc:
                                body = f"Error invoking tool: {exc}"
                        tool_cache[signature] = body
                    messages.append(ToolMessage(content=body, tool_call_id=tid))
                if budget_exhausted:
                    messages.append(
                        HumanMessage(
                            content=(
                                "The dataset-tool call budget is exhausted. Do not call more tools. "
                                "Write the final dataset report now using only the evidence already observed."
                            )
                        )
                    )
                    ai = self.llm.invoke(messages)
                    report = _stringify_message_content(ai.content).strip()
                    logger.info(
                        "DatasetAnalyzer: tool-call budget exhausted after {} unique call(s); "
                        "synthesized final report",
                        tool_calls_used,
                    )
                    break
            else:
                messages.append(
                    HumanMessage(
                        content=(
                            "The dataset-tool budget is exhausted. Do not call more tools. "
                            "Write the final dataset report now using only the evidence already observed."
                        )
                    )
                )
                ai = self.llm.invoke(messages)
                report = _stringify_message_content(ai.content).strip()
                logger.info(
                    "DatasetAnalyzer: tool budget exhausted after {} round(s); synthesized final report",
                    self.max_tool_rounds,
                )

            if not report:
                return _analysis_failure_payload(
                    error="Model returned an empty final report",
                    dataset_path=user_path_abs,
                    sample_observation=sample_observation,
                    path_facts=path_facts,
                )

            structured = self._structured_from_report(report)
            structured["status"] = "success"
            structured["dataset_path"] = user_path_abs
            structured["sample_observation"] = sample_observation
            structured["path_facts"] = path_facts
            structured["exploration_report"] = report
            logger.info(
                "DatasetAnalyzer: success — modality={} task_type={} format={}",
                structured.get("modality"),
                structured.get("task_type"),
                structured.get("format"),
            )
            return structured
        except Exception as exc:
            logger.exception("DatasetAnalyzer: exploration or JSON extraction failed")
            return _analysis_failure_payload(
                error=f"LLM analysis failed: {exc}",
                dataset_path=user_path_abs,
                sample_observation=sample_observation,
                path_facts=path_facts,
                exploration_report=report,
            )

    def _structured_from_report(self, report: str) -> Dict[str, Any]:
        """Second LLM pass: map free-form report to the legacy JSON fields for synth/graph."""
        prompt = DATASET_JSON_FROM_REPORT_PROMPT.format(report=report)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        text = _stringify_message_content(response.content)
        try:
            data = self._parse_json(text)
        except Exception:
            logger.warning("DatasetAnalyzer: JSON extraction failed; using minimal fallback fields")
            data = {
                "modality": "unknown",
                "task_type": "unknown",
                "format": "custom",
                "description": report[:2000],
                "classes": [],
                "splits": {"train": None, "val": None, "test": None},
                "structure_summary": report[:1500],
                "recommended_config": {"key_path": "", "notes": "See exploration_report for details."},
            }
        data.setdefault("modality", "unknown")
        data.setdefault("task_type", "unknown")
        data.setdefault("format", "custom")
        data.setdefault("description", "")
        data.setdefault("classes", [])
        data.setdefault("splits", {"train": None, "val": None, "test": None})
        data.setdefault("structure_summary", "")
        rc = data.setdefault("recommended_config", {})
        if not isinstance(rc, dict):
            rc = {}
            data["recommended_config"] = rc
        rc.setdefault("key_path", "")
        rc.setdefault("notes", "")
        return data

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]
        return json.loads(content.strip())
