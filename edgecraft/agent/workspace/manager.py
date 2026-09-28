"""WorkspaceManager: per-trial directory isolation for EdgeCraft.

In the code-centric architecture, each trial workspace contains:

    workspaces/{tenant_scope}/{run_id}/{trial_id}/
        train.py        ← LLM-generated training script
        infer.py        ← LLM-generated inference/benchmark script
        config/         ← dataset config (data.yaml)
        outputs/        ← model artifacts (best.pt, best.onnx, etc.)
        meta.json       ← SolutionVariant snapshot

The train.py and infer.py scripts are executed via subprocess.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from loguru import logger
from edgecraft.agent.metrics import ultralytics_results_dict_key
from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.config.settings import settings

if TYPE_CHECKING:
    from edgecraft.agent.modality_handler import ModalityHandler


_WORKSPACE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_workspace_component(value: str, *, label: str) -> str:
    """Reject traversal and ambiguous path components at the workspace boundary."""
    text = str(value or "").strip()
    if not _WORKSPACE_COMPONENT.fullmatch(text) or text in {".", ".."}:
        raise ValueError(
            f"invalid {label}: use 1-128 letters, digits, dots, underscores, or hyphens"
        )
    return text


def tenant_workspace_key(tenant_id: str) -> str:
    """Return an opaque stable namespace without placing tenant text in paths."""
    tenant = validate_workspace_component(tenant_id or "default", label="tenant_id")
    return "tenant_" + hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:16]


def default_workspace_root() -> Path:
    return Path(settings.EDGECRAFT_ROOT) / "workspaces"


def run_workspace_path(
    run_id: str,
    *,
    tenant_id: str = "default",
    base_dir: Optional[Path] = None,
) -> Path:
    run = validate_workspace_component(run_id, label="run_id")
    return (base_dir or default_workspace_root()) / tenant_workspace_key(tenant_id) / run


def ensure_private_run_workspace(
    run_id: str,
    *,
    tenant_id: str = "default",
    base_dir: Optional[Path] = None,
) -> Path:
    """Create tenant/run directories with owner-only POSIX permissions."""
    run_path = run_workspace_path(run_id, tenant_id=tenant_id, base_dir=base_dir)
    for path in (run_path.parent, run_path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass
    return run_path


def trial_bank_path(
    run_id: str,
    *,
    tenant_id: str = "default",
    base_dir: Optional[Path] = None,
) -> Path:
    return run_workspace_path(run_id, tenant_id=tenant_id, base_dir=base_dir) / "trial_bank.json"


class WorkspaceManager:
    """Creates and manages per-trial workspace directories."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir: Path = base_dir or default_workspace_root()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _tenant_weights_cache(self, tenant_id: str) -> Path:
        """Return a tenant-private cache shared only by that tenant's trials."""
        return (
            self.base_dir
            / tenant_workspace_key(tenant_id)
            / ".cache"
            / "pretrained_weights"
        )

    def create_trial_workspace(
        self,
        run_id: str,
        trial_id: str,
        *,
        tenant_id: str = "default",
    ) -> Path:
        """Create the directory skeleton for a trial and return its root path."""
        trial = validate_workspace_component(trial_id, label="trial_id")
        run_path = ensure_private_run_workspace(
            run_id,
            tenant_id=tenant_id,
            base_dir=self.base_dir,
        )
        ws = run_path / trial
        # Create required directories
        (ws / "config").mkdir(parents=True, exist_ok=True)
        (ws / "outputs").mkdir(parents=True, exist_ok=True)
        try:
            ws.chmod(0o700)
        except OSError:
            pass
        # Canonical workspace-local cache for pretrained/model artifacts.
        (ws / "artifacts" / "pretrained").mkdir(parents=True, exist_ok=True)

        # Reuse public/pretrained downloads across this tenant's trials without
        # exposing a cache populated by a different tenant.
        weights_cache = self._tenant_weights_cache(tenant_id)
        weights_cache.mkdir(parents=True, exist_ok=True)
        weights_link = ws / "weights"
        if not weights_link.exists() and not weights_link.is_symlink():
            weights_link.symlink_to(weights_cache.resolve())
            logger.debug(f"weights/ → tenant cache: {weights_cache}")

        logger.debug(f"Workspace created: {ws}")
        return ws

    def materialize_variant(
        self,
        variant: SolutionVariant,
        ws: Path,
        dataset_path: str,
        handler: "ModalityHandler",
    ) -> None:
        """Write scripts and config files into workspace.

        In code-centric architecture:
        - train.py and infer.py come from SolutionVariant code fields
        - config/data.yaml is written by ModalityHandler
        - meta.json captures the full SolutionVariant
        """
        parent_ws = self._parent_workspace(ws, variant)

        inherited_train = self._inherited_component_code(parent_ws, variant, "train.py")
        train_code = inherited_train or self.prepare_script(
            variant.train_code or "", variant=variant, stage="train"
        )
        if not train_code:
            logger.warning(f"No train_code in variant {variant.trial_id}, writing placeholder")
            train_code = (
                "#!/usr/bin/env python3\n"
                "# No train_code provided\n"
                "import json\n"
                "print(json.dumps({'status': 'error', 'error': 'No train_code provided'}))\n"
            )
        variant.train_code = train_code
        (ws / "train.py").write_text(train_code)

        inherited_loader = self._inherited_component_code(parent_ws, variant, "loader.py")
        loader_code_raw = getattr(variant, "loader_code", "") or ""
        if inherited_loader is not None:
            loader_code = inherited_loader
            variant.loader_code = inherited_loader
            (ws / "loader.py").write_text(inherited_loader)
            logger.debug(f"Inherited loader.py from parent for {variant.trial_id}")
        elif loader_code_raw.strip():
            loader_code = self.prepare_script(
                loader_code_raw,
                variant=variant,
                stage="loader",
            )
            variant.loader_code = loader_code
            (ws / "loader.py").write_text(loader_code)

        inherited_infer = self._inherited_component_code(parent_ws, variant, "infer.py")
        if inherited_infer is not None:
            infer_code = inherited_infer
            logger.debug(f"Inherited infer.py from parent for {variant.trial_id}")
        elif (variant.infer_code or "").strip():
            infer_code = self.prepare_script(
                variant.infer_code or "",
                variant=variant,
                stage="infer",
            )
        else:
            infer_code = ""

        if not infer_code:
            logger.warning(f"No infer_code in variant {variant.trial_id}, writing placeholder")
            infer_code = (
                "#!/usr/bin/env python3\n"
                "# No infer_code provided\n"
                "import json\n"
                "print(json.dumps({'status': 'error', 'error': 'No infer_code provided'}))\n"
            )
        variant.infer_code = infer_code
        (ws / "infer.py").write_text(infer_code)

        # Delegate dataset config writing to ModalityHandler
        handler.materialize_config(variant, ws / "config", dataset_path)

        # Write meta.json after inheritance resolution, so the snapshot matches
        # the executable scripts materialized for this node.
        (ws / "meta.json").write_text(
            json.dumps(variant.model_dump(mode="json"), indent=2)
        )

        logger.debug(f"Variant materialised into {ws} with train.py, optional loader.py, and infer.py")

    @staticmethod
    def prepare_script(code: str, *, variant: SolutionVariant, stage: str) -> str:
        """Materialize trusted code verbatim when normalization is explicitly disabled."""
        if not getattr(settings, "NORMALIZE_GENERATED_CODE", True):
            return code
        return WorkspaceManager.normalize_script_contract(code, variant=variant, stage=stage)

    @staticmethod
    def _parent_workspace(ws: Path, variant: SolutionVariant) -> Optional[Path]:
        parent_id = (getattr(variant, "parent_trial_id", None) or "").strip()
        if not parent_id:
            return None
        parent_ws = ws.parent / parent_id
        return parent_ws if parent_ws.exists() else None

    @staticmethod
    def _component_declared(items: List[str], filename: str) -> bool:
        text = "\n".join(str(item).lower() for item in (items or []))
        if not text:
            return False
        aliases = {
            "train.py": ("train.py", "train", "model", "training", "export"),
            "loader.py": ("loader.py", "loader", "dataset loader", "parser", "split"),
            "infer.py": ("infer.py", "infer", "runtime", "edge", "benchmark"),
        }.get(filename, (filename,))
        return any(alias in text for alias in aliases)

    @classmethod
    def _inherited_component_code(
        cls,
        parent_ws: Optional[Path],
        variant: SolutionVariant,
        filename: str,
    ) -> Optional[str]:
        if parent_ws is None:
            return None
        if not cls._component_declared(getattr(variant, "inherited_components", []), filename):
            return None
        if cls._component_declared(getattr(variant, "changed_components", []), filename):
            return None
        source = parent_ws / filename
        if not source.exists():
            return None
        return source.read_text()

    @staticmethod
    def normalize_script_contract(
        code: str,
        *,
        variant: SolutionVariant,
        stage: str,
    ) -> str:
        """Apply the workspace script contract in a stage/family-aware way.

        This is intentionally idempotent.  It is used both during first
        materialization and after LLM repair so retries cannot bypass the
        workspace/edge contracts.
        """
        if not code:
            return code

        stage_key = (stage or "").lower()
        family = (getattr(variant, "model_family", "") or "").lower()
        modality = str(getattr(variant, "modality", "") or "").lower()
        patched = WorkspaceManager._normalize_common_script_paths(code)

        if stage_key == "loader":
            patched = WorkspaceManager._normalize_project_dataset_literals(patched)
            patched = WorkspaceManager._normalize_loader_smoke_entrypoint(patched)
            if "audio" in modality:
                patched = WorkspaceManager._patch_hf_audio_decode_false(patched)
                patched = WorkspaceManager._patch_torchaudio_load_wave_fallback(patched)
            return patched

        if stage_key == "train":
            patched = WorkspaceManager._normalize_project_dataset_literals(patched)
            if family == "ultralytics":
                patched = WorkspaceManager._patch_ultralytics_train_api(patched)
            if family in {"transformers", "huggingface"}:
                patched = WorkspaceManager._patch_transformers_export_api(patched)
                patched = WorkspaceManager._patch_transformers_torch_onnx_export(patched)
            if family == "whisper" or "audio" in modality:
                patched = WorkspaceManager._patch_hf_audio_decode_false(patched)
                patched = WorkspaceManager._patch_audio_torchcodec_fallback(patched)
                patched = WorkspaceManager._patch_torchaudio_load_wave_fallback(patched)
                patched = WorkspaceManager._patch_whisper_seq2seq_collator(patched)
            patched = WorkspaceManager._patch_metric_api_compat(patched)
            patched = WorkspaceManager._cap_train_epochs(patched)
            return patched

        if stage_key in {"infer", "edge_benchmark"}:
            patched = WorkspaceManager._remove_edge_runtime_installs(patched)
            if family not in {"tiny_text_hash"}:
                patched = WorkspaceManager._normalize_infer_model_path(
                    patched, getattr(variant, "export_format", None) or "onnx"
                )
            if family in {"transformers", "huggingface"} or "text" in modality:
                patched = WorkspaceManager._patch_infer_edge_local_dataset(patched)
                patched = WorkspaceManager._patch_transformers_infer_runtime(patched)
            if family == "ultralytics":
                patched = WorkspaceManager._strip_foreign_hf_export_imports(patched)
            if not bool(getattr(settings, "EDGE_EVAL_WITH_DATASET", False)):
                patched = WorkspaceManager._replace_real_image_with_dummy(patched)
            return patched

        return patched

    @staticmethod
    def _patch_torchaudio_load_wave_fallback(code: str) -> str:
        """Avoid torchcodec-backed torchaudio.load for local WAV reads."""
        if not code or "torchaudio.load" not in code:
            return code
        helper = (
            "def _edgecraft_load_wav(path):\n"
            "    import wave\n"
            "    import numpy as np\n"
            "    import torch\n"
            "    with wave.open(str(path), 'rb') as wf:\n"
            "        channels = max(1, wf.getnchannels())\n"
            "        width = wf.getsampwidth()\n"
            "        sr = wf.getframerate()\n"
            "        frames = wf.readframes(wf.getnframes())\n"
            "    dtype = np.uint8 if width == 1 else np.int16\n"
            "    arr = np.frombuffer(frames, dtype=dtype).astype(np.float32)\n"
            "    arr = (arr - 128.0) / 128.0 if width == 1 else arr / 32768.0\n"
            "    if channels > 1:\n"
            "        arr = arr.reshape(-1, channels).mean(axis=1)\n"
            "    return torch.from_numpy(arr.copy()).float().unsqueeze(0), int(sr)\n"
        )
        patched = code.replace("torchaudio.load(", "_edgecraft_load_wav(")
        if "_edgecraft_load_wav" not in code:
            patched = WorkspaceManager._insert_after_module_header(patched, helper + "\n")
        return patched

    @staticmethod
    def _patch_hf_audio_decode_false(code: str) -> str:
        """Avoid torchcodec for local HuggingFace Audio datasets."""
        if not code or "load_from_disk" not in code or "audio" not in code:
            return code
        patched = code
        patched = re.sub(
            r"^([ \t]*)ds\s*=\s*ds\.cast_column\(\s*[\"']audio[\"']\s*,\s*ds\.features\[[\"']audio[\"']\]\.cast_to\(\s*[\"']array[\"']\s*\)\s*\)\s*$",
            lambda m: f"{m.group(1)}pass",
            patched,
            flags=re.MULTILINE,
        )
        patched = re.sub(
            r"([A-Za-z_][A-Za-z0-9_\[\]\.]*?)\.cast_column\(\s*[\"']audio[\"']\s*,\s*None\s*\)",
            lambda m: f"{m.group(1)}.cast_column('audio', Audio(decode=False))",
            patched,
        )
        patched = re.sub(r"Audio\(([^)]*)decode\s*=\s*True([^)]*)\)", r"Audio(\1decode=False\2)", patched)
        if 'x["array"]' in patched or "x['array']" in patched:
            helper = '\n\ndef _edgecraft_audio_array(x):\n    if isinstance(x, dict):\n        if x.get("array") is not None:\n            return np.asarray(x["array"], dtype=np.float32)\n        if x.get("bytes"):\n            import io\n            with wave.open(io.BytesIO(x["bytes"]), "rb") as wf:\n                channels = max(1, wf.getnchannels())\n                width = wf.getsampwidth()\n                frames = wf.readframes(wf.getnframes())\n                dtype = np.uint8 if width == 1 else np.int16\n                arr = np.frombuffer(frames, dtype=dtype).astype(np.float32)\n                arr = (arr - 128.0) / 128.0 if width == 1 else arr / 32768.0\n                if channels > 1:\n                    arr = arr.reshape(-1, channels).mean(axis=1)\n                return arr.astype(np.float32)\n        if x.get("path"):\n            with wave.open(str(x["path"]), "rb") as wf:\n                channels = max(1, wf.getnchannels())\n                width = wf.getsampwidth()\n                frames = wf.readframes(wf.getnframes())\n                dtype = np.uint8 if width == 1 else np.int16\n                arr = np.frombuffer(frames, dtype=dtype).astype(np.float32)\n                arr = (arr - 128.0) / 128.0 if width == 1 else arr / 32768.0\n                if channels > 1:\n                    arr = arr.reshape(-1, channels).mean(axis=1)\n                return arr.astype(np.float32)\n    return np.asarray(x, dtype=np.float32)\n'
            if '_edgecraft_audio_array' not in patched:
                patched = helper + "\n" + patched
            patched = patched.replace('np.array(x["array"], dtype=np.float32)', '_edgecraft_audio_array(x)')
            patched = patched.replace("np.array(x['array'], dtype=np.float32)", '_edgecraft_audio_array(x)')
        path_branch = (
            '                elif "path" in audio_info and audio_info["path"]:\n'
            '                    arr = _read_wav(Path(audio_info["path"]))\n'
        )
        bytes_branch = (
            '                elif "bytes" in audio_info and audio_info["bytes"]:\n'
            '                    import io\n'
            '                    with wave.open(io.BytesIO(audio_info["bytes"]), "rb") as wf:\n'
            '                        channels = max(1, wf.getnchannels())\n'
            '                        width = wf.getsampwidth()\n'
            '                        frames = wf.readframes(wf.getnframes())\n'
            '                        dtype = np.uint8 if width == 1 else np.int16\n'
            '                        arr = np.frombuffer(frames, dtype=dtype).astype(np.float32)\n'
            '                        arr = (arr - 128.0) / 128.0 if width == 1 else arr / 32768.0\n'
            '                        if channels > 1:\n'
            '                            arr = arr.reshape(-1, channels).mean(axis=1)\n'
            + path_branch
        )
        if "audio_info" in patched and "audio_info[\"bytes\"]" not in patched and path_branch in patched:
            patched = patched.replace(path_branch, bytes_branch)
        patched = patched.replace("if waveform is not None:\n            samples.append({\"waveform\": waveform, \"label\": d[\"label\"]})", "if audio is not None:\n            samples.append({\"audio\": audio, \"label\": d.get(\"label\")})")
        patched = patched.replace("if waveform is not None:\n            samples.append({\"waveform\": waveform, \"label\": d[\"action\"]})", "if audio is not None:\n            samples.append({\"audio\": audio, \"label\": d.get(\"action\")})")
        if "from datasets import" in patched and "Audio" not in patched.split("from datasets import", 1)[1].split("\n", 1)[0]:
            patched = re.sub(
                r"from datasets import ([^\n]+)",
                lambda m: (
                    m.group(0)
                    if "Audio" in m.group(1)
                    else f"from datasets import {m.group(1).rstrip()}, Audio"
                ),
                patched,
                count=1,
            )
        elif "from datasets import" not in patched:
            patched = "from datasets import Audio\n" + patched

        lines = patched.splitlines()
        out: List[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            stripped = line.strip()
            match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*load_from_disk\(", stripped)
            if not inserted and match:
                var_name = match.group(1)
                indent = line[: len(line) - len(line.lstrip())]
                out.extend(
                    [
                        f"{indent}try:",
                        f"{indent}    if 'audio' in getattr({var_name}, 'column_names', []):",
                        f"{indent}        {var_name} = {var_name}.cast_column('audio', Audio(decode=False))",
                        f"{indent}    for _split in list({var_name}.keys()) if hasattr({var_name}, 'keys') else []:",
                        f"{indent}        if 'audio' in getattr({var_name}[_split], 'column_names', []):",
                        f"{indent}            {var_name}[_split] = {var_name}[_split].cast_column('audio', Audio(decode=False))",
                        f"{indent}except Exception:",
                        f"{indent}    pass",
                    ]
                )
                inserted = True
        return "\n".join(out) + ("\n" if patched.endswith("\n") else "")

    @staticmethod
    def _remove_edge_runtime_installs(code: str) -> str:
        """Remove runtime dependency installation from edge infer scripts.

        Edge jobs run in prebuilt images.  Installing onnxruntime-gpu on Jetson
        at benchmark time is slow, often impossible, and should not block
        crafting progress.
        """
        if not code:
            return code
        lines = []
        for line in code.splitlines(keepends=True):
            lower = line.lower()
            if (
                "pip install" in lower
                and any(pkg in lower for pkg in ("onnxruntime-gpu", "onnxruntime", "tensorrt"))
            ):
                lines.append("# EdgeCraft removed runtime pip install; use prebuilt Docker runtime providers.\n")
                continue
            if (
                "subprocess" in lower
                and "pip" in lower
                and any(pkg in lower for pkg in ("onnxruntime-gpu", "onnxruntime", "tensorrt"))
            ):
                lines.append("# EdgeCraft removed runtime pip install subprocess call.\n")
                continue
            lines.append(line)
        return "".join(lines)

    @staticmethod
    def _normalize_common_script_paths(code: str) -> str:
        """Patch common LLM path mistakes so scripts are runnable in workspace.

        Current workspace contract:
        - dataset config is always at config/data.yaml
        """
        if not code:
            return code
        patched = code

        # data='data.yaml' / data="./data.yaml" -> data='config/data.yaml'
        patched = re.sub(
            r"(\bdata\s*=\s*[\"'])(?:\./)?data\.ya?ml([\"'])",
            r"\1config/data.yaml\2",
            patched,
        )
        # Normalize named config assignments, but do not rewrite a path segment
        # in an already-correct expression such as Path(root) / "config" / "data.yaml".
        patched = re.sub(
            r"(\b(?:data|data_yaml|config_path)\s*=\s*[\"'])(?:\./)?data\.ya?ml([\"'])",
            r"\1config/data.yaml\2",
            patched,
        )
        # Absolute or project-relative dataset yaml literals bypass the
        # materialized workspace config and lose path normalization.
        patched = re.sub(
            r"([\"'])(?:/[^\"']*/)?datasets/[^\"']+/data[^/\"']*\.ya?ml\1",
            r"\1config/data.yaml\1",
            patched,
        )
        # Path("data.yaml") -> Path("config/data.yaml")
        patched = re.sub(
            r"(Path\([\"'])(?:\./)?data\.ya?ml([\"']\))",
            r"\1config/data.yaml\2",
            patched,
        )

        # Canonicalize cache/artifact paths to workspace-writable locations.
        # This avoids permission errors from absolute paths like /home/edgecraft/.cache/...
        patched = WorkspaceManager._normalize_cache_paths(patched)
        return patched

    @staticmethod
    def _normalize_loader_smoke_entrypoint(code: str) -> str:
        """Keep loader smoke anchored to the materialized workspace config.

        LLMs often call load_train_val("datasets/foo/data.yaml") in the
        __main__ smoke block after correctly defining load_train_val(config_path).
        That bypasses config/data.yaml, loses path normalization, and breaks
        evidence provenance. The loader may still read dataset_root from config;
        only the smoke entrypoint is canonicalized.
        """
        if not code:
            return code
        patched = code
        patched = re.sub(
            r"\b(load_train_val|load_test)\(\s*([\"'])(?!config/data\.ya?ml)[^\"']*\.ya?ml\2",
            r"\1(\2config/data.yaml\2",
            patched,
        )
        patched = re.sub(
            r"\b(load_train_val|load_test)\(\s*([\"'])(?:datasets|/data|/home|/tmp)[^\"']*\2",
            r"\1(\2config/data.yaml\2",
            patched,
        )
        return patched

    @staticmethod
    def _normalize_project_dataset_literals(code: str) -> str:
        """Resolve bare datasets/... reads from trial workspace to project root."""
        if not code:
            return code
        project_root = str(Path(settings.EDGECRAFT_ROOT).resolve()).replace("\\", "/")
        return re.sub(
            r"([\"'])datasets/([^\"']+)([\"'])",
            lambda m: f"{m.group(1)}{project_root}/datasets/{m.group(2)}{m.group(3)}",
            code,
        )

    @staticmethod
    def _normalize_cache_paths(code: str) -> str:
        """Normalize LLM-generated cache/model paths to workspace-local directories.

        Canonical contract inside trial workspace:
        - artifacts/pretrained/ : downloaded/pretrained weights
        - .cache/               : generic process/cache files
        """
        if not code:
            return code
        patched = code

        # Quoted absolute cache paths -> workspace-local cache.
        # Catch any /home/<user>/.cache/edgecraft/model_weights patterns
        patched = re.sub(
            r"([\"'])/home/[^/\"']+/\.cache/edgecraft/model_weights([\"'])",
            r"\1artifacts/pretrained\2",
            patched,
        )
        patched = re.sub(
            r"([\"'])/home/[^/\"']+/\.cache(?:/edgecraft)?([\"'])",
            r"\1.cache\2",
            patched,
        )
        patched = re.sub(
            r"([\"'])~/.cache(?:/edgecraft)?([\"'])",
            r"\1.cache\2",
            patched,
        )

        # pathlib.Path absolute cache roots -> workspace-local paths.
        patched = re.sub(
            r"Path\(\s*[\"']/home/[^/\"']+/\.cache/edgecraft/model_weights[\"']\s*\)",
            'Path("artifacts/pretrained")',
            patched,
        )
        patched = re.sub(
            r"Path\(\s*[\"']/home/[^/\"']+/\.cache(?:/edgecraft)?[\"']\s*\)",
            'Path(".cache")',
            patched,
        )

        return patched

    @staticmethod
    def _patch_ultralytics_train_api(code: str) -> str:
        """Fix common LLM mistakes against current Ultralytics YOLO APIs.

        - ``results.save(path)`` does not exist (val metrics are DetMetrics; train has ``save_dir``).
        - Prefer copying ``best.pt`` from ``Path(results.save_dir) / 'weights' / 'best.pt'``.
        - Metrics belong in ``results.results_dict`` (e.g. ``metrics/mAP50-95(B)``).
        """
        if not code:
            return code
        if (
            "ultralytics" not in code.lower()
            and "results.save" not in code
            and "train_results.save" not in code
            and "results.metrics" not in code
            and "train_results.metrics" not in code
            and "data={" not in code.replace(" ", "")
            and not any(old in code for old in ("yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov3"))
        ):
            return code
        patched = code

        try:
            tree = ast.parse(patched)
            replacements: list[tuple[str, str]] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg == "data" and isinstance(kw.value, ast.Dict):
                        segment = ast.get_source_segment(patched, kw.value)
                        if segment:
                            replacements.append((f"data={segment}", 'data="config/data.yaml"'))
            for old, new in sorted(set(replacements), key=lambda item: len(item[0]), reverse=True):
                patched = patched.replace(old, new)
        except SyntaxError:
            pass

        # Wrong: results.save(...) or train_results.save(...) after model.train()
        def _repl_save(m: re.Match) -> str:
            var = m.group(1)
            return (
                f"best_src = Path({var}.save_dir) / \"weights\" / \"best.pt\"\n"
                f"if best_src.exists():\n"
                f"    shutil.copy2(best_src, model_path)"
            )

        patched = re.sub(
            r"^[ \t]*(results|train_results)\.save\s*\([^)]*\)\s*$",
            _repl_save,
            patched,
            flags=re.MULTILINE,
        )

        # Rewrite only real model.train(...) keyword nodes. A nested-parenthesis
        # regex can backtrack catastrophically on generated training recipes.
        keyword_names = {
            "batch_size": "batch",
            "learning_rate": "lr0",
            "lr": "lr0",
            "image_size": "imgsz",
            "img_size": "imgsz",
            "weightdecay": "weight_decay",
        }
        try:
            tree = ast.parse(patched)
            lines = patched.splitlines(keepends=True)
            line_offsets = [0]
            for line in lines:
                line_offsets.append(line_offsets[-1] + len(line))

            renames: list[tuple[int, int, str]] = []
            for node in ast.walk(tree):
                is_model_train = (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "train"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "model"
                )
                if not is_model_train:
                    continue
                call_start = line_offsets[node.lineno - 1] + node.col_offset
                for keyword in node.keywords:
                    replacement = keyword_names.get(keyword.arg or "")
                    if not replacement:
                        continue
                    value_start = (
                        line_offsets[keyword.value.lineno - 1]
                        + keyword.value.col_offset
                    )
                    name_start = patched.rfind(keyword.arg, call_start, value_start)
                    if name_start < 0:
                        continue
                    between = patched[name_start + len(keyword.arg):value_start]
                    if re.fullmatch(r"\s*=\s*", between):
                        renames.append(
                            (name_start, name_start + len(keyword.arg), replacement)
                        )
            for start, end, replacement in sorted(renames, reverse=True):
                patched = patched[:start] + replacement + patched[end:]
        except SyntaxError:
            pass

        # Wrong: results.metrics[...] or results.metrics.xxx on train output.
        # Route all supported names through unified MetricRegistry mapping.
        def _replace_bracket_metric(match: re.Match) -> str:
            var_name = match.group(1)
            metric_name = match.group(2)
            key = ultralytics_results_dict_key(metric_name)
            if not key:
                return match.group(0)
            return f'float({var_name}.results_dict.get("{key}", 0))'

        def _replace_dot_metric(match: re.Match) -> str:
            var_name = match.group(1)
            metric_name = match.group(2)
            key = ultralytics_results_dict_key(metric_name)
            if not key:
                return match.group(0)
            return f'float({var_name}.results_dict.get("{key}", 0))'

        patched = re.sub(
            r"(results|train_results)\.metrics\s*\[\s*[\"']([A-Za-z0-9_]+)[\"']\s*\]",
            _replace_bracket_metric,
            patched,
        )
        patched = re.sub(
            r"(results|train_results)\.metrics\.([A-Za-z0-9_]+)\b",
            _replace_dot_metric,
            patched,
        )

        # Wrong: bare ``results.metrics`` assigned to a variable or used directly.
        # LLM Debugger often generates: metrics = results.metrics
        # Correct form is: metrics = results.results_dict
        patched = re.sub(
            r"\b(results|train_results)\.metrics\b(?!\s*[\[.(])",
            r"\1.results_dict",
            patched,
        )

        # Wrong: the YOLO auto-rename bug — yolov5n is silently renamed to yolov5nu
        # by Ultralytics.  Normalise known renamed models directly in the path so the
        # download target matches the cached filename Ultralytics actually creates.
        _YOLO_RENAMES = {
            "yolov5n": "yolov5nu", "yolov5s": "yolov5su", "yolov5m": "yolov5mu",
            "yolov5l": "yolov5lu", "yolov5x": "yolov5xu",
            "yolov3": "yolov3u", "yolov3-spp": "yolov3-sppu",
        }
        for _old, _new in _YOLO_RENAMES.items():
            # Replace f-string variable assignment: model_name = "yolov5n" -> model_name = "yolov5nu"
            # Do this FIRST so the string-literal regex below doesn't double-fire.
            patched = patched.replace(f'model_name = "{_old}"', f'model_name = "{_new}"')
            # Replace quoted path strings: "yolov5n.pt" -> "yolov5nu.pt"
            # Only match when .pt suffix is present to avoid renaming bare name variables.
            patched = re.sub(
                r'(["\'])' + re.escape(_old) + r'\.pt\1',
                lambda m, n=_new: m.group(1) + n + ".pt" + m.group(1),
                patched,
            )

        # Wrong: YOLO(f"{model_name}.pt") or YOLO("yolov8n.pt") downloads to cwd,
        # not to cache_dir. Replace bare YOLO(name.pt) calls in the download branch
        # with YOLO(str(cache_dir / f"{model_name}.pt")) so the file lands in cache.
        patched = re.sub(
            r'YOLO\(f["\']\{model_name\}\.pt["\']\)',
            'YOLO(str(cache_dir / f"{model_name}.pt"))',
            patched,
        )

        # Wrong: model.export(format="onnx") uses opset 20 by default which breaks
        # CPUExecutionProvider on older ONNX Runtime versions (Jetson jp6 docker).
        # Force opset=12 which has broad kernel support.
        patched = re.sub(
            r'(model\.export\s*\([^)]*format\s*=\s*["\']onnx["\'][^)]*?)\)',
            lambda m: m.group(0) if 'opset' in m.group(0) else m.group(1) + ', opset=12)',
            patched,
        )

        needs_shutil = "shutil.copy2" in patched and "import shutil" not in patched
        needs_path = "Path(" in patched and "from pathlib import Path" not in patched
        if needs_shutil or needs_path:
            extra = ""
            if needs_path:
                extra += "from pathlib import Path\n"
            if needs_shutil:
                extra += "import shutil\n"
            patched = WorkspaceManager._insert_after_module_header(patched, extra)
        return patched

    @staticmethod
    def _patch_transformers_export_api(code: str) -> str:
        """Block unstable transformers.onnx.export and rewrite to a stable helper.

        Some generated scripts call ``transformers.onnx.export`` with signatures
        that drift across transformers versions (e.g. missing ``config`` arg).
        We normalize that call path to a torch.onnx-based helper with a stable
        keyword contract.
        """
        if not code:
            return code
        if (
            "transformers.onnx.export" not in code
            and "from transformers.onnx import export" not in code
        ):
            return code

        patched = code
        # Remove unstable import path.
        patched = re.sub(
            r"^[ \t]*from\s+transformers\.onnx\s+import\s+export\s*$",
            "",
            patched,
            flags=re.MULTILINE,
        )

        # Redirect old call sites to stable helper while preserving kwargs.
        patched = re.sub(r"\btransformers\.onnx\.export\s*\(", "stable_hf_export_onnx(", patched)
        patched = re.sub(r"(?<!\.)\bexport\s*\(", "stable_hf_export_onnx(", patched)

        patched = WorkspaceManager._ensure_stable_hf_export_helper(patched)
        return patched

    @staticmethod
    def _ensure_stable_hf_export_helper(code: str) -> str:
        """Ensure the local HF ONNX export helper exists."""
        if not code or "stable_hf_export_onnx(" not in code:
            return code
        patched = code
        helper = (
            "def stable_hf_export_onnx(\n"
            "    *,\n"
            "    model,\n"
            "    preprocessor=None,\n"
            "    tokenizer=None,\n"
            "    output=None,\n"
            "    output_path=None,\n"
            "    max_length=128,\n"
            "    opset=13,\n"
            "    **kwargs,\n"
            "):\n"
            "    tok = tokenizer or preprocessor\n"
            "    if tok is None:\n"
            "        raise ValueError(\"stable_hf_export_onnx requires tokenizer/preprocessor\")\n"
            "    out_path = output_path or output\n"
            "    if out_path is None:\n"
            "        raise ValueError(\"stable_hf_export_onnx requires output path\")\n"
            "    input_ids = torch.ones((1, int(max_length)), dtype=torch.long)\n"
            "    attention_mask = torch.ones((1, int(max_length)), dtype=torch.long)\n"
            "    _device = next(model.parameters()).device\n"
            "    model = model.eval().cpu()\n"
            "\n"
            "    class _HFExportWrapper(torch.nn.Module):\n"
            "        def __init__(self, wrapped_model):\n"
            "            super().__init__()\n"
            "            self.wrapped_model = wrapped_model\n"
            "\n"
            "        def forward(self, input_ids, attention_mask):\n"
            "            out = self.wrapped_model(input_ids=input_ids, attention_mask=attention_mask)\n"
            "            return out.logits if hasattr(out, \"logits\") else out[0]\n"
            "\n"
            "    torch.onnx.export(\n"
            "        _HFExportWrapper(model),\n"
            "        (input_ids, attention_mask),\n"
            "        str(out_path),\n"
            "        input_names=[\"input_ids\", \"attention_mask\"],\n"
            "        output_names=[\"logits\"],\n"
            "        dynamic_axes={\n"
            "            \"input_ids\": {0: \"batch\", 1: \"seq\"},\n"
            "            \"attention_mask\": {0: \"batch\", 1: \"seq\"},\n"
            "            \"logits\": {0: \"batch\"},\n"
            "        },\n"
            "        opset_version=max(int(opset), 18),\n"
            "    )\n"
            "    model = model.to(_device)\n"
            "    return str(out_path)\n"
        )
        if "def stable_hf_export_onnx(" not in patched:
            patched = WorkspaceManager._insert_after_module_header(patched, helper + "\n")

        needs_torch = "stable_hf_export_onnx(" in patched and "import torch" not in patched
        if needs_torch:
            patched = WorkspaceManager._insert_after_module_header(patched, "import torch\n")
        return patched

    @staticmethod
    def _fix_stable_hf_export_scope_refs(code: str) -> str:
        """Resolve tokenizer/processor from local or global scope in export calls."""
        if not code or "stable_hf_export_onnx" not in code:
            return code
        patched = re.sub(
            r"tokenizer=tokenizer if 'tokenizer' in globals\(\) else None",
            "tokenizer=locals().get('tokenizer') or globals().get('tokenizer')",
            code,
        )
        patched = re.sub(
            r"preprocessor=processor if 'processor' in globals\(\) else None",
            "preprocessor=locals().get('processor') or globals().get('processor')",
            patched,
        )
        return patched

    @staticmethod
    def _patch_transformers_torch_onnx_export(code: str) -> str:
        """Normalize raw HF ``torch.onnx.export`` calls onto CPU.

        PyTorch 2.x's dynamo exporter can fail when a fine-tuned HF model is on
        CUDA while dummy inputs are on CPU.  For generated Transformers train
        scripts, route ONNX export through the same stable helper used for the
        deprecated ``transformers.onnx.export`` path.
        """
        code = WorkspaceManager._fix_stable_hf_export_scope_refs(code)
        if not code or "torch.onnx.export" not in code:
            return code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        lines = code.splitlines(keepends=True)
        line_offsets: list[int] = []
        total = 0
        for line in lines:
            line_offsets.append(total)
            total += len(line)

        def _is_torch_onnx_export(node: ast.Call) -> bool:
            func = node.func
            return (
                isinstance(func, ast.Attribute)
                and func.attr == "export"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "onnx"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "torch"
            )

        def _output_literal(node: ast.Call) -> str:
            if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                if isinstance(node.args[2].value, str):
                    return repr(node.args[2].value)
            for kw in node.keywords:
                if kw.arg in {"f", "output", "output_path"} and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        return repr(kw.value.value)
            return repr("outputs/best.onnx")

        helper_ranges = [
            (node.lineno, node.end_lineno or node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "stable_hf_export_onnx"
        ]

        def _inside_helper(node: ast.AST) -> bool:
            lineno = getattr(node, "lineno", 0)
            return any(start <= lineno <= end for start, end in helper_ranges)

        replacements: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_torch_onnx_export(node):
                continue
            if _inside_helper(node):
                continue
            if not hasattr(node, "end_lineno") or node.end_lineno is None:
                continue
            start = line_offsets[node.lineno - 1] + node.col_offset
            end = line_offsets[node.end_lineno - 1] + node.end_col_offset
            replacements.append(
                (
                    start,
                    end,
                    (
                        "stable_hf_export_onnx("
                        "model=model, "
                        "tokenizer=locals().get('tokenizer') or globals().get('tokenizer'), "
                        "preprocessor=locals().get('processor') or globals().get('processor'), "
                        f"output={_output_literal(node)}, "
                        "max_length=max_length if 'max_length' in globals() else 128, "
                        "opset=18"
                        ")"
                    ),
                )
            )
        if not replacements:
            return code

        patched = code
        for start, end, replacement in sorted(replacements, reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        patched = WorkspaceManager._ensure_stable_hf_export_helper(patched)
        return WorkspaceManager._fix_stable_hf_export_scope_refs(patched)

    @staticmethod
    def _patch_metric_api_compat(code: str) -> str:
        """Replace removed ``datasets.load_metric`` with ``evaluate.load``."""
        if not code or "load_metric" not in code:
            return code

        patched_lines: list[str] = []
        inserted_evaluate_import = False
        for line in code.splitlines(keepends=True):
            m = re.match(r"^(\s*)from\s+datasets\s+import\s+(.+?)(\s*(?:#.*)?\n?)$", line)
            if not m:
                patched_lines.append(line)
                continue
            names = [part.strip() for part in m.group(2).split(",")]
            if "load_metric" not in names:
                patched_lines.append(line)
                continue
            names = [name for name in names if name != "load_metric"]
            if names:
                patched_lines.append(f"{m.group(1)}from datasets import {', '.join(names)}{m.group(3)}")
            patched_lines.append(f"{m.group(1)}from evaluate import load as _edgecraft_evaluate_load\n")
            inserted_evaluate_import = True

        patched = "".join(patched_lines)
        patched = re.sub(r"(?<!\.)\bload_metric\s*\(", "_edgecraft_evaluate_load(", patched)
        if "_edgecraft_evaluate_load(" in patched and not inserted_evaluate_import:
            if "from evaluate import load as _edgecraft_evaluate_load" not in patched:
                patched = WorkspaceManager._insert_after_module_header(
                    patched,
                    "from evaluate import load as _edgecraft_evaluate_load\n",
                )
        return patched

    @staticmethod
    def _strip_foreign_hf_export_imports(code: str) -> str:
        """Keep Ultralytics infer scripts free of HF export helper imports."""
        if not code:
            return code
        patched = re.sub(
            r"^[ \t]*from\s+ultralytics\.utils\.export\s+import\s+stable_hf_export_onnx[^\n]*\n",
            "",
            code,
            flags=re.MULTILINE,
        )
        patched = re.sub(
            r"^[ \t]*from\s+transformers\.onnx\s+import\s+export[^\n]*\n",
            "",
            patched,
            flags=re.MULTILINE,
        )
        return patched

    @staticmethod
    def _patch_audio_torchcodec_fallback(code: str) -> str:
        """Avoid datasets Audio/torchcodec path; load manifest CSV with soundfile."""
        if not code:
            return code
        if "audiofolder" not in code and "Audio(sampling_rate" not in code:
            return code

        helper = (
            "def _load_audio_manifest_dataset(data_dir, sample_rate=16000):\n"
            "    import csv\n"
            "    from pathlib import Path as _Path\n"
            "    import soundfile as sf\n"
            "    from datasets import Dataset, DatasetDict\n"
            "\n"
            "    root = _Path(data_dir)\n"
            "\n"
            "    def _read_split(split_name):\n"
            "        split_dir = root / split_name\n"
            "        meta = split_dir / 'metadata.csv'\n"
            "        if not meta.exists():\n"
            "            return None\n"
            "        rows = []\n"
            "        with meta.open(newline='', encoding='utf-8') as f:\n"
            "            for rec in csv.DictReader(f):\n"
            "                wav = split_dir / rec['file_name']\n"
            "                if not wav.exists():\n"
            "                    continue\n"
            "                arr, sr = sf.read(str(wav), dtype='float32')\n"
            "                rows.append({\n"
            "                    'audio': {'array': arr, 'sampling_rate': int(sr)},\n"
            "                    'text': rec.get('text', ''),\n"
            "                })\n"
            "        if not rows:\n"
            "            return None\n"
            "        return Dataset.from_list(rows)\n"
            "\n"
            "    out = {}\n"
            "    train_ds = _read_split('train')\n"
            "    if train_ds is None:\n"
            "        raise FileNotFoundError(f'No train/metadata.csv under {root}')\n"
            "    out['train'] = train_ds\n"
            "    for split in ('validation', 'val', 'test'):\n"
            "        ds = _read_split(split)\n"
            "        if ds is not None:\n"
            "            out[split if split != 'val' else 'validation'] = ds\n"
            "            break\n"
            "    return DatasetDict(out)\n"
        )
        patched = code
        if "_load_audio_manifest_dataset" not in patched:
            patched = WorkspaceManager._insert_after_module_header(patched, helper + "\n")

        # Replace audiofolder loader (sample_rate may be defined on a later line).
        patched = re.sub(
            r"(\w+)\s*=\s*load_dataset\(\s*[\"']audiofolder[\"']\s*,\s*data_dir\s*=\s*(?:str\()?(\w+)\)?\s*\)",
            r"\1 = _load_audio_manifest_dataset(\2, sample_rate if 'sample_rate' in dir() else 16000)",
            patched,
            count=1,
        )
        # Drop Audio/torchcodec cast — manifest loader already provides decoded arrays.
        patched = re.sub(
            r"^[ \t]*\w+\s*=\s*\w+\.cast_column\(\s*[\"']audio[\"']\s*,\s*Audio\([^)]*\)\s*\)\s*\n",
            "",
            patched,
            flags=re.MULTILINE,
        )
        return patched

    @staticmethod
    def _patch_whisper_seq2seq_collator(code: str) -> str:
        """Use feature/tokenizer pads for Whisper instead of DataCollatorForSeq2Seq."""
        if not code or "input_features" not in code:
            return code
        if "DataCollatorForSeq2Seq" not in code and "data_collator" not in code:
            return code
        if "_edgecraft_whisper_collate_fn" in code:
            return code

        helper = (
            "def _edgecraft_whisper_collate_fn(batch):\n"
            "    proc = processor\n"
            "    feats = [{'input_features': b['input_features']} for b in batch]\n"
            "    labels = [{'input_ids': b['labels']} for b in batch]\n"
            "    feat_batch = proc.feature_extractor.pad(feats, return_tensors='pt')\n"
            "    label_batch = proc.tokenizer.pad(labels, return_tensors='pt')\n"
            "    feat_batch['labels'] = label_batch['input_ids']\n"
            "    return feat_batch\n"
        )
        patched = WorkspaceManager._insert_after_module_header(code, helper + "\n")
        patched = re.sub(
            r"data_collator\s*=\s*DataCollatorForSeq2Seq\([^)]*\)",
            "data_collator = _edgecraft_whisper_collate_fn",
            patched,
        )
        return patched

    @staticmethod
    def _patch_transformers_infer_runtime(code: str) -> str:
        """Keep HF text infer on ONNXRuntime; Ultralytics YOLO is CV-only."""
        if not code:
            return code
        is_hf_text = (
            "AutoTokenizer" in code
            or "AutoModelForSequenceClassification" in code
            or ("transformers" in code and "YOLO" in code)
        )
        if not is_hf_text or "YOLO" not in code:
            return code

        patched = re.sub(
            r"^[ \t]*from\s+ultralytics\s+import\s+YOLO[^\n]*\n",
            "",
            code,
            flags=re.MULTILINE,
        )
        patched = re.sub(
            r"^[ \t]*import\s+ultralytics[^\n]*\n",
            "",
            patched,
            flags=re.MULTILINE,
        )
        if "_edgecraft_hf_ort_session" not in patched:
            helper = (
                "def _edgecraft_hf_ort_session():\n"
                "    import os\n"
                "    import onnxruntime as ort\n"
                "    candidates = ['outputs/best.onnx', 'outputs/best.engine']\n"
                "    model_path = next((p for p in candidates if os.path.exists(p)), 'outputs/best.onnx')\n"
                "    if model_path.endswith('.engine'):\n"
                "        model_path = 'outputs/best.onnx'\n"
                "    providers = [p for p in ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if p in ort.get_available_providers()]\n"
                "    return ort.InferenceSession(model_path, providers=providers)\n"
            )
            patched = WorkspaceManager._insert_after_module_header(patched, helper + "\n")

        # Drop try/except YOLO bootstrap; keep ONNXRuntime path only.
        patched = re.sub(
            r"try:\s*\n(?:[ \t].*\n)*?[ \t]*except\s+Exception\s*:\s*\n",
            "",
            patched,
            count=1,
        )
        if "sess = _edgecraft_hf_ort_session()" not in patched:
            patched = re.sub(
                r"sess\s*=\s*ort\.InferenceSession\([^)]+\)",
                "sess = _edgecraft_hf_ort_session()",
                patched,
                count=1,
            )
        patched = re.sub(
            r"if\s+['\"]sess['\"]\s+in\s+locals\(\)\s*:\s*\n(\s+)(.+?)\nelse:\s*\n\s+(.+?)(\n)",
            r"\1\2\4",
            patched,
            flags=re.DOTALL,
        )
        patched = re.sub(
            r"model\s*=\s*YOLO\([^)]*\)\s*\n",
            "",
            patched,
        )
        return patched

    @staticmethod
    def _patch_infer_edge_local_dataset(code: str) -> str:
        """Remove huggingface ``datasets`` dependency from edge infer.py."""
        if not code or "load_dataset" not in code:
            return code

        helper = (
            "class _EdgeTextDS:\n"
            "    def __init__(self, records):\n"
            "        self._records = records or []\n"
            "    def __getitem__(self, key):\n"
            "        if key == 'text':\n"
            "            return [r['text'] for r in self._records]\n"
            "        if key in ('label', 'labels'):\n"
            "            return [r['label'] for r in self._records]\n"
            "        raise KeyError(key)\n"
            "\n"
            "def _edge_load_text_eval_samples(max_samples=200):\n"
            "    import csv, json, os\n"
            "    from pathlib import Path as _Path\n"
            "    root = _Path(os.getenv('EDGE_DATASET_DIR', 'dataset'))\n"
            "    files = []\n"
            "    if root.is_dir():\n"
            "        for name in ('test.jsonl', 'test.json', 'validation.jsonl', 'val.jsonl', 'test.csv'):\n"
            "            p = root / name\n"
            "            if p.exists():\n"
            "                files.append(p)\n"
            "    records = []\n"
            "    for path in files:\n"
            "        if path.suffix == '.jsonl':\n"
            "            with path.open(encoding='utf-8') as f:\n"
            "                for line in f:\n"
            "                    line = line.strip()\n"
            "                    if not line:\n"
            "                        continue\n"
            "                    rec = json.loads(line)\n"
            "                    text = rec.get('text') or ' '.join(\n"
            "                        x for x in (rec.get('title'), rec.get('description')) if x\n"
            "                    )\n"
            "                    label = rec.get('label', rec.get('labels'))\n"
            "                    if text is None or label is None:\n"
            "                        continue\n"
            "                    records.append({'text': str(text), 'label': int(label)})\n"
            "                    if len(records) >= max_samples:\n"
            "                        return records\n"
            "        elif path.suffix == '.csv':\n"
            "            with path.open(newline='', encoding='utf-8') as f:\n"
            "                for row in csv.DictReader(f):\n"
            "                    text = row.get('text') or row.get('sentence') or row.get('content')\n"
            "                    label = row.get('label') or row.get('labels')\n"
            "                    if text is None or label is None:\n"
            "                        continue\n"
            "                    records.append({'text': str(text), 'label': int(label)})\n"
            "                    if len(records) >= max_samples:\n"
            "                        return records\n"
            "    return records\n"
        )
        patched = code
        if "_edge_load_text_eval_samples" not in patched:
            patched = WorkspaceManager._insert_after_module_header(patched, helper + "\n")
        patched = re.sub(
            r"^[ \t]*from\s+datasets\s+import\s+load_dataset[^\n]*\n",
            "",
            patched,
            flags=re.MULTILINE,
        )
        patched = re.sub(
            r"^[ \t]*import\s+datasets[^\n]*\n",
            "",
            patched,
            flags=re.MULTILINE,
        )

        def _replace_load_dataset(match: re.Match) -> str:
            var = match.group(1)
            limit = 200
            lim_m = re.search(r"\[:(\d+)\]", match.group(2))
            if lim_m:
                limit = int(lim_m.group(1))
            return f"{var} = _EdgeTextDS(_edge_load_text_eval_samples({limit}))"

        patched = re.sub(
            r"(\w+)\s*=\s*(load_dataset\([^)]*\))",
            _replace_load_dataset,
            patched,
            count=1,
        )
        return patched

    @staticmethod
    def _cap_train_epochs(code: str) -> str:
        """Cap train epochs in generated scripts for fast development verification."""
        if not code:
            return code
        cap = int(getattr(settings, "MAX_TRAIN_EPOCHS", 0) or 0)
        if cap <= 0:
            return code
        patched = code

        def _cap_num_literal(m: re.Match) -> str:
            key = m.group(1)
            value = int(m.group(2))
            if value <= cap:
                return m.group(0)
            return f"{key}{cap}"

        # epochs = 4 / EPOCHS=10
        patched = re.sub(
            r"(\b(?:epochs|num_epochs|n_epochs)\s*=\s*)(\d+)\b",
            _cap_num_literal,
            patched,
            flags=re.IGNORECASE,
        )
        # TrainingArguments(... num_train_epochs=4 ...)
        patched = re.sub(
            r"(\bnum_train_epochs\s*=\s*)(\d+)\b",
            _cap_num_literal,
            patched,
            flags=re.IGNORECASE,
        )
        patched = re.sub(
            r"(\badd_argument\(\s*['\"]--(?:epochs|num-epochs|n-epochs)['\"][^\n)]*?"
            r"\bdefault\s*=\s*)(\d+)\b",
            _cap_num_literal,
            patched,
            flags=re.IGNORECASE,
        )
        # epochs = 2 if args.probe == "quality" else 60
        def _cap_ternary_full_branch(m: re.Match) -> str:
            value = int(m.group(3))
            if value <= cap:
                return m.group(0)
            return f"{m.group(1)}{m.group(2)}{cap}"

        patched = re.sub(
            r"(\b(?:epochs|num_epochs|n_epochs)\s*=\s*)([^\n#]+?\belse\s+)(\d+)\b",
            _cap_ternary_full_branch,
            patched,
            flags=re.IGNORECASE,
        )
        return patched

    @staticmethod
    def _insert_after_module_header(code: str, block: str) -> str:
        """Insert ``block`` after shebang, docstring, and required future imports."""
        lines = code.splitlines(keepends=True)
        if not lines:
            return block
        i = 0
        if lines[i].startswith("#!"):
            i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i < len(lines):
            stripped = lines[i].lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                delim = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(delim) >= 2:
                    i += 1
                else:
                    i += 1
                    while i < len(lines) and delim not in lines[i]:
                        i += 1
                    if i < len(lines):
                        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        while i < len(lines) and lines[i].lstrip().startswith("from __future__ import "):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
        return "".join(lines[:i]) + block + "".join(lines[i:])

    @staticmethod
    def _normalize_infer_model_path(code: str, export_format: str) -> str:
        """Rewrite model-specific filenames in infer.py to outputs/best.{format}.

        LLMs frequently generate model paths like 'yolov8n.onnx', 'yolo11n.pt',
        'outputs/yolo11n_fp16/weights/best.onnx', etc. These never exist on the
        edge device — only outputs/best.{ext} does.
        """
        if not code:
            return code
        if "edgecraft_hash_sgd_v1" in code or "hash-sgd-numpy" in code:
            return code
        fmt = export_format.lower().strip(".")
        canonical = f"outputs/best.{fmt}"
        patched = code

        # Pattern 1: model-specific filenames (bare or with path prefix)
        # e.g. "yolov8n.onnx", 'yolo11s.pt', "./yolov5n6.engine"
        patched = re.sub(
            r"""(['"])(?:[./]*)?(?:yolo(?:v?\d+[a-z]?\d*[a-z]?)|"""
            r"""efficientnet[a-z0-9_]*|mobilenet[a-z0-9_]*|"""
            r"""resnet\d+|shufflenet[a-z0-9_]*)"""
            r"""\.(?:onnx|pt|engine|tflite|torchscript)\1""",
            rf"\1{canonical}\1",
            patched,
            flags=re.IGNORECASE,
        )

        # Pattern 2: nested paths under outputs/ with wrong subdirs
        # e.g. "outputs/yolo11n_fp16/weights/best.onnx" → "outputs/best.onnx"
        patched = re.sub(
            r"""(['"])outputs/[^'"]*?/(?:weights/)?best\.(\w+)\1""",
            rf"\1outputs/best.{fmt}\1",
            patched,
        )

        # Pattern 3: bare "best.onnx" without the outputs/ prefix
        patched = re.sub(
            r"""(['"])(?!\.\./|outputs/)best\.(onnx|pt|engine|tflite)\1""",
            rf"\1outputs/best.{fmt}\1",
            patched,
        )

        # Pattern 3b: canonical output path but wrong extension for the
        # selected deploy target, e.g. outputs/best.engine after a dev run
        # forces export_format=onnx.
        patched = re.sub(
            r"""(['"])outputs/best\.(onnx|pt|engine|tflite)\1""",
            rf"\1outputs/best.{fmt}\1",
            patched,
        )

        # Pattern 4: hardcoded providers list that doesn't check availability.
        # Replace InferenceSession(..., providers=[...hardcoded...]) with safe fallback.
        patched = re.sub(
            r'ort\.InferenceSession\s*\(([^,)]+),\s*providers\s*=\s*\[[^\]]+\]\s*\)',
            r'ort.InferenceSession(\1, providers=[p for p in ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"] if p in ort.get_available_providers()])',
            patched,
        )

        return patched

    @staticmethod
    def _replace_real_image_with_dummy(code: str) -> str:
        """Replace image file loading in infer.py with synthetic dummy input.

        LLMs often write cv2.imread("data/sample.jpg"), Image.open("test.png"),
        or img_path = "data/sample.jpg" — these files don't exist on the edge
        device.  Replace with numpy random arrays.
        """
        if not code:
            return code

        _IMG_EXTS = r"\.(?:jpg|jpeg|png|bmp|tif|tiff|webp)"
        _DUMMY = "np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)"

        patched = code
        changed = False

        # Pattern 1: cv2.imread("anything.jpg")
        new = re.sub(
            r"""cv2\.imread\s*\(\s*['"][^'"]+['"]\s*\)""",
            _DUMMY, patched,
        )
        if new != patched:
            patched, changed = new, True

        # Pattern 2: Image.open("anything.jpg")
        new = re.sub(
            r"""Image\.open\s*\(\s*['"][^'"]+['"]\s*\)""",
            f"Image.fromarray({_DUMMY})", patched,
        )
        if new != patched:
            patched, changed = new, True

        # Pattern 3: variable assignment with image-path string literal
        # e.g. img_path = "data/sample.jpg"  or  IMAGE = 'test.png'
        new = re.sub(
            r"""^(\s*\w+\s*=\s*)(['"])[^'"]*?""" + _IMG_EXTS + r"""\2""",
            rf"\g<1>{_DUMMY}",
            patched,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if new != patched:
            patched, changed = new, True

        # Pattern 4: remove assert statements that check image path existence
        # e.g. assert Path(img_path).exists(), "..."
        patched = re.sub(
            r"""^[ \t]*assert\s+Path\s*\(\s*\w+\s*\)\.exists\(\).*$""",
            "",
            patched,
            flags=re.MULTILINE,
        )

        if not changed:
            return code

        if "np.random" in patched and "import numpy" not in patched:
            patched = WorkspaceManager._insert_after_module_header(
                patched, "import numpy as np\n"
            )

        return patched
