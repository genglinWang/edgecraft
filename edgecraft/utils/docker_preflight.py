"""Remote Docker image preflight on the edge host (SSH).

Class A: local tag exists, arch matches host, container starts with production-like ``docker run``.
Class B: ``python3`` and required imports (no GPU / CUDA checks).

``docker run`` flags are sourced from ``edgecraft.tools.deploy.docker_opts`` so they stay aligned with
``EdgeRunner`` and ``OfflineProfiler``.
"""
from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

from edgecraft.models import ensure_registries_initialized
from edgecraft.config.device_manifest import load_declared_devices
from edgecraft.models.family_registry import DeviceRegistry
from edgecraft.models.specs import RuntimeId
from edgecraft.tools.deploy.docker_opts import (
    edge_runner_infer_docker_opts_bash,
    offline_profiler_prebuilt_docker_args_bash,
)
from edgecraft.utils.network import run_remote_command

Flavor = Literal["edge_runner", "offline_profiler"]

# Synth edge benchmark: typical stack for vision + ORT exports.
_SYNTH_IMPORT_PY = (
    "import json\n"
    "import importlib\n"
    "import shutil\n"
    "import sys\n"
    "import numpy\n"
    "import torch\n"
    "import onnxruntime\n"
    "try:\n"
    "    import tensorrt\n"
    "    trt_version = getattr(tensorrt, '__version__', 'available')\n"
    "except Exception as exc:\n"
    "    trt_version = f'unavailable:{exc.__class__.__name__}'\n"
    "probe_modules = {\n"
    "    'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'sklearn',\n"
    "    'pyarrow': 'pyarrow', 'joblib': 'joblib',\n"
    "    'torch': 'torch', 'torchvision': 'torchvision', 'torchaudio': 'torchaudio',\n"
    "    'onnx': 'onnx', 'onnxruntime': 'onnxruntime', 'tensorrt': 'tensorrt',\n"
    "    'datasets': 'datasets', 'PIL': 'PIL', 'cv2': 'cv2', 'yaml': 'yaml',\n"
    "    'ultralytics': 'ultralytics', 'timm': 'timm', 'librosa': 'librosa', 'soundfile': 'soundfile',\n"
    "    'transformers': 'transformers', 'tokenizers': 'tokenizers',\n"
    "    'tflite_runtime': 'tflite_runtime', 'psutil': 'psutil',\n"
    "}\n"
    "packages = {}\n"
    "for name, module in probe_modules.items():\n"
    "    try:\n"
    "        mod = importlib.import_module(module)\n"
    "        packages[name] = str(getattr(mod, '__version__', 'available'))\n"
    "    except Exception as exc:\n"
    "        packages[name] = f'unavailable:{exc.__class__.__name__}'\n"
    "disk = shutil.disk_usage('.')\n"
    "print(json.dumps({\n"
    "    'status': 'edgecraft_docker_preflight_ok',\n"
    "    'python': sys.version.split()[0],\n"
    "    'torch_version': getattr(torch, '__version__', ''),\n"
    "    'torch_cuda_available': bool(torch.cuda.is_available()),\n"
    "    'onnxruntime_version': getattr(onnxruntime, '__version__', ''),\n"
    "    'available_providers': list(onnxruntime.get_available_providers()),\n"
    "    'tensorrt_version': trt_version,\n"
    "    'python_packages': packages,\n"
    "    'disk_free_mb': int(disk.free / 1024 / 1024),\n"
    "}))\n"
)

_RUNTIME_TO_MODULE = {
    RuntimeId.PYTORCH: "torch",
    RuntimeId.ONNXRUNTIME: "onnxruntime",
    RuntimeId.TENSORRT: "tensorrt",
}


def _validate_image_ref(name: str) -> Optional[str]:
    if not name or not name.strip():
        return "Docker image name is empty."
    s = name.strip()
    if not re.match(r"^[a-zA-Z0-9._/@:-]+$", s):
        return (
            "Docker image name contains unsupported characters "
            "(use [a-zA-Z0-9._/@:-] only)."
        )
    return None


def _imports_python_for_profile(runtimes: Optional[Sequence[RuntimeId]]) -> str:
    """Build a small Python script for B-check imports."""
    if not runtimes:
        mods = ["torch", "onnxruntime", "tensorrt"]
    else:
        mods = []
        for r in runtimes:
            m = _RUNTIME_TO_MODULE.get(r)
            if m and m not in mods:
                mods.append(m)
        if not mods:
            mods = ["torch", "onnxruntime", "tensorrt"]
    lines = ["import sys"]
    for m in mods:
        lines.append(f"import {m}")
    lines.append("print('edgecraft_docker_preflight_ok')")
    return "\n".join(lines) + "\n"


def _bash_escape_double(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


def _build_remote_script_b64_py(
    image: str,
    flavor: Flavor,
    python_payload: str,
    docker_opts_literal: str,
    remote_tmp_dir: str = "",
) -> str:
    """Remote bash: A1/A2/A3 + B via ``python3 -c`` with base64-wrapped payload."""
    py_b64 = base64.b64encode(python_payload.encode("utf-8")).decode("ascii")
    inner_py_cmd = (
        "import base64; "
        f"exec(compile(base64.b64decode('{py_b64}'.encode('ascii')).decode('utf-8'), '<pf>', 'exec'))"
    )
    inner_escaped = _bash_escape_double(inner_py_cmd)

    img_b64 = base64.b64encode(image.encode("utf-8")).decode("ascii")
    tmp_root = remote_tmp_dir.strip()

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"IMAGE=$(printf '%s' '{img_b64}' | base64 -d)",
        "",
        "# --- A1: local tag only (no pull) ---",
        'if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then',
        '  echo "A1: docker image not found locally: $IMAGE"',
        "  exit 11",
        "fi",
        "",
        "# --- A2: architecture ---",
        'IMG_ARCH=$(docker image inspect "$IMAGE" --format "{{.Architecture}}" 2>/dev/null || true)',
        'if [ -z "$IMG_ARCH" ]; then',
        '  echo "A2: could not read image Architecture from docker inspect"',
        "  exit 12",
        "fi",
        "norm_arch() {",
        '  case "$1" in aarch64|arm64) echo arm64 ;; x86_64|amd64) echo amd64 ;; *) echo "$1" ;; esac',
        "}",
        'H_ARCH=$(norm_arch "$(uname -m)")',
        'I_ARCH=$(norm_arch "$IMG_ARCH")',
        'if [ "$H_ARCH" != "$I_ARCH" ]; then',
        '  echo "A2: host arch ${H_ARCH} vs image arch ${I_ARCH} (raw image=${IMG_ARCH})"',
        "  exit 13",
        "fi",
        "",
        "# --- A3 + B: docker run (same flags as production) ---",
        f"TMP_ROOT={shlex.quote(tmp_root)}" if tmp_root else 'TMP_ROOT="${TMPDIR:-/tmp}"',
        'mkdir -p "$TMP_ROOT"',
        'WORKDIR=$(mktemp -d "$TMP_ROOT/edgecraft-preflight.XXXXXX")',
        'trap "rm -rf \\"$WORKDIR\\"" EXIT',
        'cd "$WORKDIR"',
    ]

    if flavor == "edge_runner":
        lines.append(f'DOCKER_OPTS="{docker_opts_literal}"')
        lines.append(f'docker run $DOCKER_OPTS "$IMAGE" python3 -c "{inner_escaped}"')
    else:
        lines.append(f'DOCKER_ARGS="{docker_opts_literal}"')
        lines.append(f'docker run $DOCKER_ARGS "$IMAGE" python3 -c "{inner_escaped}"')

    return "\n".join(lines) + "\n"


@dataclass
class DockerPreflightResult:
    ok: bool
    message: str
    details: Dict[str, Any] = None


def _extract_json_details(text: str) -> Dict[str, Any]:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def run_docker_image_preflight(
    *,
    device_ip: str,
    ssh_key: Optional[str],
    docker_image: str,
    device_id: str,
    flavor: Flavor,
    profile_runtime_ids: Optional[Sequence[RuntimeId]] = None,
    total_timeout_sec: int = 60,
    attempts: int = 2,
) -> DockerPreflightResult:
    """Run class A + B checks on the edge host over SSH."""
    bad = _validate_image_ref(docker_image)
    if bad:
        return DockerPreflightResult(False, bad)

    ensure_registries_initialized()
    spec = DeviceRegistry.get(device_id)
    has_gpu = spec.has_gpu if spec else False

    if flavor == "edge_runner":
        opts = edge_runner_infer_docker_opts_bash(has_gpu)
        py = _SYNTH_IMPORT_PY
    else:
        opts = offline_profiler_prebuilt_docker_args_bash(has_gpu)
        py = _imports_python_for_profile(profile_runtime_ids)

    script = _build_remote_script_b64_py(
        docker_image.strip(),
        flavor,
        py,
        opts,
        str((load_declared_devices().get(device_id) or {}).get("remote_base") or ""),
    )

    b64_script = base64.b64encode(script.encode("utf-8")).decode("ascii")
    remote_cmd = f"printf '%s' '{b64_script}' | base64 -d | bash"
    max_attempts = max(1, int(attempts or 1))
    last_msg = ""
    for attempt in range(1, max_attempts + 1):
        try:
            code, out, err = run_remote_command(
                device_ip,
                remote_cmd,
                ssh_key=ssh_key,
                timeout=total_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            last_msg = (
                f"docker preflight timed out after {total_timeout_sec}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            if attempt < max_attempts:
                time.sleep(2)
                continue
            return DockerPreflightResult(False, last_msg)

        combined = (out or "") + ("\n" + err if err else "")
        if code == 0:
            return DockerPreflightResult(
                True,
                combined.strip() or "docker preflight ok",
                _extract_json_details(combined),
            )
        last_msg = combined.strip() or f"exit {code}"
        # Non-zero semantic failures (missing image, arch mismatch, import error)
        # are deterministic; retrying only hides the real contract violation.
        return DockerPreflightResult(False, last_msg)

    return DockerPreflightResult(False, last_msg or "docker preflight failed")
