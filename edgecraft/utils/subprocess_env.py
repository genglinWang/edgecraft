"""Minimal environment for generated candidate subprocesses."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional


_PASSTHROUGH = {
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "LANG",
    "LC_ALL",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
}


def candidate_subprocess_env(
    workspace: str | Path,
    overrides: Optional[Mapping[str, object]] = None,
) -> dict[str, str]:
    """Build a runtime-only environment without controller credentials.

    Controller API/LLM tokens, SSH agent state, and host-level model caches are
    intentionally absent. Trusted controller code may add explicit per-stage
    facts through ``overrides``.
    """
    root = Path(workspace).resolve()
    private_home = root / ".edgecraft-home"
    private_tmp = root / ".edgecraft-tmp"
    model_cache = root / "weights"
    for path in (private_home, private_tmp, model_cache):
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _PASSTHROUGH or key.startswith("LC_")
    }
    env.update(
        {
            "HOME": str(private_home),
            "TMPDIR": str(private_tmp),
            "TMP": str(private_tmp),
            "TEMP": str(private_tmp),
            "HF_HOME": str(model_cache / "huggingface"),
            "TORCH_HOME": str(model_cache / "torch"),
            "ULTRALYTICS_CONFIG_DIR": str(private_home / "ultralytics"),
            "EDGECRAFT_MODEL_CACHE_DIR": str(model_cache),
        }
    )
    env.update({str(key): str(value) for key, value in (overrides or {}).items()})
    return env
