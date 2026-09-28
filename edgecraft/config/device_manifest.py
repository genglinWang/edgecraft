"""Read declared device/runtime facts from the canonical project manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_DEVICE_MANIFEST = Path(__file__).with_name("device_runtime_docker.json")

_RUNTIME_ALIASES = {
    "litert": "tflite",
}

_RUNTIME_FORMATS = {
    "pytorch": "pt",
    "onnxruntime": "onnx",
    "tensorrt": "engine",
    "tflite": "tflite",
}


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


def load_declared_devices(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the declared resource pool; live preflight remains authoritative."""
    configured = os.getenv("EDGECRAFT_DEVICE_MANIFEST", "").strip()
    manifest = path or (Path(configured).expanduser() if configured else DEFAULT_DEVICE_MANIFEST)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    devices = payload.get("devices")
    if not isinstance(devices, dict):
        raise ValueError(f"device manifest has no devices mapping: {manifest}")
    return {
        str(device_id): _expand_environment(facts)
        for device_id, facts in devices.items()
        if isinstance(facts, Mapping)
    }


def normalized_runtimes(facts: Mapping[str, Any]) -> list[str]:
    """Normalize public runtime names without claiming they passed live probing."""
    result: list[str] = []
    for value in facts.get("runtimes") or []:
        runtime = _RUNTIME_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
        if runtime and runtime not in result:
            result.append(runtime)
    return result


def supported_formats(facts: Mapping[str, Any]) -> list[str]:
    """Project declared runtimes onto artifact formats consumed by EdgeRunner."""
    return [
        _RUNTIME_FORMATS[runtime]
        for runtime in normalized_runtimes(facts)
        if runtime in _RUNTIME_FORMATS
    ]
