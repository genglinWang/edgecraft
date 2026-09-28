"""Temporary overrides of the global ``settings`` object for a single CLI command run.

Use this when Click options should shadow the process-local EdgeCraft settings object.
Add new synth-related overrides in ``build_synth_settings_overrides`` (one place to extend).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping


@contextmanager
def temporary_settings_attrs(settings_obj: Any, updates: Mapping[str, Any]) -> Iterator[None]:
    """Apply ``updates`` as attributes on ``settings_obj``; restore previous values on exit."""
    keys = tuple(updates.keys())
    previous = {k: getattr(settings_obj, k) for k in keys}
    try:
        for k, v in updates.items():
            setattr(settings_obj, k, v)
        yield
    finally:
        for k, val in previous.items():
            setattr(settings_obj, k, val)


def build_synth_settings_overrides(
    *,
    debugger_enabled: str | None,
    debugger_retries: int | None,
) -> Dict[str, Any]:
    """Map ``edgecraft synth`` CLI options to ``Settings`` attribute names.

    Raises:
        ValueError: Invalid combination (caller maps to user-facing message / Abort).
    """
    out: Dict[str, Any] = {}
    if debugger_enabled is not None:
        out["DEBUGGER_ENABLED"] = debugger_enabled.lower() == "on"
    if debugger_retries is not None:
        if debugger_retries < 0:
            raise ValueError("--debugger-retries must be >= 0")
        out["DEBUGGER_MAX_RETRIES"] = debugger_retries
    return out


def describe_settings_overrides(updates: Mapping[str, Any]) -> str:
    """Single-line summary for console (dim)."""
    if not updates:
        return ""
    parts = [f"{k}={updates[k]!r}" for k in sorted(updates.keys())]
    return "CLI override active: " + ", ".join(parts)
