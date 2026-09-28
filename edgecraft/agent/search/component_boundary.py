"""File-level facts for one coherent parent-to-child mutation boundary."""

from __future__ import annotations

from collections.abc import Iterable


COMPONENT_FILES = ("loader.py", "train.py", "infer.py", "config/data.yaml")


def declared_component_files(components: Iterable[str]) -> set[str]:
    """Return files explicitly named by free-form component declarations."""
    text = " ".join(str(component).lower() for component in components)
    return {name for name in COMPONENT_FILES if name in text}
