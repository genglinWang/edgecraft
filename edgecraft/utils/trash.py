"""Single non-destructive filesystem cleanup boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from edgecraft.config.settings import edgecraft_env


def move_to_trash(path: str | Path) -> bool:
    """Move an existing path to trash; never fall back to permanent deletion."""
    target = Path(path)
    if not target.exists() and not target.is_symlink():
        return False
    command = edgecraft_env("TRASH_COMMAND") or shutil.which("trash")
    if not command:
        raise RuntimeError(f"trash command unavailable; refusing to delete {target}")
    subprocess.run([command, str(target)], check=True)
    return True
