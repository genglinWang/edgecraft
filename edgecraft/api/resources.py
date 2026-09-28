"""Server-side resolution of tenant-owned dataset and credential handles."""
from __future__ import annotations

import os
from pathlib import Path

from edgecraft.agent.workspace.manager import (
    tenant_workspace_key,
    validate_workspace_component,
)


class TenantResourceError(ValueError):
    """A supplied resource handle is absent or outside its tenant root."""


def _relative_handle(handle: str, *, label: str) -> Path:
    text = str(handle or "").strip()
    if not text or len(text) > 240 or "\\" in text:
        raise TenantResourceError(f"invalid {label}")
    path = Path(text)
    if path.is_absolute() or not path.parts:
        raise TenantResourceError(f"invalid {label}")
    for part in path.parts:
        try:
            validate_workspace_component(part, label=label)
        except ValueError as exc:
            raise TenantResourceError(f"invalid {label}") from exc
    return path


def _resolve(
    root: str | Path,
    tenant_id: str,
    handle: str,
    *,
    label: str,
    require_file: bool,
) -> Path:
    relative = _relative_handle(handle, label=label)
    tenant_root = (
        Path(root).expanduser() / tenant_workspace_key(tenant_id)
    ).resolve()
    candidate = (tenant_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(tenant_root)
    except ValueError as exc:
        raise TenantResourceError(f"invalid {label}") from exc
    if require_file and not candidate.is_file():
        raise TenantResourceError(f"unknown {label}")
    if not require_file and not candidate.exists():
        raise TenantResourceError(f"unknown {label}")
    return candidate


def resolve_dataset_handle(root: str | Path, tenant_id: str, handle: str) -> Path:
    """Resolve a dataset handle beneath the authenticated tenant namespace."""
    return _resolve(
        root,
        tenant_id,
        handle,
        label="dataset_handle",
        require_file=False,
    )


def resolve_credential_handle(root: str | Path, tenant_id: str, handle: str) -> Path:
    """Resolve a least-privilege SSH identity managed by the service operator."""
    path = _resolve(
        root,
        tenant_id,
        handle,
        label="credential_handle",
        require_file=True,
    )
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise TenantResourceError("credential_handle permissions must be owner-only")
    return path
