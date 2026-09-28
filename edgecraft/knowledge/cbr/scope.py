"""Pure tenant namespace helpers for private case memory."""
from __future__ import annotations

import hashlib
from typing import Optional

from edgecraft.config.settings import settings


def case_store_namespace(
    tenant_id: str,
    *,
    collection_name: str = "edgecraft_cases",
    scope: Optional[str] = None,
) -> str:
    """Return an opaque per-tenant CBR namespace."""
    selected_scope = str(scope or settings.CBR_SCOPE or "tenant").strip().lower()
    if selected_scope != "tenant":
        raise ValueError("CBR is available only with EDGECRAFT_CBR_SCOPE=tenant")
    tenant = str(tenant_id or "default").strip()
    if not tenant:
        raise ValueError("tenant_id is required for CBR")
    token = hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:16]
    return f"{collection_name}_tenant_{token}"
