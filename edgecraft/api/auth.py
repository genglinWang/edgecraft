"""Small tenant authorization boundary for the research control plane.

The paper mechanism does not depend on a particular identity provider.  This
module therefore accepts a reviewer/operator supplied token-to-tenant mapping
and exposes no default credential.  With no mapping, the API is local-only
single-operator mode (the CLI binds it to loopback by default).
"""
from __future__ import annotations

import hmac
import json
from typing import Dict, Optional

from edgecraft.agent.workspace.manager import validate_workspace_component


class TenantAuthenticationError(ValueError):
    pass


def parse_tenant_tokens(raw: str) -> Dict[str, str]:
    """Validate a JSON object mapping opaque bearer tokens to tenant IDs."""
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TenantAuthenticationError(
            "EDGECRAFT_API_TENANT_TOKENS must be a JSON object"
        ) from exc
    if not isinstance(payload, dict) or not payload:
        raise TenantAuthenticationError(
            "EDGECRAFT_API_TENANT_TOKENS must be a non-empty JSON object"
        )
    parsed: Dict[str, str] = {}
    for token, tenant in payload.items():
        token_text = str(token or "")
        if len(token_text) < 16:
            raise TenantAuthenticationError("API bearer tokens must be at least 16 characters")
        parsed[token_text] = validate_workspace_component(
            str(tenant or ""),
            label="tenant_id",
        )
    return parsed


def authenticate_tenant(
    authorization: Optional[str],
    *,
    token_config: str,
) -> Optional[str]:
    """Return the authenticated tenant, or None in local single-operator mode."""
    mappings = parse_tenant_tokens(token_config)
    if not mappings:
        return None
    scheme, separator, supplied = str(authorization or "").partition(" ")
    if not separator or scheme.lower() != "bearer" or not supplied:
        raise TenantAuthenticationError("a bearer token is required")
    for expected, tenant in mappings.items():
        if hmac.compare_digest(supplied, expected):
            return tenant
    raise TenantAuthenticationError("invalid bearer token")
