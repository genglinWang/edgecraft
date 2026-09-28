"""EdgeCraft API package."""
from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    """Load FastAPI only when an ASGI entry point is requested."""
    if name in {"app", "create_app"}:
        from edgecraft.api.server import app, create_app

        return {"app": app, "create_app": create_app}[name]
    raise AttributeError(name)
