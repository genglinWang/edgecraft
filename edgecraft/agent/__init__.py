"""Public agent API without eagerly importing the execution graph."""
from importlib import import_module
from typing import Any

__all__ = ["create_agent_graph", "compile_agent", "run_agent", "AgentState"]


def __getattr__(name: str) -> Any:
    if name == "AgentState":
        return getattr(import_module(".state", __name__), name)
    if name in {"create_agent_graph", "compile_agent", "run_agent"}:
        return getattr(import_module(".graph", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
