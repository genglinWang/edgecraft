"""Lazy node exports for the EdgeCraft iterative search graph."""
from importlib import import_module
from typing import Any

__all__ = [
    "proposal_generator_node",
    "pipeline_executor_node",
    "scorer_node",
    "reflector_node",
    # v1 compat
    "planner_node",
    "executor_node",
]


_NODE_MODULES = {
    "proposal_generator_node": ".proposal_generator",
    "pipeline_executor_node": ".pipeline_executor",
    "scorer_node": ".reflector",
    "reflector_node": ".reflector",
    # Compatibility imports; the v2 graph does not use these nodes.
    "planner_node": ".planner",
    "executor_node": ".executor",
}


def __getattr__(name: str) -> Any:
    module_name = _NODE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(import_module(module_name, __name__), name)
    except Exception:
        if name in {"planner_node", "executor_node"}:
            return None
        raise
