"""Generic crafting contracts shared across the EdgeCraft agent.

These models intentionally describe framework-level state, not dataset-specific
logic.  They let the synthesis loop preserve useful partial progress when an
optimization target such as ONNX or TensorRT is blocked.
"""
from __future__ import annotations

import ast
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CraftingProgress(str, Enum):
    """Layered progress state for a trial."""

    PENDING = "pending"
    SCRIPT_GENERATED = "script_generated"
    TRAIN_SUCCEEDED = "train_succeeded"
    EXPORT_SUCCEEDED = "export_succeeded"
    EDGE_RUNNABLE = "edge_runnable"
    EDGE_OPTIMIZED = "edge_optimized"
    CONSTRAINT_SATISFIED = "constraint_satisfied"


class ArtifactRecord(BaseModel):
    """One produced artifact or sidecar file."""

    path: str
    kind: str = Field("", description="pt, onnx, onnx_external_data, engine, torchscript, etc.")
    role: str = Field("candidate", description="primary, fallback, sidecar, candidate")
    runtime: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    required_on_edge: bool = False
    provenance: Optional[str] = None


class RuntimeAttempt(BaseModel):
    """One attempted runtime execution/export path."""

    runtime: str
    artifact_path: Optional[str] = None
    provider: Optional[str] = None
    status: str = Field("unknown", description="success, failed, skipped")
    error: Optional[str] = None
    fallback_reason: Optional[str] = None


class RuntimeReport(BaseModel):
    """Runtime negotiation and fallback details."""

    requested_runtime: Optional[str] = None
    runtime_used: Optional[str] = None
    runtime_provider: Optional[str] = None
    artifact_used: Optional[str] = None
    fallback_reason: Optional[str] = None
    attempts: List[RuntimeAttempt] = Field(default_factory=list)


class ArtifactContract(BaseModel):
    """Artifacts produced by a trial."""

    primary_artifact: Optional[str] = None
    fallback_artifacts: List[str] = Field(default_factory=list)
    artifacts: Dict[str, ArtifactRecord] = Field(default_factory=dict)
    export_attempts: List[RuntimeAttempt] = Field(default_factory=list)


def inspect_loader_train_contract(loader_code: str, train_code: str) -> Dict[str, Any]:
    """Return source-declared loader/train interface evidence.

    Dynamic interfaces remain unknown and are left to the measured component
    probe. This function only records direct return expressions and bindings.
    """
    if not loader_code.strip() or not train_code.strip():
        return {}
    try:
        loader_tree = ast.parse(loader_code)
        train_tree = ast.parse(train_code)
    except SyntaxError:
        return {}

    return_arities: set[int] = set()
    return_expressions: List[str] = []

    def collect_returns(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return
        if isinstance(node, ast.Return):
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List)) and not any(
                isinstance(item, ast.Starred) for item in value.elts
            ):
                return_arities.add(len(value.elts))
            elif isinstance(value, (ast.Dict, ast.Set)):
                return_arities.add(1)
            if value is not None:
                expression = ast.unparse(value)
                if expression not in return_expressions:
                    return_expressions.append(expression[:600])
            return
        for child in ast.iter_child_nodes(node):
            collect_returns(child)

    for node in loader_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "load_train_val":
            for statement in node.body:
                collect_returns(statement)

    unpack_arities: set[int] = set()
    direct_bindings: List[str] = []
    train_bindings: List[str] = []
    for node in ast.walk(train_tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name != "load_train_val":
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)) and not any(
                isinstance(item, ast.Starred) for item in target.elts
            ):
                unpack_arities.add(len(target.elts))
            else:
                direct_binding = ast.unparse(target)
                if direct_binding not in direct_bindings:
                    direct_bindings.append(direct_binding[:600])
            binding = ast.unparse(target)
            if binding not in train_bindings:
                train_bindings.append(binding[:600])

    return {
        "source": "static_code",
        "loader_return_arities": sorted(return_arities),
        "loader_return_expressions": return_expressions[:6],
        "train_unpack_arities": sorted(unpack_arities),
        "train_bindings": train_bindings[:8],
        "train_direct_bindings": direct_bindings[:8],
    }


def artifact_kind(path: str) -> str:
    """Return a stable artifact kind from a filename."""
    lower = (path or "").lower()
    name = lower.rsplit("/", 1)[-1]
    if name in {
        "dataset_contract.json",
        "failure.json",
        "diagnostic.json",
        "train_summary.json",
        "sample_observation.json",
    }:
        return "diagnostic"
    if lower.endswith(".onnx.data") or lower.endswith(".data"):
        return "onnx_external_data"
    if lower.endswith(".onnx"):
        return "onnx"
    if lower.endswith(".engine"):
        return "engine"
    if lower.endswith(".tflite"):
        return "tflite"
    if lower.endswith(".torchscript") or lower.endswith(".ts"):
        return "torchscript"
    if lower.endswith(".pt") or lower.endswith(".pth"):
        return "pt"
    return "unknown"
