"""Verification evidence for multi-fidelity EdgeCraft trials.

These models are deliberately small data containers.  They do not schedule,
score, repair, or override the synthesis loop; they only make verification
evidence explicit so the tree and LLM prompts can reason from the same facts.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


EFFICIENCY_MEASUREMENT_CONTRACT_VERSION = "edge_efficiency_v3"
EFFICIENCY_PROBE_WARMUP = 5
EFFICIENCY_PROBE_MIN_WARMUP_SECONDS = 5.0
EFFICIENCY_PROBE_REPETITIONS = 5
EFFICIENCY_PROBE_MIN_MEASURE_SECONDS = 5.0
EFFICIENCY_PROBE_SESSIONS = 3


class MetricEstimate(BaseModel):
    """One measured or estimated metric with optional uncertainty."""

    value: Optional[float] = None
    sigma: Optional[float] = None
    source: str = ""
    unit: str = ""


class ResourceCost(BaseModel):
    """Actual resources spent to acquire one piece of evidence."""

    gpu_s: float = 0.0
    compile_s: float = 0.0
    device_s: float = 0.0


class Evidence(BaseModel):
    """One factual observation with an explicit decision authority."""

    id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    probe_id: str
    quantity: str
    fidelity: Literal["static", "measured_congruent", "proxy"]
    outcome: Literal["pass", "fail", "unknown"] = "unknown"
    value: Optional[float] = None
    sigma: Optional[float] = None
    unit: str = ""
    artifact_fingerprint: Dict[str, Any] = Field(default_factory=dict)
    environment_fingerprint: str = ""
    protocol: Dict[str, Any] = Field(default_factory=dict)
    resource_cost: ResourceCost = Field(default_factory=ResourceCost)


class VerificationDecision(BaseModel):
    action: Literal["prune", "escalate", "accept"] = "escalate"
    basis: List[str] = Field(default_factory=list)
    reason: str = ""
    next_probe: str = ""


class VerificationReport(BaseModel):
    """Evidence collected at one verification fidelity level."""

    level: str = Field("L2", description="L0 | L1 | L2")
    status: str = Field("unknown", description="pass | fail | skipped | unknown")
    evidence_source: str = Field("", description="static | measured | simulated")
    decision: str = Field("none", description="none | prune | escalate | accept")
    admission_status: str = Field(
        "admitted",
        description="admitted | rejected; contract-invalid candidates are not verifier prunes",
    )
    metric_estimates: Dict[str, MetricEstimate] = Field(default_factory=dict)
    evidence: List[Evidence] = Field(default_factory=list)
    decision_basis: List[str] = Field(default_factory=list)
    next_probe: str = ""
    artifact_fingerprint: Dict[str, Any] = Field(default_factory=dict)
    artifact_status: str = ""
    runtime_status: str = ""
    errors: List[str] = Field(default_factory=list)
    promoted_from: str = ""
    source: str = ""
    notes: str = ""


class GapSlackBrief(BaseModel):
    """Constraint alignment evidence for one user metric."""

    metric: str
    comparison: str
    target: float
    value: Optional[float] = None
    sigma: Optional[float] = None
    known: bool = False
    signed_gap: Optional[float] = None
    gap: Optional[float] = None
    slack: Optional[float] = None
    source: str = ""
    evidence_id: str = ""
    fidelity: str = ""
    normalized_gap_low: Optional[float] = None
    normalized_gap_high: Optional[float] = None
    normalized_slack_low: Optional[float] = None
    objective_delta: Optional[float] = None

    @classmethod
    def from_constraint(
        cls,
        *,
        metric: str,
        comparison: str,
        target: float,
        value: Optional[float],
        sigma: Optional[float] = None,
        source: str = "",
        evidence_id: str = "",
        fidelity: str = "",
        calibration_error: float = 0.0,
        kappa: float = 2.0,
        objective_delta: Optional[float] = None,
    ) -> "GapSlackBrief":
        known = value is not None
        gap: Optional[float] = None
        slack: Optional[float] = None
        signed_gap: Optional[float] = None
        if known:
            if comparison == "lte":
                signed_gap = float(value) - float(target)
                gap = max(0.0, signed_gap)
                slack = max(0.0, -signed_gap)
            elif comparison == "gte":
                signed_gap = float(target) - float(value)
                gap = max(0.0, signed_gap)
                slack = max(0.0, -signed_gap)
            elif comparison == "eq":
                signed_gap = abs(float(value) - float(target))
                gap = signed_gap
                slack = 0.0 if gap else abs(float(target))
        gap_low: Optional[float] = None
        gap_high: Optional[float] = None
        slack_low: Optional[float] = None
        if known:
            scale = max(abs(float(target)), 1e-12)
            uncertainty = kappa * abs(float(sigma or 0.0)) + abs(float(calibration_error or 0.0))
            if comparison == "lte":
                signed = (float(value) - float(target)) / scale
            elif comparison == "gte":
                signed = (float(target) - float(value)) / scale
            else:
                signed = abs(float(value) - float(target)) / scale
            radius = uncertainty / scale
            gap_low = max(0.0, signed - radius)
            gap_high = max(0.0, signed + radius)
            slack_low = max(0.0, -signed - radius)
        return cls(
            metric=metric,
            comparison=comparison,
            target=float(target),
            value=float(value) if value is not None else None,
            sigma=sigma,
            known=known,
            signed_gap=signed_gap,
            gap=gap,
            slack=slack,
            source=source,
            evidence_id=evidence_id,
            fidelity=fidelity,
            normalized_gap_low=gap_low,
            normalized_gap_high=gap_high,
            normalized_slack_low=slack_low,
            objective_delta=objective_delta,
        )


class ArtifactFingerprint(BaseModel):
    """Small graph identity for L1/L2 congruence checks."""

    artifact_format: str = ""
    artifact_hash: str = ""
    graph_hash: str = ""
    op_set: List[str] = Field(default_factory=list)
    op_signatures: List[str] = Field(default_factory=list)
    subgraph_signatures: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    subgraph_signature_hashes: List[str] = Field(default_factory=list)
    param_count: Optional[int] = None
    param_shapes: List[List[int]] = Field(default_factory=list)
    input_shapes: List[List[Any]] = Field(default_factory=list)
    input_specs: List[Dict[str, Any]] = Field(default_factory=list)
    output_specs: List[Dict[str, Any]] = Field(default_factory=list)
    source_artifact_hash: str = ""
    artifact_properties: Dict[str, Any] = Field(default_factory=dict)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_runtime_name(runtime: str) -> str:
    """Normalize artifact/runtime aliases to the backend that executes them."""
    name = str(runtime or "").strip().lower().replace("-", "_")
    return {
        "onnx": "onnxruntime",
        "ort": "onnxruntime",
        "pt": "torch",
        "pytorch": "torch",
        "engine": "tensorrt",
        "trt": "tensorrt",
        "tflite": "litert",
    }.get(name, name)


def _runtime_package_version(cfg: Dict[str, Any], runtime: str) -> str:
    """Select the package version that identifies one concrete runtime."""
    canonical = canonical_runtime_name(runtime)
    direct = {
        "tensorrt": cfg.get("trt_version"),
        "onnxruntime": cfg.get("onnxruntime_version"),
        "torch": cfg.get("torch_version"),
    }.get(canonical)
    if direct:
        return str(direct)
    if canonical != "litert":
        return ""
    packages = dict(cfg.get("python_packages") or {})
    grouped = cfg.get("runtime_python_packages") or {}
    for group_name in ("litert", "tflite", "default"):
        packages.update(dict(grouped.get(group_name) or {}))
    for package in ("ai_edge_litert", "tflite_runtime", "tensorflow"):
        value = packages.get(package)
        if value and not str(value).startswith("unavailable:"):
            return str(value)
    return ""


def runtime_package_version(runtime_config: Any, runtime: str) -> str:
    """Return the concrete package version for a probed runtime environment."""
    if hasattr(runtime_config, "model_dump"):
        cfg = runtime_config.model_dump(mode="json")
    elif isinstance(runtime_config, dict):
        cfg = dict(runtime_config)
    else:
        cfg = {}
    return _runtime_package_version(cfg, runtime)


def build_environment_fingerprint(
    *,
    device_id: str,
    runtime: str,
    runtime_config: Any = None,
    protocol: Optional[Dict[str, Any]] = None,
) -> str:
    """Hash physical/runtime identity, independently of probe protocol.

    Precision, input profiles, warm-up, and repetition counts describe how a
    measurement was acquired.  They belong in the calibration protocol
    fingerprint and must not prevent reuse of a verified compatibility fact
    within the same device/runtime environment.
    """
    _ = protocol  # Retained for callers using the previous public signature.
    if hasattr(runtime_config, "model_dump"):
        cfg = runtime_config.model_dump(mode="json")
    elif isinstance(runtime_config, dict):
        cfg = dict(runtime_config)
    else:
        cfg = {}
    stable_cfg = {
        key: cfg.get(key)
        for key in (
            "device_arch",
            "trt_version",
            "onnxruntime_version",
            "onnxruntime_providers",
            "torch_version",
            "l4t_version",
            "docker_image",
        )
        if cfg.get(key) not in (None, "", [])
    }
    runtime_version = runtime_package_version(cfg, runtime)
    if runtime_version:
        stable_cfg["runtime_package_version"] = runtime_version
    return _stable_hash(
        {
            "device_id": device_id,
            "runtime": canonical_runtime_name(runtime),
            "runtime_config": stable_cfg,
        }
    )


def _onnx_attribute_signature(attr: Any) -> Any:
    """Return ONNX attribute structure without embedding learned tensor data."""
    try:
        import onnx  # type: ignore

        value = onnx.helper.get_attribute_value(attr)
        if hasattr(value, "dims") and hasattr(value, "data_type"):
            return {"tensor_shape": list(value.dims), "dtype": int(value.data_type)}
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return [
                {"tensor_shape": list(v.dims), "dtype": int(v.data_type)}
                if hasattr(v, "dims") and hasattr(v, "data_type")
                else v
                for v in value
            ]
        return value
    except Exception:
        return str(attr)


def _fill_onnx_graph_fingerprint(path: Path, fp: ArtifactFingerprint) -> None:
    import onnx  # type: ignore

    model = onnx.load(str(path), load_external_data=False)
    initializer_map = {init.name: init for init in model.graph.initializer}
    initializers = set(initializer_map)
    value_map = {
        value.name: value
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    }
    opsets = {item.domain or "": int(item.version) for item in model.opset_import}
    fp.artifact_properties = {
        "ir_version": int(model.ir_version),
        "opset_imports": {
            str(domain or "ai.onnx"): int(version)
            for domain, version in opsets.items()
        },
    }

    def value_descriptor(name: str) -> Dict[str, Any]:
        if name in initializer_map:
            tensor = initializer_map[name]
            return {
                "kind": "parameter",
                "dtype": int(tensor.data_type),
                "shape": [int(dim) for dim in tensor.dims],
            }
        value = value_map.get(name)
        if value is None or not value.type.HasField("tensor_type"):
            return {"kind": "value", "dtype": None, "shape": []}
        tensor_type = value.type.tensor_type
        shape: List[Any] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(int(dim.dim_value))
            elif dim.dim_param:
                shape.append("?")
            else:
                shape.append("?")
        return {"kind": "value", "dtype": int(tensor_type.elem_type), "shape": shape}

    graph_inputs = [
        value.name
        for value in model.graph.input
        if value.name not in initializers
    ]
    canonical_values = {
        name: f"input:{index}"
        for index, name in enumerate(graph_inputs)
    }
    nodes: List[Dict[str, Any]] = []
    optional_input_slots: List[set[int]] = []
    for node_index, node in enumerate(model.graph.node):
        optional_slots: set[int] = set()
        try:
            schema = onnx.defs.get_schema(
                node.op_type,
                opsets.get(node.domain or ""),
                node.domain or "",
            )
            optional_slots = {
                index
                for index, parameter in enumerate(schema.inputs)
                if str(getattr(parameter, "option", "")).lower().endswith("optional")
            }
        except Exception:
            # Unknown/custom operators remain strict: no input is normalized.
            pass
        optional_input_slots.append(optional_slots)
        attrs = {
            attr.name: _onnx_attribute_signature(attr)
            for attr in sorted(node.attribute, key=lambda item: item.name)
        }
        canonical_inputs: List[Dict[str, Any]] = []
        for name in node.input:
            descriptor = value_descriptor(name)
            if name in initializers:
                source = "parameter"
            else:
                source = canonical_values.get(name, "external")
            canonical_inputs.append({"source": source, **descriptor})
        canonical_outputs = [
            {"slot": output_index, **value_descriptor(name)}
            for output_index, name in enumerate(node.output)
        ]
        nodes.append({
            "domain": node.domain or "",
            "op_type": node.op_type,
            "inputs": canonical_inputs,
            "outputs": canonical_outputs,
            "attributes": attrs,
        })
        for output_index, name in enumerate(node.output):
            canonical_values[name] = f"node:{node_index}:output:{output_index}"
        fp.op_signatures.append(
            f"{node.domain or 'ai.onnx'}::{node.op_type}:{_stable_hash(attrs)[:12]}"
        )
        subgraph = {
            "domain": node.domain or "",
            "op_type": node.op_type,
            "opset": opsets.get(node.domain or ""),
            "attributes_hash": _stable_hash(attrs),
            "inputs": [value_descriptor(name) for name in node.input if name],
            "outputs": [value_descriptor(name) for name in node.output if name],
        }
        fp.subgraph_signatures.append(subgraph)
        fp.subgraph_signature_hashes.append(_stable_hash(subgraph))
    fp.op_signatures = sorted(set(fp.op_signatures))
    fp.subgraph_signature_hashes = sorted(set(fp.subgraph_signature_hashes))
    fp.op_set = sorted({node["op_type"] for node in nodes})
    fp.param_shapes = sorted(
        [list(map(int, init.dims)) for init in model.graph.initializer],
        key=lambda shape: tuple(shape),
    )
    fp.param_count = int(sum(math.prod(shape or [1]) for shape in fp.param_shapes))
    shapes: List[List[Any]] = []
    input_specs: List[Dict[str, Any]] = []
    for value in model.graph.input:
        if value.name in initializers:
            continue
        dims: List[Any] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(int(dim.dim_value))
            elif dim.dim_param:
                dims.append(str(dim.dim_param))
            else:
                dims.append("?")
        if dims:
            shapes.append(dims)
        tensor_type = value.type.tensor_type
        try:
            dtype = onnx.TensorProto.DataType.Name(int(tensor_type.elem_type))
        except Exception:
            dtype = str(int(tensor_type.elem_type))
        input_specs.append({"name": value.name, "dtype": dtype, "shape": dims})
    fp.input_shapes = shapes
    fp.input_specs = input_specs
    output_specs: List[Dict[str, Any]] = []
    for value in model.graph.output:
        dims: List[Any] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(int(dim.dim_value))
            elif dim.dim_param:
                dims.append(str(dim.dim_param))
            else:
                dims.append("?")
        tensor_type = value.type.tensor_type
        try:
            dtype = onnx.TensorProto.DataType.Name(int(tensor_type.elem_type))
        except Exception:
            dtype = str(int(tensor_type.elem_type))
        output_specs.append({"name": value.name, "dtype": dtype, "shape": dims})
    fp.output_specs = output_specs
    # Learned values can make exporters omit optional parameter inputs (for
    # example, a zero-initialized Gemm bias). Keep required weight shapes in
    # the identity and normalize only optional slots declared by ONNX schema.
    execution_nodes = []
    for node, optional_slots in zip(nodes, optional_input_slots):
        execution_node = dict(node)
        execution_node["inputs"] = [
            item
            for index, item in enumerate(node["inputs"])
            if not (
                index in optional_slots
                and item.get("source") == "parameter"
            )
        ]
        execution_nodes.append(execution_node)
    fp.graph_hash = _stable_hash(
        {
            "nodes": execution_nodes,
            "input_shapes": _canonical_shapes(fp.input_shapes),
            "opset": sorted(
                (item.domain or "", int(item.version))
                for item in model.opset_import
            ),
        }
    )


def _canonical_shapes(shapes: Any) -> List[List[Any]]:
    """Ignore exporter-chosen symbols while preserving shape constraints."""
    canonical: List[List[Any]] = []
    for shape in shapes or []:
        canonical.append([
            "?" if isinstance(dim, str) else dim
            for dim in (shape or [])
        ])
    return canonical


def _fill_torch_fingerprint(path: Path, fp: ArtifactFingerprint) -> None:
    import torch  # type: ignore

    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception:
        return
    graph = getattr(module, "inlined_graph", None) or getattr(module, "graph", None)
    if graph is not None:
        def value_descriptor(value: Any) -> Dict[str, Any]:
            value_type = value.type()
            descriptor: Dict[str, Any] = {"type": str(value_type)}
            try:
                descriptor["shape"] = [
                    int(dim) if dim is not None else "?"
                    for dim in (value_type.sizes() or [])
                ]
            except Exception:
                descriptor["shape"] = []
            try:
                descriptor["dtype"] = str(value_type.scalarType() or "")
            except Exception:
                descriptor["dtype"] = ""
            return descriptor

        nodes = list(graph.nodes())
        kinds = [str(node.kind()) for node in nodes]
        fp.op_set = sorted(set(kinds))
        for node in nodes:
            try:
                schema = str(node.schema())
            except Exception:
                schema = ""
            signature = {
                "domain": "torchscript",
                "op_type": str(node.kind()),
                "schema": schema,
                "inputs": [value_descriptor(value) for value in node.inputs()],
                "outputs": [value_descriptor(value) for value in node.outputs()],
            }
            fp.subgraph_signatures.append(signature)
            fp.subgraph_signature_hashes.append(_stable_hash(signature))
            fp.op_signatures.append(
                f"{node.kind()}:{_stable_hash(signature)[:12]}"
            )
        fp.op_signatures = sorted(set(fp.op_signatures))
        fp.subgraph_signature_hashes = sorted(set(fp.subgraph_signature_hashes))
        fp.graph_hash = _stable_hash(str(graph))
        graph_inputs = list(graph.inputs())[1:]
        fp.input_specs = [value_descriptor(value) for value in graph_inputs]
        fp.output_specs = [value_descriptor(value) for value in graph.outputs()]
        fp.input_shapes = [
            list(item.get("shape") or [])
            for item in fp.input_specs
            if item.get("shape")
        ]
    try:
        shapes = [list(map(int, tensor.shape)) for tensor in module.state_dict().values()]
        fp.param_shapes = shapes
        fp.param_count = int(sum(math.prod(shape or [1]) for shape in shapes))
    except Exception:
        pass


def _fill_tflite_fingerprint(path: Path, fp: ArtifactFingerprint) -> None:
    interpreter_cls = None
    for module_name in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            module = __import__(module_name, fromlist=["Interpreter"])
            interpreter_cls = getattr(module, "Interpreter")
            break
        except Exception:
            continue
    if interpreter_cls is None:
        return
    interpreter = interpreter_cls(model_path=str(path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    tensor_details = {
        int(item.get("index")): item
        for item in interpreter.get_tensor_details()
        if item.get("index") is not None
    }

    def tensor_descriptor(index: Any) -> Dict[str, Any]:
        item = tensor_details.get(int(index), {})
        shape = item.get("shape_signature", item.get("shape", []))
        try:
            shape = [int(dim) for dim in shape]
        except Exception:
            shape = []
        quantization = item.get("quantization_parameters") or {}
        scales = quantization.get("scales", [])
        zero_points = quantization.get("zero_points", [])
        return {
            "dtype": str(item.get("dtype") or ""),
            "shape": shape,
            "quantization": {
                "scale_count": int(getattr(scales, "size", len(scales))),
                "zero_point_count": int(getattr(zero_points, "size", len(zero_points))),
                "axis": quantization.get("quantized_dimension"),
            },
        }

    fp.input_shapes = [list(map(int, item.get("shape", []))) for item in inputs]
    fp.input_specs = [
        {
            "name": str(item.get("name") or ""),
            "dtype": str(item.get("dtype") or ""),
            "shape": list(map(int, item.get("shape_signature", item.get("shape", [])))),
        }
        for item in inputs
    ]
    fp.output_specs = [
        {
            "name": str(item.get("name") or ""),
            "dtype": str(item.get("dtype") or ""),
            "shape": list(map(int, item.get("shape_signature", item.get("shape", [])))),
        }
        for item in outputs
    ]
    ops: List[str] = []
    try:
        for item in interpreter._get_ops_details():  # noqa: SLF001
            op_name = str(item.get("op_name") or "")
            if not op_name:
                continue
            ops.append(op_name)
            signature = {
                "domain": "tflite",
                "op_type": op_name,
                "inputs": [tensor_descriptor(index) for index in item.get("inputs", [])],
                "outputs": [tensor_descriptor(index) for index in item.get("outputs", [])],
            }
            fp.subgraph_signatures.append(signature)
            fp.subgraph_signature_hashes.append(_stable_hash(signature))
            fp.op_signatures.append(f"tflite::{op_name}:{_stable_hash(signature)[:12]}")
    except Exception:
        pass
    fp.op_set = sorted({op for op in ops if op})
    fp.op_signatures = sorted(set(fp.op_signatures))
    fp.subgraph_signature_hashes = sorted(set(fp.subgraph_signature_hashes))
    fp.graph_hash = _stable_hash(
        {
            "subgraphs": fp.subgraph_signatures,
            "input_shapes": fp.input_shapes,
        }
    )


def build_artifact_fingerprint(
    artifact_path: str,
    *,
    source_artifact_path: str = "",
) -> ArtifactFingerprint:
    """Return a lightweight fingerprint for a deployable artifact."""
    path = Path(str(artifact_path or ""))
    if not path.exists() or not path.is_file():
        return ArtifactFingerprint()

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    fmt = path.suffix.lower().lstrip(".") or "file"
    fp = ArtifactFingerprint(artifact_format=fmt, artifact_hash=digest.hexdigest())
    try:
        if fmt == "onnx":
            _fill_onnx_graph_fingerprint(path, fp)
        elif fmt in {"pt", "pth", "torchscript", "ts"}:
            _fill_torch_fingerprint(path, fp)
        elif fmt == "tflite":
            _fill_tflite_fingerprint(path, fp)
        elif fmt == "engine":
            source = Path(source_artifact_path) if source_artifact_path else path.with_suffix(".onnx")
            if source.exists():
                source_fp = build_artifact_fingerprint(str(source))
                fp.graph_hash = source_fp.graph_hash
                fp.op_set = source_fp.op_set
                fp.op_signatures = source_fp.op_signatures
                fp.subgraph_signatures = source_fp.subgraph_signatures
                fp.subgraph_signature_hashes = source_fp.subgraph_signature_hashes
                fp.param_count = source_fp.param_count
                fp.param_shapes = source_fp.param_shapes
                fp.input_shapes = source_fp.input_shapes
                fp.input_specs = source_fp.input_specs
                fp.source_artifact_hash = source_fp.artifact_hash
                fp.artifact_properties = source_fp.artifact_properties
    except Exception:
        return fp
    return fp


def compare_artifact_fingerprints(
    l1: Dict[str, Any],
    l2: Dict[str, Any],
    *,
    param_tolerance: float = 0.15,
) -> Dict[str, Any]:
    """Coarse congruence check between cheap and full artifacts."""
    if not l1 or not l2:
        return {"checked": False, "match": None, "reason": "missing_fingerprint"}
    mismatches: List[str] = []
    if l1.get("artifact_format") and l2.get("artifact_format") and l1.get("artifact_format") != l2.get("artifact_format"):
        mismatches.append("artifact_format")
    if l1.get("graph_hash") and l2.get("graph_hash"):
        if l1.get("graph_hash") != l2.get("graph_hash"):
            mismatches.append("graph_hash")
    elif sorted(l1.get("op_set") or []) != sorted(l2.get("op_set") or []):
        mismatches.append("op_set")
    if _canonical_shapes(l1.get("input_shapes")) != _canonical_shapes(l2.get("input_shapes")):
        mismatches.append("input_shapes")
    p1 = l1.get("param_count")
    p2 = l2.get("param_count")
    if isinstance(p1, int) and isinstance(p2, int):
        denom = max(abs(p1), abs(p2), 1)
        if abs(p1 - p2) / float(denom) > param_tolerance:
            mismatches.append("param_count")
    l1_hashes = set(l1.get("subgraph_signature_hashes") or [])
    l2_hashes = set(l2.get("subgraph_signature_hashes") or [])
    diff = {
        "l1_only_subgraph_hashes": sorted(l1_hashes - l2_hashes)[:20],
        "l2_only_subgraph_hashes": sorted(l2_hashes - l1_hashes)[:20],
        "l1_subgraph_count": len(l1.get("subgraph_signature_hashes") or []),
        "l2_subgraph_count": len(l2.get("subgraph_signature_hashes") or []),
        "l1_input_shapes": l1.get("input_shapes") or [],
        "l2_input_shapes": l2.get("input_shapes") or [],
        "l1_param_shapes": (l1.get("param_shapes") or [])[:30],
        "l2_param_shapes": (l2.get("param_shapes") or [])[:30],
    }
    return {
        "checked": True,
        "match": not mismatches,
        "mismatches": mismatches,
        "structural_diff": diff,
    }


class VerifierPolicy:
    """The only verifier component allowed to stop a candidate."""

    def __init__(self, *, kappa: float = 2.0) -> None:
        self.kappa = max(0.0, float(kappa))

    @staticmethod
    def _metric_name(name: str) -> str:
        try:
            from edgecraft.agent.metrics import MetricRegistry

            return MetricRegistry.canonicalize_name(name)
        except Exception:
            return str(name or "").strip()

    @staticmethod
    def authorized_p2(item: Evidence) -> bool:
        """Return whether evidence is allowed to accept a candidate.

        A numeric value alone is never enough.  Acceptance evidence must bind
        the value to the artifact that actually ran, the exact requested
        runtime, and a non-empty graph/environment identity.
        """
        protocol = item.protocol or {}
        fingerprint = item.artifact_fingerprint or {}
        latency_contract = True
        if VerifierPolicy._metric_name(item.quantity) == "Latency":
            latency_contract = (
                protocol.get("latency_statistic") == "p95"
                and int(protocol.get("measurement_sessions") or 0) >= 2
            )
        return (
            item.probe_id == "full"
            and item.fidelity == "measured_congruent"
            and item.outcome == "pass"
            and protocol.get("decision_authority") == "p2_accept"
            and protocol.get("artifact_executed") is True
            and protocol.get("runtime_exact") is True
            and protocol.get("fingerprint_verified") is True
            and protocol.get("evaluation_valid") is True
            and bool(fingerprint.get("graph_hash"))
            and bool(item.environment_fingerprint)
            and latency_contract
        )

    @staticmethod
    def _authorized_p1_prune(item: Evidence) -> bool:
        """Return whether cheap evidence has conservative prune authority."""
        protocol = item.protocol or {}
        fingerprint = item.artifact_fingerprint or {}
        pair_ids = protocol.get("calibration_pair_ids") or []
        latency_contract = True
        if VerifierPolicy._metric_name(item.quantity) == "Latency":
            latency_contract = (
                protocol.get("latency_statistic") == "p95"
                and protocol.get("latency_uncertainty")
                == "sample_std_across_session_p95"
                and int(protocol.get("measurement_sessions") or 0) >= 2
                and item.sigma is not None
            )
        mutation_count = protocol.get("p1_training_mutation_count")
        construction_guard = (
            protocol.get("p1_construction_guard") == "edgecraft_p1_guard_v1"
            and protocol.get("p1_construction_guard_status") == "passed"
            and bool(protocol.get("p1_candidate_source_sha256"))
            and bool(protocol.get("p1_guard_source_sha256"))
            and protocol.get("p1_candidate_source_unchanged") is True
            and protocol.get("p1_guard_source_unchanged") is True
            and protocol.get("p1_process_spawn_policy") == "blocked"
            and isinstance(mutation_count, int)
            and not isinstance(mutation_count, bool)
            and mutation_count == 0
        )
        return (
            item.probe_id == "efficiency"
            and item.fidelity == "measured_congruent"
            and item.outcome == "pass"
            and protocol.get("decision_authority") == "p1_prune"
            and protocol.get("congruence_status") == "calibrated_exact_graph"
            and protocol.get("runtime_exact") is True
            and protocol.get("fingerprint_verified") is True
            and bool(fingerprint.get("graph_hash"))
            and bool(item.environment_fingerprint)
            and bool(protocol.get("calibration_protocol_fingerprint"))
            and bool(protocol.get("calibration_snapshot_id"))
            and isinstance(pair_ids, list)
            and bool(pair_ids)
            and protocol.get("calibration_error") is not None
            and construction_guard
            and latency_contract
        )

    def decide(
        self,
        evidence: List[Evidence],
        user_spec: Any,
        *,
        next_probe: str = "full",
    ) -> VerificationDecision:
        for item in evidence:
            if (
                item.fidelity == "static"
                and item.outcome == "fail"
                and bool(item.protocol.get("verified_rule"))
                and bool(item.protocol.get("rule_id"))
            ):
                return VerificationDecision(
                    action="prune",
                    basis=[item.id],
                    reason="Verified static incompatibility matched the candidate artifact.",
                )

        parity = [item for item in evidence if item.probe_id == "edge_quality_parity"]
        strict_parity = [item for item in parity if item.protocol.get("strict_required")]
        if strict_parity:
            bundle = [
                item for item in evidence if item.probe_id == "edge_eval_bundle_contract"
            ]
            latest_parity = strict_parity[-1]
            latest_bundle = bundle[-1] if bundle else None
            if (
                latest_parity.outcome != "pass"
                or latest_bundle is None
                or latest_bundle.outcome != "pass"
            ):
                basis = [item.id for item in (latest_bundle, latest_parity) if item]
                return VerificationDecision(
                    action="escalate",
                    basis=basis,
                    reason=(
                        "Strict edge evaluation lacks a valid bundle and matching "
                        "local/edge quality evidence."
                    ),
                    next_probe="",
                )

        constraints = list(getattr(user_spec, "constraints", []) or []) if user_spec else []
        full_evidence = [item for item in evidence if self.authorized_p2(item)]
        if full_evidence:
            full_by_metric = {
                self._metric_name(item.quantity): item
                for item in full_evidence
                if item.value is not None
            }
            basis: List[str] = []
            for constraint in constraints:
                item = full_by_metric.get(self._metric_name(getattr(constraint, "metric", "")))
                if item is None:
                    return VerificationDecision(
                        action="escalate",
                        basis=basis,
                        reason="Full verification completed without every required metric.",
                        next_probe="",
                    )
                basis.append(item.id)
                value = float(item.value)
                target = float(getattr(constraint, "target"))
                comparison = str(getattr(constraint, "comparison", ""))
                ok = (
                    (comparison == "lte" and value <= target)
                    or (comparison == "gte" and value >= target)
                    or (comparison == "eq" and value == target)
                )
                if not ok:
                    return VerificationDecision(
                        action="escalate",
                        basis=basis,
                        reason="Full verification completed but the candidate violates a user constraint.",
                        next_probe="",
                    )
            if constraints:
                return VerificationDecision(
                    action="accept",
                    basis=basis,
                    reason="Full measured evidence satisfies every hard constraint.",
                )

        for constraint in constraints:
            target_name = self._metric_name(getattr(constraint, "metric", ""))
            matching = [
                item
                for item in evidence
                if self._metric_name(item.quantity) == target_name
                and self._authorized_p1_prune(item)
                and item.value is not None
            ]
            for item in reversed(matching):
                calibration_error = item.protocol.get("calibration_error")
                if calibration_error is None:
                    continue
                sigma = abs(float(item.sigma or 0.0))
                error = abs(float(calibration_error or 0.0))
                value = float(item.value)
                target = float(getattr(constraint, "target"))
                comparison = str(getattr(constraint, "comparison", ""))
                if comparison == "lte" and value - self.kappa * sigma - error > target:
                    return VerificationDecision(
                        action="prune",
                        basis=[item.id],
                        reason=f"Optimistic {target_name} bound still exceeds the user constraint.",
                    )
                if comparison == "gte" and value + self.kappa * sigma + error < target:
                    return VerificationDecision(
                        action="prune",
                        basis=[item.id],
                        reason=f"Optimistic {target_name} bound still misses the user constraint.",
                    )

        return VerificationDecision(
            action="escalate",
            reason="No authorized evidence proves the candidate infeasible or fully acceptable.",
            next_probe=next_probe,
        )


def build_l2_verification_report(trial: Any) -> VerificationReport:
    """Map the current full pipeline result into an L2 report."""

    metrics: Dict[str, MetricEstimate] = {}
    local = getattr(trial, "local_metrics", None)
    if local and getattr(local, "all_metrics", None):
        for name, value in local.all_metrics.items():
            try:
                metrics[name] = MetricEstimate(value=float(value), source="local_train_eval")
            except Exception:
                continue
    edge = getattr(trial, "edge_metrics", None)
    if edge:
        from edgecraft.agent.metrics import MetricRegistry

        latency_p95 = getattr(edge, "latency_p95_ms", None)
        if MetricRegistry.is_valid_measurement("Latency", latency_p95):
            metrics["Latency"] = MetricEstimate(
                value=float(latency_p95),
                source="edge_benchmark",
                unit="ms",
            )
        if MetricRegistry.is_valid_measurement("Memory_mb", getattr(edge, "memory_mb", None)):
            metrics["Memory_mb"] = MetricEstimate(
                value=float(edge.memory_mb),
                source="edge_benchmark",
                unit="MB",
            )
        for name, value in (getattr(edge, "all_metrics", {}) or {}).items():
            if name in metrics:
                continue
            try:
                if not MetricRegistry.is_valid_measurement(name, value):
                    continue
                metrics[name] = MetricEstimate(value=float(value), source="edge_benchmark")
            except Exception:
                continue

    artifact_contract = getattr(trial, "artifact_contract", None)
    runtime_report = getattr(trial, "runtime_report", None)
    primary_artifact = getattr(artifact_contract, "primary_artifact", "") if artifact_contract else ""
    runtime_used = getattr(runtime_report, "runtime_used", "") if runtime_report else ""
    level = "L1" if str(runtime_used).lower() == "surrogate" else "L2"
    evidence_source = "simulated" if level == "L1" else "measured"
    status = "pass" if not getattr(trial, "error", None) else "fail"
    errors = []
    if getattr(trial, "error", None):
        errors.append(str(trial.error))
    artifact_fp = build_artifact_fingerprint(primary_artifact).model_dump(mode="json") if primary_artifact else {}
    fidelity = "proxy" if level == "L1" else "measured_congruent"
    probe_id = "surrogate_prior" if level == "L1" else "full"
    evidence = [
        Evidence(
            probe_id=probe_id,
            quantity=name,
            fidelity=fidelity,
            outcome="unknown" if fidelity == "proxy" else "pass",
            value=estimate.value,
            sigma=estimate.sigma,
            unit=estimate.unit,
            artifact_fingerprint=artifact_fp,
            protocol={"legacy_mapped": True},
        )
        for name, estimate in metrics.items()
        if estimate.value is not None
    ]
    return VerificationReport(
        level=level,
        status=status,
        evidence_source=evidence_source,
        decision="none",
        metric_estimates=metrics,
        evidence=evidence,
        artifact_fingerprint=artifact_fp,
        artifact_status="present" if primary_artifact else "missing",
        runtime_status=str(runtime_used or ""),
        errors=errors,
        source="surrogate_gate" if level == "L1" else "full_pipeline",
        notes=(
            "Surrogate/cheap verification mapped to L1 evidence."
            if level == "L1"
            else "Current full train/export/edge verification mapped to L2 evidence."
        ),
    )


def verification_to_prompt_dict(report: Optional[VerificationReport]) -> Dict[str, Any]:
    if report is None:
        return {}
    return report.model_dump(mode="json")


for _model in (
    ResourceCost,
    Evidence,
    VerificationDecision,
    MetricEstimate,
    VerificationReport,
    GapSlackBrief,
    ArtifactFingerprint,
):
    _model.model_rebuild()
