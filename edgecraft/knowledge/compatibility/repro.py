"""Compatibility reproduction plans.

An automatic replay is useful evidence, but only a minimal reproduction proves
the structural scope of a reusable rule. This module keeps that distinction in
one small data object instead of spreading runtime-specific authority checks
through the pipeline.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReproductionPlan:
    """One dataset-free replay plan and the authority it may earn."""

    artifact_path: str = ""
    infer_script: str = ""
    runtime: str = ""
    scope: str = "unavailable"
    gate_eligible: bool = False
    pattern: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_onnx_operator_repro(
    artifact_path: str,
    *,
    op_type: str,
    output_path: str,
) -> Optional[str]:
    """Write a single-operator ONNX model with the original tensor metadata."""
    if not artifact_path or not str(artifact_path).lower().endswith(".onnx") or not op_type:
        return None
    try:
        import onnx  # type: ignore
        from onnx import TensorProto, helper, shape_inference  # type: ignore

        model = onnx.load(str(artifact_path), load_external_data=False)
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass
        node = next((item for item in model.graph.node if item.op_type.lower() == op_type.lower()), None)
        if node is None:
            return None
        value_info = {
            value.name: value
            for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
        }
        initializers = {item.name: item for item in model.graph.initializer}

        def _fallback_value(name: str):
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])

        graph_inputs = []
        graph_initializers = []
        for name in node.input:
            if not name:
                continue
            if name in initializers:
                graph_initializers.append(copy.deepcopy(initializers[name]))
            else:
                graph_inputs.append(copy.deepcopy(value_info.get(name) or _fallback_value(name)))
        graph_outputs = [
            copy.deepcopy(value_info.get(name) or _fallback_value(name))
            for name in node.output
            if name
        ]
        graph = helper.make_graph(
            [copy.deepcopy(node)],
            f"edgecraft_repro_{op_type}",
            graph_inputs,
            graph_outputs,
            initializer=graph_initializers,
        )
        repro = helper.make_model(
            graph,
            producer_name="edgecraft_failure_repro",
            opset_imports=[copy.deepcopy(item) for item in model.opset_import],
        )
        onnx.checker.check_model(repro)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(repro, str(target))
        return str(target)
    except Exception:
        return None


def extract_onnx_artifact_repro(
    artifact_path: str,
    *,
    artifact_predicate: dict,
    output_path: str,
) -> Optional[str]:
    """Write a tiny ONNX model preserving one verified header predicate."""
    if not artifact_path or not str(artifact_path).lower().endswith(".onnx"):
        return None
    expected = dict((artifact_predicate or {}).get("artifact_properties") or {})
    expected_dtypes = {
        str(item).upper() for item in (artifact_predicate or {}).get("input_dtypes") or []
    }
    if not {"ir_version", "opset_imports"}.intersection(expected) and not expected_dtypes:
        return None
    try:
        import onnx  # type: ignore
        from onnx import TensorProto, helper  # type: ignore

        source = onnx.load(str(artifact_path), load_external_data=False)
        source_input = next(
            (
                item
                for item in source.graph.input
                if item.type.HasField("tensor_type")
                and TensorProto.DataType.Name(
                    item.type.tensor_type.elem_type
                ).upper() in expected_dtypes
            ),
            None,
        )
        if expected_dtypes and source_input is None:
            return None
        if source_input is not None:
            input_info = copy.deepcopy(source_input)
            input_info.name = "input"
            output_info = copy.deepcopy(source_input)
            output_info.name = "output"
        else:
            input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
            output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])],
            "edgecraft_artifact_header_repro",
            [input_info],
            [output_info],
        )
        source_opset = next(
            (int(item.version) for item in source.opset_import if not item.domain),
            13,
        )
        repro = helper.make_model(
            graph,
            producer_name="edgecraft_failure_repro",
            opset_imports=[
                helper.make_opsetid("", source_opset if not expected_dtypes else min(source_opset, 13))
            ],
        )
        repro.ir_version = int(
            expected.get("ir_version", min(int(source.ir_version or 8), 8) if expected_dtypes else source.ir_version)
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(repro, str(target))
        return str(target)
    except Exception:
        return None


def write_runtime_repro_infer(
    output_path: str,
    *,
    input_specs: Optional[List[Dict[str, Any]]] = None,
    warmup: int = 3,
    min_warmup_seconds: float = 0.0,
    repetitions: int = 10,
    min_measure_seconds: float = 1.5,
    measurement_sessions: int = 3,
) -> str:
    """Write a dataset-free synthetic-input replay for supported edge runtimes."""
    script = '''#!/usr/bin/env python3
# edgecraft-runtime-driver: engine=trtexec
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

artifact = Path(os.environ.get("EDGECRAFT_ARTIFACT_PATH", "outputs/repro.onnx"))
runtime = os.environ.get("EDGECRAFT_RUNTIME", "onnx").strip().lower()
input_specs = json.loads('__EDGECRAFT_INPUT_SPECS__')
warmup = __EDGECRAFT_WARMUP__
min_warmup_seconds = __EDGECRAFT_MIN_WARMUP_SECONDS__
repetitions = __EDGECRAFT_REPETITIONS__
min_measure_seconds = __EDGECRAFT_MIN_MEASURE_SECONDS__
measurement_sessions = __EDGECRAFT_MEASUREMENT_SESSIONS__
resolved_input_specs = list(input_specs)

def now_ns():
    return int(time.time() * 1_000_000_000)

def percentile(samples, fraction):
    ordered = sorted(samples)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]

def shape_from(spec):
    shape = spec.get("shape") or [1]
    return [int(dim) if isinstance(dim, int) and dim > 0 else 1 for dim in shape]

def measure(call):
    all_samples = []
    session_means = []
    session_p95 = []
    session_p99 = []
    actual_warmup = 0
    actual_warmup_seconds = 0.0
    actual_repetitions = 0
    benchmark_started_ns = now_ns()
    for _ in range(measurement_sessions):
        warmup_started = time.perf_counter()
        session_warmup = 0
        while session_warmup < warmup or time.perf_counter() - warmup_started < min_warmup_seconds:
            call()
            session_warmup += 1
        actual_warmup += session_warmup
        actual_warmup_seconds += time.perf_counter() - warmup_started
        samples = []
        benchmark_started = time.perf_counter()
        while len(samples) < repetitions or time.perf_counter() - benchmark_started < min_measure_seconds:
            started = time.perf_counter()
            call()
            samples.append((time.perf_counter() - started) * 1000.0)
        all_samples.extend(samples)
        actual_repetitions += len(samples)
        session_means.append(statistics.mean(samples))
        session_p95.append(percentile(samples, 0.95))
        session_p99.append(percentile(samples, 0.99))
    benchmark_ended_ns = now_ns()
    return (
        statistics.mean(all_samples),
        statistics.pstdev(all_samples),
        statistics.mean(session_p95),
        statistics.mean(session_p99),
        statistics.stdev(session_p95) if len(session_p95) > 1 else None,
        session_p95,
        benchmark_started_ns,
        benchmark_ended_ns,
        actual_repetitions,
        actual_warmup,
        actual_warmup_seconds,
    )

provider = ""
latency_std_ms = None
latency_p95_ms = None
latency_p99_ms = None
latency_p95_session_std_ms = None
latency_p95_sessions_ms = []
warmup_unit = "iterations"
if runtime in {"engine", "tensorrt"}:
    print("failed_stage=artifact_load", flush=True)
    if not artifact.is_file() or artifact.suffix != ".engine":
        raise RuntimeError(f"TensorRT reproduction artifact missing: {artifact}")
    trtexec = shutil.which("trtexec")
    if not trtexec and Path("/usr/src/tensorrt/bin/trtexec").is_file():
        trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not trtexec:
        raise RuntimeError("TensorRT reproduction requires trtexec")
    trt_warmup_ms = max(warmup, int(math.ceil(min_warmup_seconds * 1000.0)))
    session_means = []
    session_p95 = []
    session_p99 = []
    actual_repetitions = 0
    benchmark_start_ns = now_ns()
    for _ in range(measurement_sessions):
        session_started_ns = now_ns()
        completed = subprocess.run(
            [
                trtexec,
                f"--loadEngine={artifact}",
                f"--warmUp={trt_warmup_ms}",
                f"--iterations={repetitions}",
                f"--duration={max(1, int(math.ceil(min_measure_seconds)))}",
            ],
            capture_output=True,
            text=True,
        )
        session_ended_ns = now_ns()
        output = "\\n".join((completed.stdout, completed.stderr))
        if completed.returncode != 0:
            raise RuntimeError(output[-2000:] or "TensorRT inference failed")
        match = re.search(
            r"GPU Compute Time:\\s*min\\s*=\\s*([0-9.]+)\\s*ms.*?mean\\s*=\\s*([0-9.]+)\\s*ms",
            output,
            re.S,
        )
        p95_match = re.search(r"percentile\\(95%\\)\\s*=\\s*([0-9.]+)\\s*ms", output)
        p99_match = re.search(r"percentile\\(99%\\)\\s*=\\s*([0-9.]+)\\s*ms", output)
        if not match or not p95_match:
            raise RuntimeError("TensorRT inference completed without mean and p95 timing evidence")
        session_means.append(float(match.group(2)))
        session_p95.append(float(p95_match.group(1)))
        session_p99.append(float(p99_match.group(1)) if p99_match else float(p95_match.group(1)))
        throughput_match = re.search(r"Throughput:\\s*([0-9.]+)\\s*qps", output)
        throughput_qps = float(throughput_match.group(1)) if throughput_match else None
        actual_repetitions += max(
            repetitions,
            int(round((throughput_qps or 0.0) * (session_ended_ns - session_started_ns) / 1e9)),
        )
    benchmark_end_ns = now_ns()
    latency_ms = statistics.mean(session_means)
    latency_std_ms = statistics.pstdev(session_means)
    latency_p95_ms = statistics.mean(session_p95)
    latency_p99_ms = statistics.mean(session_p99)
    latency_p95_session_std_ms = statistics.stdev(session_p95) if len(session_p95) > 1 else None
    latency_p95_sessions_ms = session_p95
    actual_warmup = trt_warmup_ms * measurement_sessions
    actual_warmup_seconds = trt_warmup_ms * measurement_sessions / 1000.0
    runtime_used = "engine"
    provider = "TensorRT.trtexec"
    warmup_unit = "ms"
    execution_scope = "synthetic_inference"
    benchmark_scope = "runtime_process"
elif runtime in {"pt", "pth", "torch", "pytorch", "torchscript"}:
    import torch

    print("failed_stage=artifact_load", flush=True)
    module = torch.jit.load(str(artifact), map_location="cpu")
    module.eval()
    args = []
    for spec in input_specs or [{"shape": [1]}]:
        dtype_name = str(spec.get("dtype") or "").lower()
        dtype = torch.int64 if "int64" in dtype_name or "long" in dtype_name else torch.float32
        args.append(torch.zeros(shape_from(spec), dtype=dtype))
    def invoke():
        with torch.no_grad():
            module(*args)
    print("failed_stage=inference", flush=True)
    latency_ms, latency_std_ms, latency_p95_ms, latency_p99_ms, latency_p95_session_std_ms, latency_p95_sessions_ms, benchmark_start_ns, benchmark_end_ns, actual_repetitions, actual_warmup, actual_warmup_seconds = measure(invoke)
    runtime_used = "torch"
    provider = "torchscript_cpu"
    execution_scope = "synthetic_inference"
    benchmark_scope = "inference_loop"
elif runtime in {"tflite", "litert"}:
    import numpy as np

    interpreter_cls = None
    provider = ""
    for module_name in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            module = __import__(module_name, fromlist=["Interpreter"])
            interpreter_cls = getattr(module, "Interpreter")
            provider = module_name
            break
        except Exception:
            continue
    if interpreter_cls is None:
        raise RuntimeError("LiteRT interpreter is unavailable")
    print("failed_stage=artifact_load", flush=True)
    interpreter = interpreter_cls(model_path=str(artifact))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    resolved_input_specs = []
    for item in input_details:
        shape = [int(dim) if int(dim) > 0 else 1 for dim in item.get("shape", [1])]
        resolved_input_specs.append({
            "name": str(item.get("name") or ""),
            "shape": shape,
            "dtype": str(item["dtype"]),
        })
        interpreter.set_tensor(int(item["index"]), np.zeros(shape, dtype=item["dtype"]))
    print("failed_stage=inference", flush=True)
    latency_ms, latency_std_ms, latency_p95_ms, latency_p99_ms, latency_p95_session_std_ms, latency_p95_sessions_ms, benchmark_start_ns, benchmark_end_ns, actual_repetitions, actual_warmup, actual_warmup_seconds = measure(interpreter.invoke)
    runtime_used = "litert"
    execution_scope = "synthetic_inference"
    benchmark_scope = "inference_loop"
else:
    import numpy as np
    import onnxruntime as ort

    print("failed_stage=artifact_load", flush=True)
    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    feeds = {}
    resolved_input_specs = []
    for item in session.get_inputs():
        shape = [int(dim) if isinstance(dim, int) and dim > 0 else 1 for dim in item.shape]
        dtype = np.float32
        if "int64" in item.type:
            dtype = np.int64
        elif "int32" in item.type:
            dtype = np.int32
        feeds[item.name] = np.zeros(shape, dtype=dtype)
        resolved_input_specs.append({"name": item.name, "shape": shape, "dtype": item.type})
    print("failed_stage=inference", flush=True)
    latency_ms, latency_std_ms, latency_p95_ms, latency_p99_ms, latency_p95_session_std_ms, latency_p95_sessions_ms, benchmark_start_ns, benchmark_end_ns, actual_repetitions, actual_warmup, actual_warmup_seconds = measure(lambda: session.run(None, feeds))
    runtime_used = "onnxruntime"
    provider = (session.get_providers() or ["onnxruntime"])[0]
    execution_scope = "synthetic_inference"
    benchmark_scope = "inference_loop"
print(json.dumps({
    "status": "success",
    "runtime_used": runtime_used,
    "runtime_provider": provider,
    "artifact_used": str(artifact),
    "execution_scope": execution_scope,
    "measurement_protocol": {
        "warmup": warmup,
        "warmup_unit": warmup_unit,
        "min_warmup_seconds": min_warmup_seconds,
        "actual_warmup": actual_warmup,
        "actual_warmup_seconds": actual_warmup_seconds,
        "repetitions": repetitions,
        "actual_repetitions": actual_repetitions,
        "measurement_sessions": measurement_sessions,
        "latency_statistic": "p95",
        "latency_uncertainty": "sample_std_across_session_p95",
        "measurement_policy": "stabilized_minimum_duration" if min_warmup_seconds > 0 else "minimum_duration",
        "min_measure_seconds": min_measure_seconds,
        "benchmark_start_ns": benchmark_start_ns,
        "benchmark_end_ns": benchmark_end_ns,
        "benchmark_scope": benchmark_scope,
        "input_specs": resolved_input_specs,
    },
    "metrics": {
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "latency_p95_ms": latency_p95_ms,
        "latency_p99_ms": latency_p99_ms,
        "latency_p95_session_std_ms": latency_p95_session_std_ms,
        "latency_p95_sessions_ms": latency_p95_sessions_ms,
        "artifact_loaded": 1.0,
        "runtime_executed": 1.0,
        "inference_completed": 1.0,
        "warmup": warmup,
        "warmup_unit": warmup_unit,
        "repetitions": repetitions,
        "actual_repetitions": actual_repetitions,
    },
}))
'''
    script = script.replace(
        "__EDGECRAFT_INPUT_SPECS__",
        json.dumps(input_specs or [], sort_keys=True, default=str).replace("'", "\\u0027"),
    )
    script = script.replace("__EDGECRAFT_WARMUP__", str(max(0, int(warmup))))
    script = script.replace(
        "__EDGECRAFT_MIN_WARMUP_SECONDS__",
        repr(max(0.0, float(min_warmup_seconds))),
    )
    script = script.replace("__EDGECRAFT_REPETITIONS__", str(max(1, int(repetitions))))
    script = script.replace(
        "__EDGECRAFT_MIN_MEASURE_SECONDS__",
        repr(max(0.1, float(min_measure_seconds))),
    )
    script = script.replace(
        "__EDGECRAFT_MEASUREMENT_SESSIONS__",
        str(max(2, int(measurement_sessions))),
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    return str(path)


def write_onnxruntime_repro_infer(output_path: str) -> str:
    """Backward-compatible name for the runtime-aware compatibility replay."""
    return write_runtime_repro_infer(output_path)


def prepare_reproduction_plan(
    *,
    artifact_path: str,
    runtime: str,
    output_dir: str,
    op_type: str = "",
    artifact_predicate: Optional[Dict[str, Any]] = None,
    input_specs: Optional[List[Dict[str, Any]]] = None,
    source_onnx_path: str = "",
) -> ReproductionPlan:
    """Prepare the narrowest replay justified by the available artifact graph."""
    artifact = Path(str(artifact_path or ""))
    target_dir = Path(output_dir)
    normalized_runtime = str(runtime or artifact.suffix.lstrip(".")).strip().lower()
    source = Path(str(source_onnx_path or ""))
    if artifact.suffix.lower() == ".onnx":
        source = artifact
    elif not source.is_file() and artifact.suffix.lower() == ".engine":
        adjacent = artifact.with_suffix(".onnx")
        if adjacent.is_file():
            source = adjacent

    if source.is_file() and source.suffix.lower() == ".onnx":
        repro_path = str(target_dir / "outputs" / "repro.onnx")
        predicate = dict(artifact_predicate or {})
        if predicate:
            repro_artifact = extract_onnx_artifact_repro(
                str(source),
                artifact_predicate=predicate,
                output_path=repro_path,
            )
            scope = "artifact_header"
            pattern = {"artifact_predicate": predicate}
        elif op_type:
            repro_artifact = extract_onnx_operator_repro(
                str(source),
                op_type=op_type,
                output_path=repro_path,
            )
            scope = "minimal_subgraph"
            pattern = {"op_type": op_type}
        else:
            repro_artifact = None
            scope = "unavailable"
            pattern = {}
        if repro_artifact:
            return ReproductionPlan(
                artifact_path=repro_artifact,
                infer_script=write_runtime_repro_infer(str(target_dir / "infer.py")),
                runtime=normalized_runtime,
                scope=scope,
                gate_eligible=True,
                pattern=pattern,
                reason="Inspectable ONNX source produced a minimal reproduction.",
            )

    if artifact.is_file() and artifact.suffix.lower() in {
        ".onnx",
        ".engine",
        ".pt",
        ".pth",
        ".torchscript",
        ".ts",
        ".tflite",
    }:
        return ReproductionPlan(
            artifact_path=str(artifact),
            infer_script=write_runtime_repro_infer(
                str(target_dir / "infer.py"),
                input_specs=input_specs,
            ),
            runtime=normalized_runtime,
            scope="full_artifact_replay",
            gate_eligible=False,
            reason=(
                "The failure can be replayed with synthetic input, but no minimal "
                "structural predicate has been proved."
            ),
        )

    return ReproductionPlan(
        runtime=normalized_runtime,
        reason="No inspectable deployable artifact is available for automatic replay.",
    )
