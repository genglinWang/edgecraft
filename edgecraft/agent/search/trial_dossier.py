"""TrialDossier: read-only evidence view for semantic branch judgment.

The dossier is deliberately not a controller.  It does not classify, score,
repair, or override a trial.  It only turns existing EdgeCraft state into one
clean evidence object for the LLM branch evaluator and proposal context.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from edgecraft.agent.contracts import inspect_loader_train_contract
from edgecraft.agent.metrics import MetricRegistry, lookup_trial_metric
from edgecraft.config.settings import edgecraft_env


class TrialDossier(BaseModel):
    """Compact evidence envelope for one measured code-space branch."""

    user_intent: str = ""
    modality: str = ""
    task_type: str = ""
    target_device: str = ""
    trial: Dict[str, Any] = Field(default_factory=dict)
    variant: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    training_trace: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    primary_failure: Dict[str, Any] = Field(default_factory=dict)
    failure_context: Dict[str, Any] = Field(default_factory=dict)
    debug_history: List[Dict[str, Any]] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    dataset_evidence: Dict[str, Any] = Field(default_factory=dict)
    runtime_environment: Dict[str, Any] = Field(default_factory=dict)
    search_history: List[Dict[str, Any]] = Field(default_factory=list)
    best_seen: Dict[str, Any] = Field(default_factory=dict)
    sibling_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    comparison_evidence: Dict[str, Any] = Field(default_factory=dict)
    lineage_effect: Dict[str, Any] = Field(default_factory=dict)
    observations: Dict[str, Any] = Field(default_factory=dict)
    component_evidence: Dict[str, Any] = Field(default_factory=dict)
    verification: Dict[str, Any] = Field(default_factory=dict)
    gap_slack: List[Dict[str, Any]] = Field(default_factory=list)
    compatibility_hits: List[Dict[str, Any]] = Field(default_factory=list)
    failure_observations: List[Dict[str, Any]] = Field(default_factory=list)

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict for LLM prompts."""
        return self.model_dump(mode="json")


def trial_dossier_to_prompt_text(dossier: TrialDossier, *, limit: int = 7000) -> str:
    """Render the evidence object for proposal prompts.

    This is intentionally a view, not a second decision system.  The fields are
    the same dossier facts used by BranchJudgment, trimmed only to keep prompts
    bounded.
    """
    data = dossier.to_prompt_dict()
    selected: Dict[str, Any] = {}
    for key in (
        "trial",
        "primary_failure",
        "validity_evidence",
        "verification",
        "compatibility_hits",
        "failure_observations",
        "gap_slack",
        "metrics",
        "component_evidence",
        "training_trace",
        "comparison_evidence",
        "lineage_effect",
        "variant",
        "best_seen",
        "sibling_evidence",
        "artifacts",
        "observations",
        "debug_history",
        "failure_context",
    ):
        value = (
            _validity_evidence(data.get("observations") or {})
            if key == "validity_evidence"
            else data.get(key, [] if key in {"sibling_evidence", "debug_history", "gap_slack", "compatibility_hits", "failure_observations"} else {})
        )
        value = _drop_empty(value)
        if value not in ({}, [], "", None):
            selected[key] = value
    text = json.dumps(selected, ensure_ascii=False, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...(trial dossier truncated)"


def _validity_evidence(observations: Dict[str, Any]) -> Dict[str, Any]:
    """Keep compact scientific-validity facts ahead of verbose code/context."""
    return {
        key: observations[key]
        for key in (
            "feature_provenance",
            "edge_eval_bundle_contract",
            "edge_quality_parity",
            "edge_quality_provenance",
            "evaluation_validity",
        )
        if observations.get(key)
    }


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            compact = _drop_empty(item)
            if compact not in ({}, [], "", None):
                out[key] = compact
        return out
    if isinstance(value, list):
        return [
            item
            for item in (_drop_empty(v) for v in value)
            if item not in ({}, [], "", None)
        ]
    return value


def build_trial_dossier(
    trial: Any,
    state: Dict[str, Any],
    *,
    score: float,
    is_feasible: bool,
    constraint_violations: List[str],
) -> TrialDossier:
    """Build the single evidence object consumed by BranchJudgment."""
    variant = getattr(trial, "variant", None)
    expose_raw_output = bool(getattr(trial, "error", "") or getattr(trial, "error_stage", ""))
    return TrialDossier(
        user_intent=str(state.get("raw_user_intent", "") or ""),
        modality=str(state.get("modality", "") or ""),
        task_type=str(state.get("task_type", "") or ""),
        target_device=str(state.get("target_device", "") or ""),
        trial={
            "trial_id": getattr(trial, "trial_id", ""),
            "training_seed": getattr(trial, "training_seed", None),
            "score": score,
            "is_feasible": is_feasible,
            "constraint_violations": constraint_violations,
            "crafting_progress": getattr(getattr(trial, "crafting_progress", None), "value", ""),
            "stage_reached": getattr(getattr(trial, "stage_reached", None), "value", ""),
            "error_stage": getattr(trial, "error_stage", ""),
            "error": _tail(getattr(trial, "error", "") or "", 1200),
            "failure_taxonomy": getattr(trial, "failure_taxonomy", ""),
            "next_search_hint": getattr(trial, "next_search_hint", ""),
        },
        variant=_variant_summary(variant),
        metrics=_metrics_summary(trial),
        training_trace=_training_trace_summary(trial),
        artifacts=_artifact_summary(trial),
        primary_failure=_primary_failure_summary(trial),
        failure_context=_compact_failure_context(getattr(trial, "failure_context", {}) or {}),
        debug_history=[
            {
                "stage": getattr(d, "stage", ""),
                "attempt": getattr(d, "attempt", 0),
                "error_signature": getattr(d, "error_signature", ""),
                "patch_summary": getattr(d, "patch_summary", ""),
                "patch_applied": getattr(d, "patch_applied", False),
                "retry_passed": getattr(d, "success", False),
            }
            for d in list(getattr(trial, "debug_history", []) or [])[-4:]
        ],
        stdout_tail=(
            _tail(getattr(trial, "stdout", "") or "", 1600)
            if expose_raw_output else ""
        ),
        stderr_tail=(
            _tail(getattr(trial, "stderr", "") or "", 1600)
            if expose_raw_output else ""
        ),
        dataset_evidence=_compact_dataset_info(state.get("dataset_info") or {}),
        runtime_environment=_compact_runtime_config(state.get("runtime_config")),
        observations=_observation_summary(trial, state),
        component_evidence=_component_evidence_summary(trial, state),
        verification=_verification_summary(trial),
        gap_slack=_gap_slack_summary(trial),
        compatibility_hits=_compatibility_hits_summary(trial),
        failure_observations=_failure_observations_summary(trial),
        search_history=_compact_search_history(state),
        best_seen=_best_seen_summary(trial, state),
        sibling_evidence=_sibling_evidence_summary(trial, state),
        comparison_evidence=_comparison_evidence_summary(trial, state, score),
        lineage_effect=_lineage_effect_summary(trial, state),
    )


def _primary_failure_summary(trial: Any) -> Dict[str, Any]:
    """Keep the concrete failure visible before verbose component evidence."""
    ctx = getattr(trial, "failure_context", {}) or {}
    repair = ctx.get("repair_diagnosis") if isinstance(ctx, dict) else {}
    repair_evidence = repair.get("evidence") if isinstance(repair, dict) else {}
    diagnostics = ctx.get("diagnostics") if isinstance(ctx, dict) else {}
    sources = [
        repair_evidence if isinstance(repair_evidence, dict) else {},
        diagnostics if isinstance(diagnostics, dict) else {},
        ctx if isinstance(ctx, dict) else {},
    ]
    primary_error = next(
        (
            str(source.get("primary_error_tail") or "").strip()
            for source in sources
            if source.get("primary_error_tail")
        ),
        "",
    )
    return {
        key: value
        for key, value in {
            "stage": getattr(trial, "error_stage", "") or ctx.get("failed_stage", ""),
            "summary": _tail(getattr(trial, "error", "") or "", 800),
            "primary_error_tail": _tail(primary_error, 1600),
        }.items()
        if value not in (None, "", {}, [])
    }


def _lineage_effect_summary(trial: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Return factual parent-child deltas without interpreting the mutation."""
    if edgecraft_env("LINEAGE_EFFECT_CONTEXT", "1").strip().lower() in {"0", "false", "off", "no"}:
        return {}
    variant = getattr(trial, "variant", None)
    parent_id = str(getattr(variant, "parent_trial_id", "") or "")
    trial_bank = state.get("trial_bank")
    parent = trial_bank.get(parent_id) if parent_id and trial_bank is not None else None
    if parent is None:
        return {}

    user_spec = state.get("user_spec")
    metric_names = [
        str(getattr(item, "metric", "") or "")
        for item in list(getattr(user_spec, "preferences", []) or [])
    ]
    if not metric_names:
        current_local = dict(getattr(getattr(trial, "local_metrics", None), "all_metrics", {}) or {})
        parent_local = dict(getattr(getattr(parent, "local_metrics", None), "all_metrics", {}) or {})
        metric_names = sorted(set(current_local) & set(parent_local))

    quality_delta: Dict[str, float] = {}
    for raw_name in metric_names:
        name = MetricRegistry.canonicalize_name(raw_name)
        if not name or MetricRegistry.is_edge_metric(name):
            continue
        current_value = lookup_trial_metric(name, trial.local_metrics, trial.edge_metrics)
        parent_value = lookup_trial_metric(name, parent.local_metrics, parent.edge_metrics)
        if current_value is not None and parent_value is not None:
            quality_delta[name] = float(current_value) - float(parent_value)

    def _edge_value(item: Any, name: str) -> Any:
        return getattr(getattr(item, "edge_metrics", None), name, None)

    def _delta(current: Any, previous: Any) -> Any:
        if current is None or previous is None:
            return None
        try:
            return float(current) - float(previous)
        except (TypeError, ValueError):
            return None

    def _gap_map(item: Any) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for brief in list(getattr(item, "gap_slack", []) or []):
            metric = str(getattr(brief, "metric", "") or "")
            value = getattr(brief, "normalized_gap_high", None)
            if metric and value is not None:
                result[metric] = float(value)
        return result

    current_gaps = _gap_map(trial)
    parent_gaps = _gap_map(parent)
    gap_delta = {
        metric: current_gaps[metric] - parent_gaps[metric]
        for metric in sorted(set(current_gaps) & set(parent_gaps))
    }
    adherence = dict((getattr(trial, "observations", {}) or {}).get("constraint_adherence") or {})
    current_hashes = dict(adherence.get("file_hashes") or {})
    parent_hashes = dict(adherence.get("parent_file_hashes") or {})
    preserved_hashes = {
        name: digest
        for name, digest in current_hashes.items()
        if digest and digest == parent_hashes.get(name)
    }

    def _training_cost(item: Any) -> float:
        report = getattr(item, "verification_report", None)
        return sum(
            float(getattr(getattr(evidence, "resource_cost", None), "gpu_s", 0.0) or 0.0)
            for evidence in list(getattr(report, "evidence", []) or [])
        )

    def _artifact_fingerprint(item: Any) -> tuple[Dict[str, Any], List[str]]:
        report = getattr(item, "verification_report", None)
        raw = getattr(report, "artifact_fingerprint", {}) if report is not None else {}
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        fingerprint = dict(raw or {})
        compact = {
            key: fingerprint.get(key)
            for key in (
                "artifact_format",
                "graph_hash",
                "op_set",
                "param_count",
                "input_shapes",
                "artifact_properties",
            )
            if fingerprint.get(key) not in (None, "", [], {})
        }
        evidence_ids = [
            str(getattr(evidence, "id", "") or "")
            for evidence in list(getattr(report, "evidence", []) or [])
            if getattr(evidence, "artifact_fingerprint", None)
        ]
        return compact, [item for item in evidence_ids if item]

    current_artifact, current_artifact_evidence = _artifact_fingerprint(trial)
    parent_artifact, parent_artifact_evidence = _artifact_fingerprint(parent)
    comparable_artifact_fields = sorted(set(current_artifact) & set(parent_artifact))
    changed_artifact_fields = [
        name
        for name in comparable_artifact_fields
        if current_artifact[name] != parent_artifact[name]
    ]
    artifact_effect: Dict[str, Any] = {}
    if current_artifact and parent_artifact:
        artifact_effect = {
            "changed_fields": changed_artifact_fields,
            "deployed_graph_changed": bool(changed_artifact_fields),
            "parent_fingerprint": parent_artifact,
            "current_fingerprint": current_artifact,
            "parent_evidence_ids": parent_artifact_evidence,
            "current_evidence_ids": current_artifact_evidence,
        }

    return _drop_empty({
        "parent_trial_id": parent_id,
        "training_seed": getattr(trial, "training_seed", None),
        "parent_training_seed": getattr(parent, "training_seed", None),
        "paired_training_seed": (
            getattr(trial, "training_seed", None) is not None
            and getattr(trial, "training_seed", None) == getattr(parent, "training_seed", None)
        ),
        "evidence_refs": list(getattr(variant, "evidence_refs", []) or []),
        "declared_mutation": {
            "mutation_type": getattr(variant, "mutation_type", ""),
            "changed_components": list(getattr(variant, "changed_components", []) or []),
            "proposal_hypothesis": getattr(variant, "proposal_hypothesis", ""),
        },
        "actual_changed_components": adherence.get("actual_changed_files") or [],
        "preserved_component_hashes": preserved_hashes,
        "quality_delta": quality_delta,
        "latency_delta": _delta(_edge_value(trial, "latency_ms"), _edge_value(parent, "latency_ms")),
        "energy_delta": _delta(
            lookup_trial_metric("Energy_mj", trial.local_metrics, trial.edge_metrics),
            lookup_trial_metric("Energy_mj", parent.local_metrics, parent.edge_metrics),
        ),
        "constraint_gap_delta": gap_delta,
        "training_cost_delta": _training_cost(trial) - _training_cost(parent),
        "artifact_effect": artifact_effect,
    })


def _variant_summary(variant: Any) -> Dict[str, Any]:
    if variant is None:
        return {}
    return {
        "trial_id": getattr(variant, "trial_id", ""),
        "parent_trial_id": getattr(variant, "parent_trial_id", ""),
        "model_name": getattr(variant, "model_name", ""),
        "model_family": getattr(variant, "model_family", ""),
        "dataset_plan": _tail(getattr(variant, "dataset_plan", "") or "", 2000),
        "plan": _tail(getattr(variant, "plan", "") or "", 1200),
        "mutation_type": getattr(variant, "mutation_type", ""),
        "inherited_components": getattr(variant, "inherited_components", []),
        "changed_components": getattr(variant, "changed_components", []),
        "evidence_refs": getattr(variant, "evidence_refs", []),
        "proposal_hypothesis": getattr(variant, "proposal_hypothesis", ""),
        "solution_source": getattr(variant, "solution_source", ""),
        "initialization_source": getattr(variant, "initialization_source", ""),
        "representation_strategy": getattr(variant, "representation_strategy", ""),
        "representation_stage": getattr(variant, "representation_stage", ""),
        "model_capacity_strategy": getattr(variant, "model_capacity_strategy", ""),
        "training_recipe": getattr(variant, "training_recipe", ""),
        "export_runtime_strategy": getattr(variant, "export_runtime_strategy", ""),
        "search_granularity": getattr(variant, "search_granularity", ""),
        "search_intent": getattr(variant, "search_intent", ""),
        "contract_rewrite_reason": getattr(variant, "contract_rewrite_reason", ""),
    }


def _metrics_summary(trial: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    local = getattr(trial, "local_metrics", None)
    if local and getattr(local, "all_metrics", None):
        metrics["local"] = dict(list(local.all_metrics.items())[:20])
    edge = getattr(trial, "edge_metrics", None)
    runtime = getattr(trial, "runtime_report", None)
    if edge:
        metrics["edge"] = {
            "latency_ms": getattr(edge, "latency_ms", None),
            "memory_mb": getattr(edge, "memory_mb", None),
            "runtime_used": getattr(edge, "runtime_used", None) or getattr(runtime, "runtime_used", None),
            "runtime_provider": getattr(edge, "runtime_provider", None) or getattr(runtime, "runtime_provider", None),
            "artifact_used": getattr(edge, "artifact_used", None) or getattr(runtime, "artifact_used", None),
            "all_metrics": dict(list((getattr(edge, "all_metrics", {}) or {}).items())[:20]),
        }
    if metrics:
        metrics["roles"] = {
            "local": "tree-visible validation metric used for solution selection",
            "edge_efficiency": "measured latency, memory, power, and energy on the target device",
            "edge_quality": (
                "bounded parity-bundle metric; validates artifact semantics but is not an official "
                "held-out result or a replacement for the tree-visible validation metric"
            ),
        }
    return metrics


def _training_trace_summary(trial: Any) -> Dict[str, Any]:
    if edgecraft_env("TRAINING_TRACE_CONTEXT", "1").strip().lower() in {"0", "false", "off", "no"}:
        return {}
    trace = getattr(trial, "training_trace", None)
    if trace is None:
        return {}
    data = trace.model_dump(mode="json") if hasattr(trace, "model_dump") else dict(trace)
    points = list(data.get("points") or [])
    if len(points) > 12:
        indices = [round(i * (len(points) - 1) / 11) for i in range(12)]
        data["points"] = [points[index] for index in indices]
    report = getattr(trial, "verification_report", None)
    if report is not None:
        for item in reversed(list(getattr(report, "evidence", []) or [])):
            if getattr(item, "quantity", "") == "training_dynamics":
                data["evidence_id"] = str(getattr(item, "id", "") or "")
                break
    return _drop_empty(data)


def _artifact_summary(trial: Any) -> Dict[str, Any]:
    contract = getattr(trial, "artifact_contract", None)
    runtime = getattr(trial, "runtime_report", None)
    artifacts: Dict[str, Any] = {}
    if contract is not None:
        raw_artifacts = getattr(contract, "artifacts", {}) or {}
        for name, info in list(raw_artifacts.items())[:12]:
            if isinstance(info, dict):
                artifacts[name] = {
                    "kind": info.get("kind"),
                    "role": info.get("role"),
                    "runtime": info.get("runtime"),
                    "size_bytes": info.get("size_bytes"),
                }
            else:
                artifacts[name] = _json_safe(info, limit=600)
    return {
        "artifact_paths": getattr(trial, "artifact_paths", {}) or {},
        "primary_artifact": getattr(contract, "primary_artifact", "") if contract else "",
        "artifacts": artifacts,
        "runtime_used": getattr(runtime, "runtime_used", "") if runtime else "",
        "runtime_provider": getattr(runtime, "runtime_provider", "") if runtime else "",
        "artifact_used": getattr(runtime, "artifact_used", "") if runtime else "",
    }


def _verification_summary(trial: Any) -> Dict[str, Any]:
    report = getattr(trial, "verification_report", None)
    if not report:
        return {}
    data = report.model_dump(mode="json")
    evidence = []
    for item in list(data.get("evidence") or [])[-12:]:
        record = {
            key: item.get(key)
            for key in (
                "id",
                "probe_id",
                "quantity",
                "fidelity",
                "outcome",
                "value",
                "sigma",
                "unit",
                "environment_fingerprint",
                "resource_cost",
            )
            if item.get(key) not in (None, "", {}, [])
        }
        protocol = item.get("protocol") or {}
        if protocol:
            record["protocol"] = _compact_evidence_protocol(protocol)
        fingerprint = item.get("artifact_fingerprint") or {}
        if fingerprint:
            record["artifact_fingerprint"] = {
                key: fingerprint.get(key)
                for key in (
                    "artifact_format",
                    "graph_hash",
                    "input_specs",
                    "input_shapes",
                    "param_count",
                )
                if fingerprint.get(key) not in (None, "", {}, [])
            }
        evidence.append(record)
    return {
        key: value
        for key, value in {
            "level": data.get("level"),
            "status": data.get("status"),
            "decision": data.get("decision"),
            "decision_basis": data.get("decision_basis") or [],
            "next_probe": data.get("next_probe"),
            "source": data.get("source"),
            "notes": _tail(str(data.get("notes") or ""), 800),
            "artifact_fingerprint": _json_safe(data.get("artifact_fingerprint") or {}, limit=1800),
            "evidence": evidence,
        }.items()
        if value not in (None, "", {}, [])
    }


def _gap_slack_summary(trial: Any) -> List[Dict[str, Any]]:
    items = getattr(trial, "gap_slack", []) or []
    out: List[Dict[str, Any]] = []
    for item in items[:12]:
        if hasattr(item, "model_dump"):
            out.append(_json_safe(item.model_dump(mode="json"), limit=1200))
        elif isinstance(item, dict):
            out.append(_json_safe(item, limit=1200))
    return out


def _compatibility_hits_summary(trial: Any) -> List[Dict[str, Any]]:
    observations = getattr(trial, "observations", {}) or {}
    if not isinstance(observations, dict):
        return []
    hits = observations.get("compatibility_hits") or []
    if not isinstance(hits, list):
        return []
    return [_json_safe(hit, limit=1200) for hit in hits[:8]]


def _failure_observations_summary(trial: Any) -> List[Dict[str, Any]]:
    observations = getattr(trial, "observations", {}) or {}
    if not isinstance(observations, dict):
        return []
    items = observations.get("failure_observations") or []
    if not isinstance(items, list):
        return []
    return [_json_safe(item, limit=1200) for item in items[:8]]


def _compact_failure_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("repair_diagnosis", "diagnostics", "loader_smoke", "runtime_attempts"):
        value = ctx.get(key)
        if value:
            out[key] = _json_safe(value, limit=3500)
    for key in ("failed_stage", "returncode", "timeout", "command"):
        if key in ctx:
            out[key] = ctx.get(key)
    return out


def _compact_dataset_info(info: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "dataset_path",
        "dataset_root",
        "config_path",
        "modality",
        "task_type",
        "format",
        "description",
        "structure_summary",
        "splits",
        "classes",
        "schema",
        "sample_observation",
        "exploration_report",
        "path_facts",
    ]
    return {k: _json_safe(info.get(k), limit=2500) for k in keys if info.get(k) is not None}


def _compact_runtime_config(runtime_config: Any) -> Dict[str, Any]:
    if runtime_config is None:
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "available_runtimes",
        "docker_image",
        "device_arch",
        "has_gpu",
        "trt_version",
        "onnxruntime_version",
        "onnxruntime_providers",
        "torch_version",
        "l4t_version",
        "cloud_python_packages",
        "python_packages",
        "runtime_python_packages",
        "edge_artifact_bundle_max_mb",
    ):
        value = getattr(runtime_config, key, None)
        if value not in (None, "", [], {}):
            out[key] = _json_safe(value, limit=2200)
    return out


def _observation_summary(trial: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Collect factual observations already produced by the pipeline.

    This is deliberately a read-only evidence view.  It does not classify,
    score, repair, or alter control flow.
    """
    ctx = getattr(trial, "failure_context", {}) or {}
    dataset_info = state.get("dataset_info") or {}
    artifacts = _artifact_summary(trial)
    metrics = _metrics_summary(trial)
    raw_observations = getattr(trial, "observations", {}) or {}
    out: Dict[str, Any] = {}
    if isinstance(raw_observations, dict):
        out = {
            str(key): _json_safe(value, limit=2200)
            for key, value in raw_observations.items()
        }
    elif raw_observations:
        out["raw_observations"] = _json_safe(raw_observations, limit=2200)

    loader_smoke = ctx.get("loader_smoke")
    if loader_smoke:
        out.setdefault("loader_smoke", _json_safe(loader_smoke, limit=2200))
        sample_obs = _extract_sample_observation(ctx)
        if sample_obs:
            out.setdefault("sample_observation", _json_safe(sample_obs, limit=2200))
            out.setdefault("sample_summary", _sample_observation_summary(sample_obs))

    diagnostics = ctx.get("diagnostics")
    if diagnostics:
        out.setdefault("diagnostics", _json_safe(diagnostics, limit=2200))

    repair = ctx.get("repair_diagnosis") or {}
    evidence = repair.get("evidence") if isinstance(repair, dict) else None
    if isinstance(evidence, dict):
        for key in ("sample_observation", "local_metrics", "edge_metrics", "artifact_manifest"):
            if evidence.get(key):
                out.setdefault(key, _json_safe(evidence.get(key), limit=2200))

    if dataset_info.get("sample_observation"):
        dataset_sample = dataset_info.get("sample_observation")
        best_sample = _more_informative_sample(out.get("sample_observation"), dataset_sample)
        if best_sample:
            out["sample_observation"] = _json_safe(best_sample, limit=2200)
            out["sample_summary"] = _sample_observation_summary(best_sample)
    if dataset_info.get("schema"):
        out["schema"] = _json_safe(dataset_info.get("schema"), limit=1800)
    if dataset_info.get("splits"):
        out["splits"] = _json_safe(dataset_info.get("splits"), limit=1200)
    if dataset_info.get("classes"):
        out["classes"] = _json_safe(dataset_info.get("classes"), limit=1200)

    artifact_bits = {k: v for k, v in artifacts.items() if v}
    if artifact_bits:
        out["artifact_runtime"] = artifact_bits
    if metrics:
        out["measured_metrics"] = metrics

    if getattr(trial, "error", "") or getattr(trial, "error_stage", ""):
        stdout = getattr(trial, "stdout", "") or ""
        stderr = getattr(trial, "stderr", "") or ""
        if stdout:
            out["stdout_tail"] = _tail(stdout, 800)
        if stderr:
            out["stderr_tail"] = _tail(stderr, 800)
    return out


def _component_evidence_summary(current_trial: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize reusable code components as evidence, not control flow."""
    trials = _unique_trials(_all_recorded_trials(state) + [current_trial])
    if not trials:
        return {}

    dataset_sample = (state.get("dataset_info") or {}).get("sample_observation")
    measured = [_component_record(t, dataset_sample=dataset_sample) for t in trials if _has_edge_or_artifact(t)]
    measured = [m for m in measured if m]
    measured.sort(
        key=lambda x: (
            0 if x.get("runtime_used") == "onnxruntime" else 1,
            float(x.get("latency_ms") or 1e12),
            -float(x.get("score") or 0.0),
        )
    )

    failures = []
    for t in trials:
        if _has_edge_or_artifact(t):
            continue
        record = _component_record(t, dataset_sample=dataset_sample)
        taxonomy = record.get("failure_taxonomy")
        if taxonomy or record.get("error"):
            failures.append(record)

    return {
        "selected_trial_code_components": _selected_trial_code_components(current_trial),
        "measured_edge_or_artifact_branches": measured[:8],
        "implementation_failures": failures[-8:],
        "use": (
            "Treat this as an inventory of observed code components. Preserve a "
            "working loader/artifact/runtime path when exploiting it; explain "
            "with evidence before replacing it."
        ),
    }


def _selected_trial_code_components(trial: Any) -> Dict[str, Any]:
    """Expose reusable parent code as evidence for child proposals.

    The dossier still does not decide what to inherit.  It only gives the LLM
    the actual component text needed to preserve a working loader/export path.
    """
    variant = getattr(trial, "variant", None)
    if variant is None:
        return {}
    out: Dict[str, Any] = {}
    loader = str(getattr(variant, "loader_code", "") or "").strip()
    train = str(getattr(variant, "train_code", "") or "").strip()
    infer = str(getattr(variant, "infer_code", "") or "").strip()
    loader_contract = _loader_train_contract(trial)
    static_contract = inspect_loader_train_contract(loader, train)
    if static_contract:
        if loader_contract:
            loader_contract["static_code"] = static_contract
        else:
            loader_contract = static_contract
    if loader_contract:
        out["loader_train_contract"] = loader_contract

    judgment = getattr(trial, "branch_judgment", None)
    mutation_text = " ".join(
        str(value or "")
        for value in (
            getattr(judgment, "next_mutation", ""),
            getattr(judgment, "reasoning", ""),
        )
    ).lower()
    focused_sources = {
        name
        for name in ("loader.py", "train.py", "infer.py")
        if name in mutation_text
    }
    out["component_hashes"] = {
        name: _source_hash(code)
        for name, code in (
            ("loader.py", loader),
            ("train.py", train),
            ("infer.py", infer),
        )
        if code
    }

    # Keep compact producer/consumer evidence ahead of full source.  Prompt
    # rendering may truncate bulky code, but must not hide the measured data
    # path that explains a local/edge parity failure.
    edge_eval_bits = _code_lines_with_keywords(
        train,
        keywords=(
            "edge_eval_manifest",
            "edge_eval_payload",
            "sample_ids",
            "split_manifest_hash",
        ),
        limit=4200,
    )
    if edge_eval_bits:
        out["train_edge_eval_excerpt"] = edge_eval_bits
    infer_edge_eval_bits = _code_lines_with_keywords(
        infer,
        keywords=(
            "EDGE_EVAL_MANIFEST",
            "edge_eval_manifest",
            "payload_path",
            "sample_ids",
            "source_ids",
            "X_eval",
            "y_eval",
            "scaler",
            "normalize",
            "transform",
        ),
        limit=4200,
    )
    if infer_edge_eval_bits:
        out["infer_edge_eval_excerpt"] = infer_edge_eval_bits
    if loader:
        key = "loader_py_source" if "loader.py" in focused_sources else "loader_py_excerpt"
        out[key] = _code_excerpt(
            loader,
            limit=14000 if "loader.py" in focused_sources else 2600,
        )
    if train and "train.py" in focused_sources:
        out["train_py_source"] = _code_excerpt(train, limit=18000)
    train_bits = _code_lines_with_keywords(
        train,
        keywords=(
            "outputs/best",
            "edge_eval_manifest",
            "edge_eval_payload",
            "load_train_val",
            "load_test",
            "torch.onnx.export",
            "torch.save",
            "joblib.dump",
            "pickle.dump",
            "json.dumps",
        ),
        limit=1800,
    )
    if train_bits and "train.py" not in focused_sources:
        out["train_export_excerpt"] = train_bits
    if infer and "infer.py" in focused_sources:
        out["infer_py_source"] = _code_excerpt(infer, limit=14000)
    infer_bits = _code_lines_with_keywords(
        infer,
        keywords=(
            "outputs/best",
            "inferencesession",
            "onnxruntime",
            "torch.load",
            "joblib.load",
            "pickle.load",
        ),
        limit=1800,
    )
    if infer_bits and "infer.py" not in focused_sources:
        out["infer_artifact_excerpt"] = infer_bits
    return out


def _source_hash(code: str) -> str:
    import hashlib

    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _loader_train_contract(trial: Any) -> Dict[str, Any]:
    """Return the measured loader interface without sample values."""
    report = getattr(trial, "verification_report", None)
    for item in reversed(list(getattr(report, "evidence", []) or [])):
        if getattr(item, "probe_id", "") != "component_roundtrip":
            continue
        protocol = dict(getattr(item, "protocol", {}) or {})
        if protocol.get("boundary") != "loader_to_train":
            continue
        summary = protocol.get("loader_return_summary")
        if not isinstance(summary, dict):
            continue
        return {
            "evidence_id": str(getattr(item, "id", "") or ""),
            "outcome": str(getattr(item, "outcome", "") or ""),
            "return": _compact_type_contract(summary),
        }
    return {}


def _compact_type_contract(summary: Dict[str, Any], *, depth: int = 0) -> Dict[str, Any]:
    """Keep container shape and mapping keys, never sampled values."""
    if depth > 3:
        return {"type": str(summary.get("type") or "unknown")}
    kind = str(summary.get("type") or "unknown")
    out: Dict[str, Any] = {"type": kind}
    count = summary.get("len")
    if isinstance(count, int):
        out["count"] = count
    keys = summary.get("keys")
    if isinstance(keys, list):
        out["keys"] = [str(key) for key in keys[:40]]
    items = summary.get("items")
    if kind in {"tuple", "list"} and isinstance(items, list) and items:
        compact = [
            _compact_type_contract(item, depth=depth + 1)
            for item in items[:16]
            if isinstance(item, dict)
        ]
        if kind == "list":
            if compact:
                out["item"] = compact[0]
        elif compact:
            out["items"] = compact
    elif kind == "dict" and isinstance(items, dict):
        out["value_types"] = {
            str(key): str(value.get("type") or "unknown")
            for key, value in list(items.items())[:40]
            if isinstance(value, dict)
        }
    return out


def _compact_evidence_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Bound probe evidence while preserving measured interface contracts."""
    out: Dict[str, Any] = {}
    for key, value in protocol.items():
        if key == "loader_return_summary" and isinstance(value, dict):
            out[key] = _compact_type_contract(value)
        else:
            out[key] = _json_safe(value, limit=600)
    return out


def _component_record(trial: Any, *, dataset_sample: Any = None) -> Dict[str, Any]:
    variant = getattr(trial, "variant", None)
    runtime = getattr(trial, "runtime_report", None)
    edge = getattr(trial, "edge_metrics", None)
    artifact = getattr(trial, "artifact_contract", None)
    record: Dict[str, Any] = {
        "trial_id": getattr(trial, "trial_id", ""),
        "parent_trial_id": getattr(variant, "parent_trial_id", "") if variant else "",
        "model_name": getattr(variant, "model_name", "") if variant else "",
        "score": getattr(trial, "score", 0.0),
        "is_feasible": getattr(trial, "is_feasible", False),
        "progress": getattr(getattr(trial, "crafting_progress", None), "value", ""),
        "stage_reached": getattr(getattr(trial, "stage_reached", None), "value", ""),
        "changed_components": getattr(variant, "changed_components", []) if variant else [],
        "inherited_components": getattr(variant, "inherited_components", []) if variant else [],
        "failure_taxonomy": getattr(trial, "failure_taxonomy", ""),
        "next_search_hint": getattr(trial, "next_search_hint", ""),
        "error": _tail(getattr(trial, "error", "") or "", 500),
    }
    local = _metrics_summary(trial).get("local")
    if local:
        record["local_metrics"] = local
    if edge:
        record["latency_ms"] = getattr(edge, "latency_ms", None)
        record["memory_mb"] = getattr(edge, "memory_mb", None)
    if runtime:
        record["runtime_used"] = getattr(runtime, "runtime_used", None)
        record["runtime_provider"] = getattr(runtime, "runtime_provider", None)
        record["artifact_used"] = getattr(runtime, "artifact_used", None)
        record["fallback_reason"] = getattr(runtime, "fallback_reason", None)
    if artifact:
        record["primary_artifact"] = getattr(artifact, "primary_artifact", None)
    observations = getattr(trial, "observations", {}) or {}
    sample_obs = observations.get("sample_observation") if isinstance(observations, dict) else {}
    if not sample_obs:
        sample_obs = _extract_sample_observation(getattr(trial, "failure_context", {}) or {})
    sample_obs = _more_informative_sample(sample_obs, dataset_sample)
    sample_summary = _sample_observation_summary(sample_obs)
    if sample_summary:
        record["sample_summary"] = sample_summary
    record["component_note"] = _component_note(record)
    return {k: v for k, v in record.items() if v not in (None, "", [], {})}


def _component_note(record: Dict[str, Any]) -> str:
    notes: List[str] = []
    if record.get("stage_reached") == "edge_benchmark" or record.get("latency_ms") is not None:
        notes.append("loader/train/export/infer reached edge benchmark")
    if record.get("is_feasible"):
        notes.append("constraints satisfied; preserve working data/artifact/runtime path unless evidence says otherwise")
    runtime = str(record.get("runtime_used") or "")
    provider = str(record.get("runtime_provider") or "")
    if runtime == "torch" or "fallback" in provider.lower():
        notes.append("edge ran through fallback runtime; useful evidence, but do not treat slow latency as model-family proof")
    if record.get("primary_artifact") or record.get("artifact_used"):
        notes.append(f"artifact path observed: {record.get('artifact_used') or record.get('primary_artifact')}")
    sample_summary = record.get("sample_summary") or {}
    if sample_summary:
        notes.append(f"sample evidence: {sample_summary}")
        if sample_summary.get("unique_labels_observed") == 1:
            if sample_summary.get("population_representativeness") == "bounded_observation":
                notes.append("bounded label evidence does not describe the full dataset distribution")
            else:
                notes.append("classification label evidence is suspicious; preserve proven runtime path but re-check label extraction before trusting quality")
    taxonomy = str(record.get("failure_taxonomy") or "")
    if taxonomy:
        notes.append(f"failure evidence: {taxonomy}")
    return "; ".join(notes)


def _extract_sample_observation(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Find the sample observation already produced by loader/debug probes."""
    if not isinstance(ctx, dict):
        return {}
    loader = ctx.get("loader_smoke")
    if isinstance(loader, dict):
        direct = loader.get("sample_observation")
        if isinstance(direct, dict):
            return direct
        probe = loader.get("loader_return_probe")
        if isinstance(probe, dict) and isinstance(probe.get("sample_observation"), dict):
            return probe["sample_observation"]
    diagnostics = ctx.get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("sample_observation"), dict):
        return diagnostics["sample_observation"]
    repair = ctx.get("repair_diagnosis")
    if isinstance(repair, dict):
        evidence = repair.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("sample_observation"), dict):
            return evidence["sample_observation"]
    return {}


def _sample_observation_summary(obs: Any) -> Dict[str, Any]:
    """Compact loader evidence that matters for preserving or mutating code."""
    if not isinstance(obs, dict):
        return {}
    label = obs.get("label_summary") if isinstance(obs.get("label_summary"), dict) else {}
    input_summary = obs.get("input_summary") if isinstance(obs.get("input_summary"), dict) else {}
    split_counts = obs.get("split_counts") if isinstance(obs.get("split_counts"), dict) else {}
    out: Dict[str, Any] = {}
    if "num_observed_samples" in obs:
        out["num_observed_samples"] = obs.get("num_observed_samples")
    for key in ("sampling_strategy", "population_representativeness"):
        if obs.get(key) is not None:
            out[key] = obs.get(key)
    if input_summary:
        for key in ("type", "shape", "dtype"):
            if input_summary.get(key) is not None:
                out[f"input_{key}"] = input_summary.get(key)
        structural = {
            key: input_summary.get(key)
            for key in (
                "source_file",
                "metadata_directives",
                "separator_counts",
                "record_suffix",
                "class_values",
            )
            if input_summary.get(key) not in (None, "", [], {})
        }
        if structural:
            out["input_structure"] = _json_safe(structural, limit=1200)
    if label:
        for key in ("num_labels_observed", "unique_labels_observed", "label_counts"):
            if label.get(key) is not None:
                out[key] = label.get(key)
    if split_counts:
        out["split_counts"] = split_counts
    return out


def _more_informative_sample(first: Any, second: Any) -> Dict[str, Any]:
    candidates = [c for c in (first, second) if isinstance(c, dict) and c]
    if not candidates:
        return {}
    return max(candidates, key=_sample_observation_strength)


def _sample_observation_strength(obs: Dict[str, Any]) -> tuple[int, int, int, int]:
    label = obs.get("label_summary") if isinstance(obs.get("label_summary"), dict) else {}
    input_summary = obs.get("input_summary") if isinstance(obs.get("input_summary"), dict) else {}
    try:
        labels = int(label.get("num_labels_observed") or 0)
    except (TypeError, ValueError):
        labels = 0
    try:
        unique_labels = int(label.get("unique_labels_observed") or 0)
    except (TypeError, ValueError):
        unique_labels = 0
    try:
        samples = int(obs.get("num_observed_samples") or 0)
    except (TypeError, ValueError):
        samples = 0
    return (labels, unique_labels, samples, 1 if input_summary else 0)


def _has_edge_or_artifact(trial: Any) -> bool:
    artifact = getattr(trial, "artifact_contract", None)
    return bool(
        getattr(trial, "edge_metrics", None)
        or getattr(getattr(trial, "runtime_report", None), "runtime_used", None)
        or getattr(artifact, "primary_artifact", None)
        or getattr(artifact, "artifacts", None)
    )


def _unique_trials(trials: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for trial in trials:
        trial_id = getattr(trial, "trial_id", "")
        if trial_id and trial_id in seen:
            continue
        if trial_id:
            seen.add(trial_id)
        out.append(trial)
    return out


def _compact_search_history(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    bank = state.get("trial_bank")
    if not bank:
        return []
    try:
        trials = list(bank.get_all())[-8:]
    except Exception:  # noqa: BLE001
        return []
    items = []
    for t in trials:
        bj = getattr(t, "branch_judgment", None)
        items.append(
            {
                "trial_id": getattr(t, "trial_id", ""),
                "parent_trial_id": getattr(getattr(t, "variant", None), "parent_trial_id", ""),
                "model_name": getattr(getattr(t, "variant", None), "model_name", ""),
                "mutation_type": getattr(getattr(t, "variant", None), "mutation_type", ""),
                "score": getattr(t, "score", 0.0),
                "is_feasible": getattr(t, "is_feasible", False),
                "progress": getattr(getattr(t, "crafting_progress", None), "value", ""),
                "stage_reached": getattr(getattr(t, "stage_reached", None), "value", ""),
                "local_metrics": _metrics_summary(t).get("local", {}),
                "edge_metrics": _metrics_summary(t).get("edge", {}),
                "failure_taxonomy": getattr(t, "failure_taxonomy", ""),
                "next_search_hint": getattr(t, "next_search_hint", ""),
                "branch_judgment": bj.model_dump(mode="json") if bj else None,
            }
        )
    return items



def _trial_compact(t: Any) -> Dict[str, Any]:
    variant = getattr(t, "variant", None)
    bj = getattr(t, "branch_judgment", None)
    return {
        "trial_id": getattr(t, "trial_id", ""),
        "parent_trial_id": getattr(variant, "parent_trial_id", ""),
        "model_name": getattr(variant, "model_name", ""),
        "mutation_type": getattr(variant, "mutation_type", ""),
        "changed_components": getattr(variant, "changed_components", []),
        "score": getattr(t, "score", 0.0),
        "is_feasible": getattr(t, "is_feasible", False),
        "progress": getattr(getattr(t, "crafting_progress", None), "value", ""),
        "stage_reached": getattr(getattr(t, "stage_reached", None), "value", ""),
        "metrics": _metrics_summary(t),
        "failure_taxonomy": getattr(t, "failure_taxonomy", ""),
        "next_search_hint": getattr(t, "next_search_hint", ""),
        "branch_judgment": bj.model_dump(mode="json") if bj else None,
    }


def _all_recorded_trials(state: Dict[str, Any]) -> List[Any]:
    bank = state.get("trial_bank")
    if not bank:
        return []
    try:
        return list(bank.get_all())
    except Exception:  # noqa: BLE001
        return []


def _best_seen_summary(current_trial: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    trials = _all_recorded_trials(state) + [current_trial]
    if not trials:
        return {}
    best = max(trials, key=lambda t: float(getattr(t, "score", 0.0) or 0.0))
    return _trial_compact(best)


def _sibling_evidence_summary(current_trial: Any, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    variant = getattr(current_trial, "variant", None)
    parent_id = getattr(variant, "parent_trial_id", "")
    if not parent_id:
        return []
    siblings = []
    for t in _all_recorded_trials(state):
        tv = getattr(t, "variant", None)
        if getattr(t, "trial_id", "") == getattr(current_trial, "trial_id", ""):
            continue
        if getattr(tv, "parent_trial_id", "") == parent_id:
            siblings.append(_trial_compact(t))
    siblings.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return siblings[:6]


def _comparison_evidence_summary(
    current_trial: Any,
    state: Dict[str, Any],
    current_score: float,
) -> Dict[str, Any]:
    """Summarize current-vs-best evidence without making a branch decision."""
    current = _trial_compact(current_trial)
    current["score"] = current_score
    best = _best_seen_summary(current_trial, state)
    siblings = _sibling_evidence_summary(current_trial, state)
    sibling_best = siblings[0] if siblings else {}
    best_score = _safe_float(best.get("score")) if best else 0.0
    sibling_score = _safe_float(sibling_best.get("score")) if sibling_best else 0.0
    score = _safe_float(current_score)
    return {
        "current": _comparison_point(current),
        "best_seen": _comparison_point(best) if best else {},
        "sibling_best": _comparison_point(sibling_best) if sibling_best else {},
        "score_gap_to_best": score - best_score,
        "score_ratio_to_best": (score / best_score) if best_score > 0 else None,
        "score_gap_to_sibling_best": score - sibling_score if sibling_best else None,
        "current_is_best_seen": bool(best and current.get("trial_id") == best.get("trial_id")),
    }


def _comparison_point(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = summary.get("metrics") or {}
    local = metrics.get("local") or {}
    edge = metrics.get("edge") or {}
    primary_metric = {}
    for name, value in local.items():
        if isinstance(value, (int, float)):
            primary_metric = {"name": name, "value": value}
            break
    return {
        "trial_id": summary.get("trial_id", ""),
        "parent_trial_id": summary.get("parent_trial_id", ""),
        "model_name": summary.get("model_name", ""),
        "score": _safe_float(summary.get("score")),
        "is_feasible": summary.get("is_feasible", False),
        "primary_metric": primary_metric,
        "latency_ms": edge.get("latency_ms"),
        "runtime_used": edge.get("runtime_used"),
        "failure_taxonomy": summary.get("failure_taxonomy", ""),
        "branch_judgment": summary.get("branch_judgment"),
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _json_safe(value: Any, *, limit: int = 2000) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = str(value)
    if len(text) <= limit:
        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001
            return text
    return text[:limit] + "...(truncated)"


def _code_excerpt(code: str, *, limit: int) -> str:
    code = str(code or "").strip()
    if len(code) <= limit:
        return code
    return code[:limit] + "\n# ... parent component truncated ..."


def _code_lines_with_keywords(code: str, *, keywords: tuple[str, ...], limit: int) -> str:
    lines = str(code or "").splitlines()
    if not lines:
        return ""
    keep: List[int] = []
    lowered = tuple(k.lower() for k in keywords)
    for idx, line in enumerate(lines):
        if any(k in line.lower() for k in lowered):
            keep.extend(range(max(0, idx - 2), min(len(lines), idx + 3)))
    if not keep:
        return ""
    seen = set()
    chunks: List[str] = []
    previous = -2
    for idx in keep:
        if idx in seen:
            continue
        if chunks and idx > previous + 1:
            chunks.append("# ...")
        chunks.append(lines[idx])
        seen.add(idx)
        previous = idx
    text = "\n".join(chunks).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n# ... parent component truncated ..."


def _tail(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[-limit:]
