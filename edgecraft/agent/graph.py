"""LangGraph definition for the EdgeCraft constraint-aware synthesis agent.

New topology (v2)
-----------------

    ┌──────────────────────────────────────────────────────────────────┐
    │  preflight (intent / dataset / device+runtime) + SSH / run ctrl   │
    └─────────────────────┬────────────────────────────────────────────┘
                          │
                   [proposal_generator]   ← LLM live-branch select + expand
                          │
                   [pipeline_executor]    ← 5-stage pipeline
                          │
                   [scorer]               ← deterministic scoring
                          │
                   [reflector]            ← termination decision
                          │
                 ┌────────┴────────┐
               (done)          (continue)
                 │                 │
               [END]      [proposal_generator]  (loop)

State lifecycle
---------------
- ``run_id`` is assigned once when run_agent() is called.
- ``trial_bank`` and ``search_tree`` are created once and live for the
  entire search run; they are never reset.
- ``pending_variants`` is a queue; ProposalGenerator fills it with k variants
  variants, PipelineExecutor consumes one per loop iteration.
"""
from __future__ import annotations

import uuid
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from langgraph.graph import END, StateGraph
from loguru import logger

from edgecraft.agent.nodes.pipeline_executor import pipeline_executor_node
from edgecraft.agent.nodes.proposal_generator import proposal_generator_node
from edgecraft.agent.nodes.reflector import reflector_node, scorer_node
from edgecraft.agent.search.refinement_tree import RefinementTree
from edgecraft.agent.search.solution_variant import default_variant
from edgecraft.agent.search.trial_bank import TrialBank
from edgecraft.agent.state import AgentState
from edgecraft.agent.workspace.manager import ensure_private_run_workspace, trial_bank_path
from edgecraft.config.profiles import resolved_profile_manifest
from edgecraft.config.settings import settings
from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import RuntimeConfig
from edgecraft.models import ensure_registries_initialized

if TYPE_CHECKING:
    from edgecraft.utils.synth_checks import SynthCheckResult


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_reflector(
    state: AgentState,
) -> Literal["proposal_generator", "__end__"]:
    if state.get("status") in ("completed", "failed"):
        return "__end__"
    return "proposal_generator"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("proposal_generator", proposal_generator_node)
    graph.add_node("pipeline_executor", pipeline_executor_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("reflector", reflector_node)

    graph.set_entry_point("proposal_generator")

    graph.add_edge("proposal_generator", "pipeline_executor")
    graph.add_edge("pipeline_executor", "scorer")
    graph.add_edge("scorer", "reflector")
    graph.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {"proposal_generator": "proposal_generator", "__end__": END},
    )

    return graph


def compile_agent(recursion_limit: int = 200):
    graph = create_agent_graph()
    return graph.compile(
        checkpointer=None,
        interrupt_before=None,
        interrupt_after=None,
    )


# ---------------------------------------------------------------------------
# Initialise helpers
# ---------------------------------------------------------------------------

def _infer_modality_from_dataset(info: Optional[Dict]) -> Optional[Modality]:
    if not info or info.get("status") != "success":
        return None
    try:
        return Modality(info.get("modality", ""))
    except ValueError:
        return None


def _infer_task_type_from_dataset(info: Optional[Dict]) -> Optional[TaskType]:
    if not info or info.get("status") != "success":
        return None
    try:
        return TaskType(info.get("task_type", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _planning_root_trial_id(run_id: str, trials: list[Any]) -> str:
    """Return a stable root id, preserving the missing root of legacy banks."""
    trial_ids = {str(getattr(trial, "trial_id", "") or "") for trial in trials}
    for trial in trials:
        variant = getattr(trial, "variant", None)
        parent_id = str(
            getattr(trial, "parent_trial_id", "")
            or getattr(variant, "parent_trial_id", "")
            or ""
        )
        if parent_id and parent_id not in trial_ids:
            return parent_id
    return f"trial_{uuid.uuid5(uuid.NAMESPACE_URL, f'edgecraft:{run_id}').hex[:8]}"


def run_agent(
    *,
    preflight: SynthCheckResult,
    ssh_key: Optional[str] = None,
    max_iterations: int = 24,
    runtime_config: Optional[RuntimeConfig] = None,
    run_id: Optional[str] = None,
    branching_factor: Optional[int] = None,
    tenant_id: str = "default",
    force_real_edgebench: bool = False,
    continuation_parent_trial_id: Optional[str] = None,
) -> Dict:
    """Run the constraint-aware EdgeCraft agent after successful preflight.

    Args:
        preflight: Successful ``run_synth_preflight`` result (intent, dataset, device).
        ssh_key: Path to SSH private key for edge execution.
        max_iterations: Maximum number of admitted synthesis trials.
        runtime_config: Optional override; default is ``preflight.device_ctx.runtime_config``.
        run_id: Optional existing run id to resume.
        branching_factor: Children per tree expansion; default from settings.
        tenant_id: Logical tenant for batch/scheduler compatibility.

    Returns:
        dict with keys: status, run_id, best_trial, all_trials,
                        iterations, modality, task_type, error.
    """
    if not preflight.success:
        raise ValueError("run_agent requires preflight.success=True")
    if not (preflight.intent and preflight.dataset and preflight.device_ctx):
        raise ValueError("run_agent requires preflight intent, dataset, and device_ctx")
    if run_id is None:
        run_id = f"run_{uuid.uuid4().hex[:12]}"

    target_device = preflight.device_ctx.device_id
    device_ip = preflight.device_ctx.device_ip
    raw_user_intent = preflight.intent.raw_user_intent
    dataset_path = str(Path(preflight.dataset.root_path).resolve())
    logger.info(f"run_agent: run_id={run_id} device={target_device}")

    ensure_registries_initialized()
    effective_branching_factor = (
        int(branching_factor)
        if branching_factor is not None
        else int(settings.TREE_BRANCHING_FACTOR)
    )
    # A run's trial budget must not be made unreachable by a smaller hidden
    # tree-depth limit. Keep the historical floor for short runs while allowing
    # a single lineage to consume the requested long-run budget.
    effective_max_depth = max(6, int(max_iterations))

    # Dataset + UserSpec from preflight (no re-analysis, no duplicate parsing)
    dataset_info = copy.deepcopy(preflight.dataset.info) if preflight.dataset.info else {}
    if preflight.dataset.config_path:
        dataset_info["config_path"] = preflight.dataset.config_path
    parsed_user_spec = preflight.user_spec

    # Determine modality and task_type.  preflight.user_spec is already
    # reconciled with dataset evidence, so it can correct weak intent-only
    # parses such as tabular classification drifting into vision.
    modality: Modality = (
        (parsed_user_spec and parsed_user_spec.input_type)
        or _infer_modality_from_dataset(dataset_info)
        or Modality.VISION
    )
    task_type: TaskType = (
        (parsed_user_spec and parsed_user_spec.task_type)
        or _infer_task_type_from_dataset(dataset_info)
        or TaskType.OBJECT_DETECTION
    )

    if runtime_config is None:
        runtime_config = preflight.device_ctx.runtime_config
    if runtime_config is None:
        raise ValueError(
            "run_agent requires preflight.device_ctx.runtime_config "
            "(from run_synth_preflight) or an explicit runtime_config= override."
        )

    # ------------------------------------------------------------------
    # Persistent trial bank and stable planning root
    # ------------------------------------------------------------------
    tenant_id = tenant_id or "default"
    persist_path = trial_bank_path(run_id, tenant_id=tenant_id)
    ensure_private_run_workspace(run_id, tenant_id=tenant_id)
    profile_manifest_path = persist_path.parent / "run_profile.json"
    profile_manifest_path.write_text(
        json.dumps(
            resolved_profile_manifest(settings, settings.PROFILE),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trial_bank = TrialBank(run_id=run_id, persist_path=persist_path)
    existing_trials = trial_bank.get_all()

    search_tree = RefinementTree(
        max_depth=effective_max_depth,
        branching_factor=effective_branching_factor,
        epsilon=0.15,
    )
    root_variant = default_variant(modality, task_type)
    root_variant.trial_id = _planning_root_trial_id(run_id, existing_trials)
    if runtime_config and "onnx" not in runtime_config.allowed_export_formats():
        root_variant.export_format = runtime_config.allowed_export_formats()[0]
    search_tree.set_root(root_variant)
    search_tree.restore_results(existing_trials)
    resumed_iterations = trial_bank.count()

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    exploration = (dataset_info or {}).get("exploration_report") or ""
    initial_state: AgentState = {
        # User inputs
        "raw_user_intent": raw_user_intent,
        "dataset_path": dataset_path,
        "dataset_report": exploration if isinstance(exploration, str) else str(exploration),
        "dataset_info": dataset_info if dataset_info else None,
        "target_device": target_device,
        "device_ip": device_ip,
        "ssh_key": ssh_key,
        "docker_image": runtime_config.docker_image,
        # Parsed specs
        "user_spec": parsed_user_spec,
        "modality": modality,
        "task_type": task_type,
        "runtime_config": runtime_config,
        # Search identity
        "run_id": run_id,
        "tenant_id": tenant_id,
        "continuation_parent_trial_id": continuation_parent_trial_id,
        # Constraint-aware tree objects
        "trial_bank": trial_bank,
        "search_tree": search_tree,
        "best_feasible_trial": trial_bank.get_best_feasible(
            require_evaluation_valid=bool(settings.REQUIRE_EDGE_QUALITY_PARITY)
        ),
        # Per-iteration
        "pending_variants": [],
        "current_variant": None,
        "current_trial_result": None,
        "reflector_diagnosis": None,
        "branch_selection_history": [],
        "l1_forced_audit_trial_ids": [],
        # Control
        "iteration": resumed_iterations,
        "max_iterations": max_iterations,
        "branching_factor": effective_branching_factor,
        "force_real_edgebench": bool(force_real_edgebench),
        "status": "searching",
        "error": None,
        # Conversation
        "messages": [{"role": "user", "content": raw_user_intent}],
    }

    if resumed_iterations >= max_iterations:
        logger.info(
            f"run_agent: run {run_id} already has {resumed_iterations} trial(s); "
            f"requested total budget is {max_iterations}"
        )
        initial_state["status"] = "completed"
        final_state = initial_state
    else:
        agent = compile_agent(recursion_limit=max_iterations * 5 + 20)
        config = {"recursion_limit": max_iterations * 5 + 20}
        logger.debug(
            f"Launching agent with max_iterations={max_iterations} "
            f"resumed_iterations={resumed_iterations}"
        )
        final_state = agent.invoke(initial_state, config=config)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_trials = final_state["trial_bank"].get_all()
    best_feasible = final_state.get("best_feasible_trial") or final_state["trial_bank"].get_best_feasible(
        require_evaluation_valid=bool(settings.REQUIRE_EDGE_QUALITY_PARITY)
    )
    best_verified_fallback = (
        None
        if best_feasible is not None
        else final_state["trial_bank"].get_best_verified(final_state.get("user_spec"))
    )
    best = best_feasible or best_verified_fallback
    total_trials = len(all_trials)
    real_edge_measured_trials = sum(
        1 for t in all_trials
        if t.stage_reached.value == "edge_benchmark" and t.edge_metrics is not None and t.error is None
    )
    surrogate_only_trials = sum(
        1 for t in all_trials
        if t.edge_metrics is not None and t.stage_reached.value != "edge_benchmark"
    )
    total_debug_attempts = sum(int(getattr(t, "debug_attempts", 0)) for t in all_trials)
    avg_trial_wall_s = (
        sum(float(getattr(t, "duration_seconds", 0.0)) for t in all_trials) / total_trials
        if total_trials > 0 else 0.0
    )
    failure_breakdown: Dict[str, int] = {}
    for t in all_trials:
        if not t.error:
            continue
        stage = (t.error_stage or "unknown_stage").strip() or "unknown_stage"
        failure_breakdown[stage] = failure_breakdown.get(stage, 0) + 1
    failed_status = final_state.get("status", "unknown")
    error_message = final_state.get("error")
    if failed_status == "completed" and best_verified_fallback is not None:
        failed_status = "completed_infeasible"
    elif failed_status == "completed" and best is None:
        failed_status = "failed"
        if not error_message:
            error_message = "No feasible solution found under current constraints."
    measurement_summary = {
        "force_real_edgebench": bool(force_real_edgebench),
        "total_trials": total_trials,
        "real_edge_measured_trials": real_edge_measured_trials,
        "surrogate_only_trials": surrogate_only_trials,
        "total_debug_attempts": total_debug_attempts,
        "avg_trial_wall_s": avg_trial_wall_s,
        "failure_breakdown": failure_breakdown,
    }

    return {
        "status": failed_status,
        "run_id": run_id,
        "tenant_id": tenant_id,
        "best_trial": best.model_dump(mode="json") if best else None,
        "selection_kind": (
            "feasible" if best_feasible is not None
            else "best_p2_verified_fallback" if best_verified_fallback is not None
            else "none"
        ),
        "constraints_satisfied": best_feasible is not None,
        "remaining_gaps": (
            [
                item.model_dump(mode="json")
                for item in best_verified_fallback.gap_slack
                if item.known and float(item.gap or 0.0) > 0.0
            ]
            if best_verified_fallback is not None
            else []
        ),
        "all_trials": [t.trial_id for t in all_trials],
        "iterations": final_state.get("iteration", 0),
        "modality": modality.value,
        "task_type": task_type.value,
        "dataset_info": dataset_info,
        "measurement_summary": measurement_summary,
        "trial_bank_path": str(persist_path),
        "profile_manifest_path": str(profile_manifest_path),
        "error": error_message,
        "reflector_diagnosis": final_state.get("reflector_diagnosis"),
        "branch_selection_history": final_state.get("branch_selection_history") or [],
    }
