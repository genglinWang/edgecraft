"""Reflector node: decides whether to continue searching or terminate.

Termination triggers (in priority order):
1. iteration >= max_iterations                    (budget exhausted)
2. TrialBank convergence detected                 (no improvement for N trials)
3. LLM diagnosis says should_terminate            (called every N iterations or when stuck)
4. Minimum feasible trials threshold              (search found a good-enough solution)

The Scorer node is responsible for updating the TrialBank and best_feasible_trial;
the Reflector only reads them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from edgecraft.agent.contracts import CraftingProgress
from edgecraft.agent.prompts.system import REFLECTOR_DIAGNOSIS_PROMPT, SYSTEM_PROMPT
from edgecraft.agent.search.branch_judgment import judge_branch
from edgecraft.agent.search.prompt_sampler import PromptSampler
from edgecraft.agent.state import AgentState
from edgecraft.config.settings import edgecraft_env, settings
from edgecraft.knowledge.cbr.case_store import Case, CaseStore
from edgecraft.utils.llm import create_chat_llm

# Call LLM diagnosis every N iterations (in between: cheap heuristics only)
_LLM_DIAGNOSIS_EVERY = 5
# If best feasible score >= this threshold, the LLM can decide to terminate early
_EARLY_STOP_SCORE_THRESHOLD = 0.92
# Minimum iterations before early-stop is considered
_MIN_ITERATIONS_FOR_EARLY_STOP = 3


def _get_llm():
    return create_chat_llm(temperature=0.1, purpose="reflector")


def _continue_after_feasible(trial_bank) -> bool:
    """Experiment mode: keep searching after the first feasible artifact."""
    flag = edgecraft_env("CONTINUE_AFTER_FEASIBLE").strip().lower()
    return flag in {"1", "true", "yes"} and trial_bank.get_best_feasible() is not None


def _fixed_trial_budget() -> bool:
    """Keep causal evaluation arms on the same declared trial budget."""
    flag = edgecraft_env("FIXED_TRIAL_BUDGET").strip().lower()
    return flag in {"1", "true", "yes"}


def _llm_diagnose(state: AgentState) -> Dict[str, Any]:
    """Call LLM for a structured diagnosis.  Falls back to {} on any error."""
    user_spec = state.get("user_spec")
    trial_bank = state["trial_bank"]
    search_tree = state["search_tree"]

    sampler = PromptSampler()
    best_node_id = (
        search_tree.best_node().node_id
        if search_tree.best_node()
        else (search_tree._root_id or "")
    )
    prompt_context = sampler.sample(
        trial_bank=trial_bank,
        search_tree=search_tree,
        selected_node_id=best_node_id,
        user_intent=state.get("raw_user_intent", ""),
        user_spec=user_spec,
        runtime_config=state.get("runtime_config"),
        top_k=5,
        failure_k=3,
    )

    prompt_text = REFLECTOR_DIAGNOSIS_PROMPT.format(
        search_context=prompt_context.to_text(),
    )

    llm = _get_llm()
    try:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt_text)]
        )
        content = response.content.strip()
        # Strip markdown fences
        import re
        content = re.sub(r"```(?:json)?", "", content).strip("` \n")
        diagnosis = json.loads(content)
        logger.debug(
            f"Reflector LLM diagnosis: {diagnosis.get('diagnosis')}  "
            f"terminate={diagnosis.get('should_terminate')}"
        )
        return diagnosis
    except Exception as exc:
        logger.warning(f"Reflector LLM diagnosis failed: {exc}")
        return {}


def scorer_node(state: AgentState) -> AgentState:
    """Scorer node: compute trial score, update TrialBank, update best_feasible.

    This node is placed BEFORE the Reflector in the graph so that the Reflector
    always sees an up-to-date TrialBank.
    """
    from edgecraft.agent.search.scorer import Scorer

    if state.get("status") in {"failed", "completed"}:
        return state

    trial = state.get("current_trial_result")
    if trial is None:
        logger.warning("scorer_node called with no current_trial_result.")
        if state.get("status") != "failed":
            state["status"] = "searching"
        return state

    user_spec = state.get("user_spec")
    scorer = Scorer()

    # Compute score without handler (data-driven from UserSpec)
    score = scorer.compute_score(trial, user_spec)
    is_feasible = scorer.is_feasible(trial, user_spec) if user_spec else False
    constraint_check = scorer.check_constraints(trial, user_spec) if user_spec else None

    trial.score = score
    trial.is_feasible = is_feasible
    if constraint_check:
        trial.constraint_violations = constraint_check.violations
        if is_feasible:
            trial.crafting_progress = CraftingProgress.CONSTRAINT_SATISFIED

    trial.branch_judgment = judge_branch(
        trial,
        state,
        score=score,
        is_feasible=is_feasible,
        constraint_violations=trial.constraint_violations,
    )

    # Record measured score + feasibility on the refinement tree.
    # No upward propagation: each node carries its own real measurement.
    search_tree = state["search_tree"]
    metric_dict: Dict[str, float] = {}
    if trial.local_metrics and trial.local_metrics.all_metrics:
        metric_dict.update(trial.local_metrics.all_metrics)
    if trial.edge_metrics:
        if trial.edge_metrics.latency_ms is not None:
            metric_dict["latency_ms"] = trial.edge_metrics.latency_ms
        if trial.edge_metrics.memory_mb is not None:
            metric_dict["memory_mb"] = trial.edge_metrics.memory_mb
    search_tree.record_result(
        trial.trial_id,
        score=score,
        feasible=is_feasible,
        pruned=bool(
            trial.verification_report
            and trial.verification_report.decision == "prune"
        ),
        metrics=metric_dict,
        branch_judgment=trial.branch_judgment,
    )

    # Add to TrialBank
    trial_bank = state["trial_bank"]
    trial_bank.add(trial)
    _record_trial_as_cbr_case(state, trial)

    # Update best feasible
    new_best = False
    if is_feasible:
        best = state.get("best_feasible_trial")
        if best is None or scorer.is_better(trial, best, user_spec):
            state["best_feasible_trial"] = trial
            new_best = True

    # Increment iteration counter
    state["iteration"] = state.get("iteration", 0) + 1
    iter_str = f"{state['iteration']}/{state['max_iterations']}"
    feasible_str = "FEASIBLE ★" if is_feasible else "infeasible"
    best_str = "  [NEW BEST]" if new_best else ""
    violations_str = (
        f"  violations={trial.constraint_violations}" if trial.constraint_violations else ""
    )
    logger.info(
        f"[{iter_str}] {trial.trial_id}  score={score:.4f}  {feasible_str}"
        f"{best_str}{violations_str}"
    )
    state["status"] = "reflecting"
    return state


def _record_trial_as_cbr_case(state: AgentState, trial) -> None:
    """Best-effort CBR recording for both successes and informative failures."""
    try:
        user_spec = state.get("user_spec")
        if not user_spec:
            return
        if not _trial_has_reusable_memory_value(trial):
            return
        metrics: Dict[str, float] = {}
        if trial.local_metrics and trial.local_metrics.all_metrics:
            metrics.update(trial.local_metrics.all_metrics)
        if trial.edge_metrics:
            if trial.edge_metrics.latency_ms is not None:
                metrics["Latency"] = trial.edge_metrics.latency_ms
            if trial.edge_metrics.memory_mb is not None:
                metrics["Memory_mb"] = trial.edge_metrics.memory_mb

        constraints = {}
        for c in user_spec.constraints or []:
            constraints[c.metric] = {
                "comparison": c.comparison,
                "target": c.target,
                "unit": c.unit,
            }

        lessons = "; ".join(
            x for x in [
                f"progress={trial.crafting_progress.value}",
                f"failure={trial.failure_taxonomy}" if trial.failure_taxonomy else "",
                trial.next_search_hint or "",
            ]
            if x
        )
        diagnostics = (trial.failure_context or {}).get("diagnostics") or {}
        dataset_signature = diagnostics.get("dataset_signature") or {}
        layout_contract = diagnostics.get("layout_contract") or ""
        if not layout_contract:
            layout_contract = str(((state.get("dataset_info") or {}).get("format") or ""))
        failure_signature = trial.failure_taxonomy or trial.error_stage or ""
        case = Case(
            id=f"{state.get('run_id', 'run')}_{trial.trial_id}",
            intent=state.get("raw_user_intent", "") or user_spec.description,
            modality=state["modality"],
            task_type=state["task_type"],
            dataset_description=str((state.get("dataset_info") or {}).get("description", "")),
            dataset_signature=dataset_signature,
            layout_contract=str(layout_contract),
            failure_signature=str(failure_signature),
            constraints=constraints,
            target_device=state.get("target_device", ""),
            model_name=trial.variant.model_name,
            model_source=trial.variant.model_family,
            training_config={
                "quant_mode": trial.variant.quant_mode,
                "export_format": trial.variant.export_format,
                "search_dimension": trial.variant.search_dimension,
            },
            optimization_config={
                "crafting_progress": trial.crafting_progress.value,
                "runtime_report": trial.runtime_report.model_dump(mode="json"),
                "artifact_contract": trial.artifact_contract.model_dump(mode="json"),
                "failure_taxonomy": trial.failure_taxonomy,
                "failure_signature": failure_signature,
                "layout_contract": layout_contract,
                "dataset_signature": dataset_signature,
                "patch_type": ";".join(d.patch_summary for d in trial.debug_history[-2:]),
                "next_search_hint": trial.next_search_hint,
                "branch_judgment": trial.branch_judgment.model_dump(mode="json") if trial.branch_judgment else None,
                "memory_admission": _memory_admission_reason(trial),
            },
            metrics=metrics,
            success=bool(trial.is_feasible),
            lessons_learned=lessons,
            source="edgecraft_trial",
        )
        if str(settings.CBR_SCOPE or "").strip().lower() == "tenant":
            CaseStore(tenant_id=state.get("tenant_id") or "default").add_case(case)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"CBR trial recording skipped: {exc}")


def _trial_has_reusable_memory_value(trial) -> bool:
    """Return whether a trial should enter the local CBR memory.

    CBR is a memory of useful evidence, not a transcript of every attempt.
    """
    if bool(getattr(trial, "is_feasible", False)):
        return True

    judgment = getattr(trial, "branch_judgment", None)
    if judgment and getattr(judgment, "source", "") == "llm":
        has_evidence = bool(
            getattr(judgment, "reasoning", "")
            or getattr(judgment, "next_mutation", "")
            or getattr(judgment, "evidence_cited", [])
        )
        return has_evidence and getattr(judgment, "judgment", "") in {
            "promising_failure",
            "weak_branch",
            "dead_end",
        }

    return bool(
        getattr(trial, "next_search_hint", None)
        and (getattr(trial, "failure_taxonomy", None) or getattr(trial, "error_stage", None))
    )


def _memory_admission_reason(trial) -> str:
    if bool(getattr(trial, "is_feasible", False)):
        return "feasible_trial"
    judgment = getattr(trial, "branch_judgment", None)
    if judgment:
        return f"branch_judgment:{getattr(judgment, 'judgment', 'unknown')}"
    if getattr(trial, "next_search_hint", None):
        return "failure_with_next_search_hint"
    return "not_recorded"


def reflector_node(state: AgentState) -> AgentState:
    """Reflector node: check termination conditions and update status."""
    if state.get("status") in {"failed", "completed"}:
        return state
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    trial_bank = state["trial_bank"]

    # ------------------------------------------------------------------
    # 1. Budget exhausted
    # ------------------------------------------------------------------
    if iteration >= max_iterations:
        reason = f"Reached max iterations ({max_iterations})."
        logger.info(f"Reflector: done — {reason}")
        state["status"] = "completed"
        state["messages"].append({
            "role": "assistant",
            "content": f"Search finished. {reason} "
                       f"Best feasible: {_best_summary(state)}",
        })
        return state

    if _fixed_trial_budget():
        state["status"] = "searching"
        return state

    # ------------------------------------------------------------------
    # 2. Convergence heuristic (no LLM)
    # ------------------------------------------------------------------
    continue_after_feasible = _continue_after_feasible(trial_bank)

    if not continue_after_feasible and trial_bank.has_converged(patience=4):
        reason = "No improvement over the last 4 feasible trials (converged)."
        logger.info(f"Reflector: done — {reason}")
        state["status"] = "completed"
        state["messages"].append({
            "role": "assistant",
            "content": f"Search converged. {reason} "
                       f"Best feasible: {_best_summary(state)}",
        })
        return state

    # ------------------------------------------------------------------
    # 3. LLM diagnosis (every _LLM_DIAGNOSIS_EVERY iterations or when stuck)
    # ------------------------------------------------------------------
    run_llm = (iteration % _LLM_DIAGNOSIS_EVERY == 0) or _is_stuck(trial_bank)

    if run_llm and iteration >= _MIN_ITERATIONS_FOR_EARLY_STOP:
        diagnosis = _llm_diagnose(state)
        state["reflector_diagnosis"] = diagnosis

        if diagnosis.get("should_terminate") and continue_after_feasible:
            logger.info(
                "Reflector: continuing after feasible artifact because "
                "EDGECRAFT_CONTINUE_AFTER_FEASIBLE is enabled."
            )
            diagnosis["should_terminate"] = False
            diagnosis["continue_after_feasible"] = True

        if diagnosis.get("should_terminate"):
            reason = diagnosis.get("reasoning", "LLM recommends termination.")
            logger.info(f"Reflector: done (LLM) — {reason}")
            state["status"] = "completed"
            state["messages"].append({
                "role": "assistant",
                "content": f"Search terminated by reflector. {reason} "
                           f"Best feasible: {_best_summary(state)}",
            })
            return state

        # Inject the diagnosis direction into a message so ProposalGenerator
        # can pick it up via the message history (optional, non-critical)
        direction = diagnosis.get("suggested_direction", "")
        if direction:
            state["messages"].append({
                "role": "system",
                "content": f"[Reflector] Suggested direction: {direction}",
            })

    # ------------------------------------------------------------------
    # Continue searching
    # ------------------------------------------------------------------
    state["status"] = "searching"
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_stuck(trial_bank) -> bool:
    """True if last 3+ consecutive trials all failed (score ≈ 0)."""
    recent = trial_bank.recent_scores(n=3)
    if len(recent) < 3:
        return False
    return all(s < 0.01 for s in recent)


def _best_summary(state: AgentState) -> str:
    best = state.get("best_feasible_trial")
    if best is None:
        return "No feasible solution found."
    return (
        f"{best.trial_id} | {best.variant.short_description()} | "
        f"score={best.score:.4f}"
    )
