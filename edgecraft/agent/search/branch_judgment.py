"""Branch-level semantic judgment for code-space iterative search.

This module is intentionally small.  It records the one piece of information
that deterministic metrics cannot express cleanly: whether a measured code
branch is still worth expanding.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from edgecraft.config.settings import settings
from edgecraft.agent.search.trial_dossier import build_trial_dossier, trial_dossier_to_prompt_text


class BranchJudgment(BaseModel):
    """LLM/fallback judgment about whether a code-space branch deserves expansion."""

    judgment: str = Field(
        "unknown",
        description="promising_success | promising_failure | weak_branch | dead_end | unknown",
    )
    should_expand: bool = Field(
        False,
        description="Whether the iterative tree should consider expanding this branch.",
    )
    expand_mode: str = Field(
        "abandon",
        description="explore | exploit | repair | pivot | abandon | unknown",
    )
    priority: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="LLM-provided parent-selection priority. This is not a metric formula.",
    )
    reasoning: str = Field("", description="Short reason grounded in measured evidence.")
    next_mutation: str = Field("", description="Concrete next code-space mutation if expanded.")
    evidence_cited: List[str] = Field(
        default_factory=list,
        description="Specific evidence items used by the judgment.",
    )
    risk: str = Field("", description="Main risk of expanding this branch.")
    source: str = Field(
        "fallback",
        description="llm | fallback | disabled; fallback is compatibility, not design intelligence.",
    )

    def to_prompt_text(self) -> str:
        bits = [
            f"judgment={self.judgment}",
            f"should_expand={self.should_expand}",
            f"expand_mode={self.expand_mode}",
            f"priority={self.priority:.2f}",
            f"source={self.source}",
        ]
        if self.next_mutation:
            bits.append(f"next_mutation={self.next_mutation}")
        if self.reasoning:
            bits.append(f"reason={self.reasoning[:240]}")
        if self.evidence_cited:
            bits.append("evidence=" + "; ".join(self.evidence_cited[:4]))
        if self.risk:
            bits.append(f"risk={self.risk[:160]}")
        return "branch_judgment: " + " | ".join(bits)


class BranchSelection(BaseModel):
    """Auditable result of choosing one parent from the current live set."""

    selected_node_id: str
    reasoning: str = ""
    evidence_cited: List[str] = Field(default_factory=list)
    candidate_node_ids: List[str] = Field(default_factory=list)
    source: str = Field(
        "llm",
        description="llm | heuristic | single_candidate | explicit_continuation",
    )


def validate_branch_selection(
    raw: Dict[str, Any],
    *,
    candidate_ids: List[str],
    evidence_by_node: Dict[str, set[str]],
) -> BranchSelection:
    """Validate that an LLM choice cannot escape the tree or invent evidence."""
    selected = str(raw.get("selected_node_id") or "")
    if selected not in candidate_ids:
        raise ValueError("LLM branch selection named a node outside the eligible live set")
    cited = [str(item) for item in (raw.get("evidence_cited") or []) if str(item)]
    allowed = evidence_by_node.get(selected, set())
    if (allowed and not cited) or (cited and not set(cited).issubset(allowed)):
        raise ValueError(
            "LLM branch selection must cite only Evidence IDs from the selected dossier"
        )
    return BranchSelection(
        selected_node_id=selected,
        reasoning=str(raw.get("reasoning") or ""),
        evidence_cited=cited[:12],
        candidate_node_ids=candidate_ids,
        source="llm",
    )


def select_live_branch(
    candidates: List[Any],
    trial_bank: Any,
    state: Dict[str, Any],
) -> BranchSelection:
    """Let the LLM select one eligible parent from a bounded evidence summary.

    The tree determines eligibility; the LLM can only choose one ID from that
    live set.  It cannot admit, prune, score, or alter verifier evidence.
    """
    candidate_ids = [str(node.node_id) for node in candidates]
    if not candidate_ids:
        raise ValueError("cannot select a branch from an empty live set")
    if len(candidate_ids) == 1:
        return BranchSelection(
            selected_node_id=candidate_ids[0],
            reasoning="Only one tree node is currently eligible for expansion.",
            candidate_node_ids=candidate_ids,
            source="single_candidate",
        )
    from langchain_core.messages import HumanMessage, SystemMessage
    from edgecraft.utils.llm import create_chat_llm, llm_credentials_available

    if not llm_credentials_available():
        raise RuntimeError(
            "paper-profile live-branch selection requires configured LLM credentials"
        )

    summaries: List[Dict[str, Any]] = []
    evidence_by_node: Dict[str, set[str]] = {}
    for node in candidates:
        trial = trial_bank.get(node.node_id) if trial_bank is not None else None
        if trial is None:
            summaries.append(
                {
                    "node_id": node.node_id,
                    "status": node.status.value,
                    "depth": node.depth,
                    "children": len(node.children_ids),
                    "role": "planning_root",
                }
            )
            evidence_by_node[node.node_id] = set()
            continue
        dossier = build_trial_dossier(
            trial,
            state,
            score=float(getattr(trial, "score", 0.0) or 0.0),
            is_feasible=bool(getattr(trial, "is_feasible", False)),
            constraint_violations=list(
                getattr(trial, "constraint_violations", []) or []
            ),
        )
        dossier_text = trial_dossier_to_prompt_text(dossier, limit=4500)
        evidence_ids = set(
            re.findall(r"\b(?:ev|rule|obs|cal)_[A-Za-z0-9]+\b", dossier_text)
        )
        evidence_by_node[node.node_id] = evidence_ids
        summaries.append(
            {
                "node_id": node.node_id,
                "status": node.status.value,
                "depth": node.depth,
                "children": len(node.children_ids),
                "dossier": json.loads(dossier_text)
                if not dossier_text.endswith("...(trial dossier truncated)")
                else dossier_text,
            }
        )

    prompt = f"""You are EdgeCraft's live-branch selector.

Choose exactly one parent from the eligible live set below. Use verified
quality progress, signed SLO gap/slack, failure evidence, lineage effects, and
remaining child capacity. Prefer a concrete evidence-backed continuation over
a fresh restart. Verification remains the sole accept/reject authority: you
only choose where the next executable child should be proposed.

Return JSON only:
{{
  "selected_node_id": "one exact eligible node_id",
  "reasoning": "one concise evidence-grounded reason",
  "evidence_cited": ["exact Evidence IDs from the selected dossier"]
}}

ELIGIBLE LIVE SET:
{json.dumps(summaries, ensure_ascii=False, indent=2, default=str)}
"""
    llm = create_chat_llm(temperature=0.0, purpose="branch_selection")
    response = llm.invoke(
        [
            SystemMessage(
                content="Select one EdgeCraft live tree branch using only supplied evidence."
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = re.sub(r"```(?:json)?", "", str(response.content).strip()).strip("` \n")
    raw = json.loads(content)
    return validate_branch_selection(
        raw,
        candidate_ids=candidate_ids,
        evidence_by_node=evidence_by_node,
    )


def judge_branch(
    trial: Any,
    state: Dict[str, Any],
    *,
    score: float,
    is_feasible: bool,
    constraint_violations: Optional[List[str]] = None,
) -> BranchJudgment:
    """Return a branch judgment.

    The paper profile is fail-closed: a missing or failed semantic judge cannot
    silently become a heuristic branch policy.  The explicitly selected
    offline profile retains the deterministic compatibility heuristic.
    """
    mode = str(settings.BRANCH_JUDGMENT_MODE or "").strip().lower()
    strict = mode in {"llm_strict", "strict"}
    if mode in {"off", "disabled", "0", "false"}:
        return BranchJudgment(
            judgment="unknown",
            should_expand=False,
            expand_mode="unknown",
            priority=0.0,
            reasoning="Branch judgment disabled.",
            source="disabled",
        )
    from edgecraft.utils.llm import llm_credentials_available

    if mode in {"llm", "auto", "llm_strict", "strict"} and llm_credentials_available():
        try:
            return _llm_judge_branch(
                trial,
                state,
                score=score,
                is_feasible=is_feasible,
                constraint_violations=constraint_violations or [],
            )
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise RuntimeError(
                    "paper-profile branch judgment failed; refusing an unreported heuristic fallback"
                ) from exc
            logger.warning(f"Branch judgment LLM failed; using fallback: {exc}")
    elif strict:
        raise RuntimeError(
            "paper-profile branch judgment requires configured LLM credentials; "
            "select --profile offline only for mechanism smoke tests"
        )
    return fallback_branch_judgment(
        trial,
        score=score,
        is_feasible=is_feasible,
        constraint_violations=constraint_violations or [],
    )


def fallback_branch_judgment(
    trial: Any,
    *,
    score: float,
    is_feasible: bool,
    constraint_violations: Optional[List[str]] = None,
) -> BranchJudgment:
    """Compatibility judgment used when LLM scoring is disabled/unavailable.

    This is deliberately conservative and marked as fallback.  It should keep
    existing behavior alive, not pretend to be the agent's design judgment.
    """
    progress = getattr(getattr(trial, "crafting_progress", None), "value", "") or str(
        getattr(trial, "crafting_progress", "") or ""
    )
    has_metrics = bool(getattr(getattr(trial, "local_metrics", None), "all_metrics", None))
    has_edge = getattr(trial, "edge_metrics", None) is not None
    has_artifact = bool(getattr(trial, "artifact_paths", None)) or bool(
        getattr(getattr(trial, "artifact_contract", None), "primary_artifact", "")
    )
    repair_diag = (getattr(trial, "failure_context", {}) or {}).get("repair_diagnosis") or {}
    next_hint = getattr(trial, "next_search_hint", "") or repair_diag.get("next_search_hint", "")

    if is_feasible:
        return BranchJudgment(
            judgment="promising_success",
            should_expand=True,
            expand_mode="exploit",
            priority=max(0.60, min(0.95, float(score or 0.0))),
            reasoning="Feasible measured branch; may be worth exploiting if budget remains.",
            next_mutation=next_hint or "Exploit the measured branch while preserving its working data/artifact path.",
            evidence_cited=["feasible=true", f"score={score:.4f}", progress],
            source="fallback",
        )
    if has_edge or has_metrics or has_artifact or progress in {
        "artifact_exported",
        "edge_evaluated",
        "train_succeeded",
    }:
        return BranchJudgment(
            judgment="promising_failure",
            should_expand=True,
            expand_mode="repair" if next_hint else "pivot",
            priority=0.55,
            reasoning="Failed branch still produced measured evidence; keep it available for code-space evolution.",
            next_mutation=next_hint or "Preserve the useful evidence and mutate the failed component.",
            evidence_cited=[
                f"progress={progress}",
                f"has_metrics={has_metrics}",
                f"has_edge={has_edge}",
                f"has_artifact={has_artifact}",
            ],
            risk="Fallback judgment is coarse; prefer LLM semantic judgment in depth experiments.",
            source="fallback",
        )
    return BranchJudgment(
        judgment="weak_branch",
        should_expand=False,
        expand_mode="abandon",
        priority=0.10,
        reasoning="No meaningful artifact, metric, edge, or progress evidence was recorded.",
        next_mutation=next_hint or "",
        evidence_cited=[f"progress={progress}", f"violations={constraint_violations or []}"],
        source="fallback",
    )


def _llm_judge_branch(
    trial: Any,
    state: Dict[str, Any],
    *,
    score: float,
    is_feasible: bool,
    constraint_violations: List[str],
) -> BranchJudgment:
    from langchain_core.messages import HumanMessage, SystemMessage
    from edgecraft.utils.llm import create_chat_llm

    dossier = build_trial_dossier(
        trial,
        state,
        score=score,
        is_feasible=is_feasible,
        constraint_violations=constraint_violations,
    )
    dossier_text = trial_dossier_to_prompt_text(dossier, limit=18000)
    prompt = f"""You are the semantic branch evaluator for EdgeCraft.

EdgeCraft searches executable code space for edge ML solutions.  Your job is
not to encourage every branch.  Your job is to decide whether this measured
branch deserves one of the next scarce tree-expansion slots.

Use the TrialDossier as the only source of facts.  Compare the current branch
against best_seen, sibling_evidence, comparison_evidence, component_evidence,
lineage_effect, observations, artifacts, metrics, training_trace, debug_history, verification, gap_slack,
compatibility_hits, failure_observations, and the user intent.

Treat lineage_effect as a factual test of the parent mutation hypothesis. A
positive or negative delta is evidence, not an automatic instruction: decide
whether to exploit, change the coherent mutation, or pivot the recipe.
Interpret parent-child quality deltas as paired evidence only when
paired_training_seed is true; otherwise acknowledge training randomness.
Use actual_changed_components and preserved component hashes as counterfactual
evidence.  When a regression appears, do not blame a byte-identical component
unless runtime evidence directly implicates it; inspect the changed producer or
the contract it emitted first.
Treat artifact_effect as the factual outcome of an export/runtime mutation. If
code changed but deployed_graph_changed is false, the declared mutation did not
change the measured artifact. Do not repeat the same ineffective mutation;
preserve working components and propose a materially different executable
boundary while citing the artifact Evidence IDs.

If gap_slack is present, judge whether a child can spend available slack to
close a real gap.  If verification is low-fidelity or uncertain, recommend a
conservative executable child that gathers the missing evidence instead of
inventing a new action type.  If compatibility_hits cite exact device/runtime
facts, do not recommend repeating the same operator/runtime path.  Scope that
fact correctly: it invalidates the matched artifact/runtime predicate, not the
dataset, loader, model family, or every deployment route.  When data and probe
contracts passed and preflight exposes another credible route, a child that
preserves those working components and changes the export/runtime boundary may
still be worth measuring.
Unverified failure_observations are soft evidence only; do not treat repeated
observations as a proven incompatibility rule.
Treat declared feature provenance and split provenance as scientific-validity
evidence. A metric supported by target-derived inputs or an unrelated split is
not a successful quality result, even when its numeric value is high.
Keep metric roles distinct. local_metrics contains the tree-visible validation
quality used for solution selection. Edge quality from the parity bundle is a
bounded same-sample semantic check; it is not an official held-out score and
does not replace local validation quality. If these values differ materially,
treat the difference as sample-coverage or evaluation-protocol uncertainty.
Do not explain it away as a dataset property or infer a model defect without
supporting evidence.
Keep the solution boundary explicit. Device authentication, network reachability,
and scheduler availability are orchestration facts, not generated solution
components. Never propose changing credentials, bypassing the requested target,
or substituting host-only execution as a code-space repair. Record that external
blocker, then judge any further child by the remaining quality, artifact, and
constraint facts; if no useful code mutation remains, do not expand the branch.

Return JSON only with:
{{
  "judgment": "promising_success|promising_failure|weak_branch|dead_end",
  "should_expand": true,
  "expand_mode": "explore|exploit|repair|pivot|abandon",
  "priority": 0.0,
  "reasoning": "one concise paragraph",
  "next_mutation": "one concrete code-space mutation",
  "evidence_cited": ["specific evidence item"],
  "risk": "main risk"
}}

Judgment contract:
- Judge continuation potential, not whether the mutation that produced the
  current node succeeded.  A falsified mutation can still leave a valuable
  parent when it preserves a proven data/artifact route and supports a
  materially different, evidence-backed child hypothesis.
- should_expand answers one question: "Is a child of this node worth one of
  the remaining measurements?"  After the planning root there is no separate
  diagnostic or fresh-root action: every additional measurement is an
  executable child.  If a different runtime or recipe is worth trying, return
  should_expand=true and express that pivot in next_mutation.  If false,
  next_mutation must be empty.
- promising_success: constraints are satisfied. Usually exploit it while
  preserving the working loader, artifact path, and runtime route.
- promising_failure: constraints are not met, but the branch has real user-data
  evidence and a concrete next mutation worth measuring before better siblings.
- weak_branch: the branch has some evidence, but the next mutation is vague,
  mostly "try harder", worse than available siblings, or not worth the next
  scarce parent slot.
- dead_end: the branch contradicts the dataset/task evidence, fakes or replaces
  user data, depends on unavailable external assets, repeats a bad lineage, or
  has no credible path to improvement.

Keep the data flow clean:
- A dataset with one local labeled split is still usable evidence; trial-local
  train/validation/test splits are valid when labels and examples are real.
- A bounded sample observation describes only the inspected sample.  Do not
  infer that the complete dataset or loader is broken from sample class
  imbalance when the loader contract itself passed; record the uncertainty.
- The requested modality/task is evidence. Do not silently reward a modality
  shift unless the dossier justifies it for the user's goal.
- Component evidence is inventory, not law. Preserve proven loader/export/infer
  components when exploiting; replace them only when the dossier shows why.
- A branch that merely can be repaired is not automatically promising. Say why
  its repair or mutation beats continuing the best measured sibling.
- Conversely, do not abandon the only proven data/artifact route solely because
  its latest quality hypothesis was falsified.  When no better expandable
  sibling exists, one concrete pivot that reuses those working components can
  be a promising_failure worth measuring.
- Treat train timeout, missing required quality metrics, and extreme edge
  latency as strong semantic evidence. A branch with an artifact but no quality
  metric is not a promising_success; it is at best a repair/pivot candidate.
  A branch that repeatedly chooses a heavy model or preprocessing path after
  timeout/high-latency evidence should be weak_branch or dead_end unless the
  next_mutation clearly preserves the real dataset path and pivots to a lighter
  representation/runtime. Do not propose reducing the real training/evaluation
  corpus as the fix; max_samples is only acceptable for smoke/debug evidence.
- For long-run depth experiments, prefer branches that keep proven user-data
  loading and artifact/export pieces while changing the single component that
  the dossier identifies as the bottleneck. Do not reward a fresh restart that
  discards working components without measured justification.
- priority is semantic parent-selection priority, not a metric formula.
- Cite concrete dossier evidence, especially parent/sibling metric and latency
  changes, artifact/runtime evidence, and the exact component to mutate.
- When citing task quality, use the complete tree-visible local validation
  metric. Never substitute the bounded edge parity-bundle metric for it.

DOSSIER:
{dossier_text}
"""
    llm = create_chat_llm(temperature=0.1, purpose="branch_judgment")
    response = llm.invoke(
        [
            SystemMessage(content="You judge EdgeCraft code-space branches from evidence."),
            HumanMessage(content=prompt),
        ]
    )
    content = str(response.content).strip()
    content = re.sub(r"```(?:json)?", "", content).strip("` \n")
    raw = json.loads(content)
    judgment = str(raw.get("judgment") or "unknown")
    should_expand = bool(raw.get("should_expand"))
    expand_mode = str(raw.get("expand_mode") or "unknown")
    priority = _clamp_priority(raw.get("priority"))

    # Keep the BranchJudgment data structure self-consistent.  Do not override
    # should_expand from the LLM: that boolean is the branch decision.  The
    # judgment label is evidence/explanation, not a second control system.
    if judgment == "promising_success" and not is_feasible:
        judgment = "promising_failure"

    return BranchJudgment(
        judgment=judgment,
        should_expand=should_expand,
        expand_mode=expand_mode,
        priority=priority,
        reasoning=str(raw.get("reasoning") or ""),
        next_mutation=str(raw.get("next_mutation") or ""),
        evidence_cited=[str(x) for x in (raw.get("evidence_cited") or [])][:8],
        risk=str(raw.get("risk") or ""),
        source="llm",
    )


def _clamp_priority(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:  # noqa: BLE001
        return 0.5
