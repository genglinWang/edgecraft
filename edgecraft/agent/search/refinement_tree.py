"""RefinementTree: LLM-guided iterative refinement tree (formerly MCTSSearchTree).

Why this is *not* MCTS
----------------------
Classical MCTS has four phases — Select / Expand / Simulate / Backpropagate —
where Simulate is a cheap rollout that estimates value via random play. In
EdgeCraft, each evaluation is a real Train + EdgeBenchmark pipeline that costs
minutes-to-hours, so there is no cheap simulation.  The tree therefore
collapses to three concerns:

* **Select**     — pick an evidence-bearing, verifier-live node to refine next.
* **Expand**     — attach k LLM-generated children to that node.
* **Record**     — write the node's measured score and feasibility.

The tree computes an eligibility-only live set. In the paper profile, a bounded
TrialDossier for each live node is passed to the LLM branch selector, whose
choice is validated against that set and its visible evidence IDs. The offline
profile supplies a deterministic score/novelty fallback for device-free smoke
checks. No PUCT, visit counts, or value backup are used: every measurement is
the real signal for that node.

Eager-expansion contract
------------------------
The search controller (`run_agent`) decides *when* to expand.  This module
exposes ``select_for_expansion`` so the controller can ask "what's the best
parent to refine right now?" without ever waiting for sibling jobs to
finish. Branches rejected by an authoritative verifier decision are marked
``PRUNED`` and never selected. A fully measured P2 candidate that misses an
SLO may remain repairable because P2 infeasibility is not an early prune.

API summary
-----------
* ``set_root(variant)``          — install root.
* ``expand(parent, children)``   — attach LLM proposals.
* ``expansion_candidates()``     — eligibility-only live set for LLM selection.
* ``select_for_expansion(...)``  — deterministic offline parent picker.
* ``record_result(node, ...)``   — store score and verifier lifecycle state.
* ``get_node`` / ``get_path_to_root`` / ``get_siblings``
* ``best_node`` / ``size`` / ``summary``
* ``layer_distribution``         — diversity hint for prompt context.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from edgecraft.agent.search.branch_judgment import BranchJudgment
from edgecraft.agent.search.solution_variant import SolutionVariant


class NodeStatus(str, Enum):
    """Lifecycle states for a tree node."""
    PENDING = "pending"           # created, not yet evaluated
    RUNNING = "running"           # submitted to the scheduler
    FEASIBLE = "feasible"         # evaluated; meets hard constraints
    INFEASIBLE = "infeasible"     # evaluated; violates constraints or errored
    PRUNED = "pruned"             # P0/P1 verifier rejected this branch


@dataclass
class TreeNode:
    """A single node in the refinement tree."""
    node_id: str                                  # == SolutionVariant.trial_id
    variant: SolutionVariant
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # Lifecycle / measured outcome
    status: NodeStatus = NodeStatus.PENDING
    score: float = 0.0                            # 0 until evaluated
    metrics: Dict[str, float] = field(default_factory=dict)
    branch_judgment: Optional[BranchJudgment] = None

    # LLM-supplied prior (P1): used for *scheduling priority*, not selection.
    # Higher score == LLM thinks this child is more promising for the device.
    prior_score: float = 0.5

    @property
    def is_evaluated(self) -> bool:
        """Backward-compatible flag for callers that ask "is this node done?"."""
        return self.status in (
            NodeStatus.FEASIBLE,
            NodeStatus.INFEASIBLE,
            NodeStatus.PRUNED,
        )


def _has_llm_judgment(node: TreeNode) -> bool:
    judgment = node.branch_judgment
    return bool(judgment and judgment.source == "llm")


class RefinementTree:
    """LLM-guided iterative refinement tree.

    Selection eligibility is computed over evidence-bearing nodes with
    remaining branching budget and depth. The paper-profile LLM choice is
    implemented outside this container in ``select_live_branch``.
    """

    def __init__(
        self,
        max_depth: int = 6,
        branching_factor: int = 3,
        epsilon: float = 0.15,
        exploration_coeff: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        # Backward compatibility: old call sites pass exploration_coeff from
        # MCTS naming; map it to epsilon-exploration when explicitly provided.
        if exploration_coeff is not None:
            epsilon = float(exploration_coeff)
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.epsilon = epsilon
        self._rng = rng or random.Random(0)
        self._nodes: Dict[str, TreeNode] = {}
        self._root_id: Optional[str] = None
        self.total_evaluations: int = 0

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def set_root(self, variant: SolutionVariant) -> str:
        node = TreeNode(node_id=variant.trial_id, variant=variant, depth=0)
        self._nodes[variant.trial_id] = node
        self._root_id = variant.trial_id
        logger.debug(f"RefinementTree root set: {variant.trial_id}")
        return variant.trial_id

    def restore_results(self, trials: List[Any]) -> int:
        """Rebuild evaluated topology from persisted TrialResult objects.

        TrialBank already owns the durable facts needed to resume a run.  A
        historical planning-root id is not durable, so trials whose recorded
        parent is absent from the bank are attached to the current planning
        root.  Real parent-child links remain unchanged.
        """
        if self._root_id is None:
            raise ValueError("set_root() must be called before restore_results().")

        restored_ids = {
            str(getattr(trial, "trial_id", ""))
            for trial in trials
            if str(getattr(trial, "trial_id", ""))
        }
        for trial in trials:
            trial_id = str(getattr(trial, "trial_id", ""))
            variant = getattr(trial, "variant", None)
            if not trial_id or variant is None or trial_id in self._nodes:
                continue
            metrics: Dict[str, float] = {}
            local_metrics = getattr(trial, "local_metrics", None)
            metrics.update(dict(getattr(local_metrics, "all_metrics", {}) or {}))
            edge_metrics = getattr(trial, "edge_metrics", None)
            if edge_metrics is not None:
                if getattr(edge_metrics, "latency_ms", None) is not None:
                    metrics["latency_ms"] = float(edge_metrics.latency_ms)
                if getattr(edge_metrics, "memory_mb", None) is not None:
                    metrics["memory_mb"] = float(edge_metrics.memory_mb)
            self._nodes[trial_id] = TreeNode(
                node_id=trial_id,
                variant=variant,
                status=self._restored_status(trial),
                score=float(getattr(trial, "score", 0.0) or 0.0),
                metrics=metrics,
                branch_judgment=getattr(trial, "branch_judgment", None),
                prior_score=float(getattr(variant, "prior_score", 0.5) or 0.5),
            )

        for trial_id in restored_ids:
            node = self._nodes.get(trial_id)
            if node is None:
                continue
            recorded_parent = str(getattr(node.variant, "parent_trial_id", "") or "")
            parent_id = recorded_parent if recorded_parent in restored_ids else self._root_id
            node.parent_id = parent_id
            parent = self._nodes[parent_id]
            if trial_id not in parent.children_ids:
                parent.children_ids.append(trial_id)

        for trial_id in restored_ids:
            node = self._nodes.get(trial_id)
            if node is None:
                continue
            depth = 1
            parent_id = node.parent_id
            seen = {trial_id}
            while parent_id and parent_id != self._root_id and parent_id not in seen:
                seen.add(parent_id)
                parent = self._nodes.get(parent_id)
                if parent is None:
                    break
                depth += 1
                parent_id = parent.parent_id
            node.depth = depth

        restored = sum(trial_id in self._nodes for trial_id in restored_ids)
        self.total_evaluations = restored
        if restored:
            logger.info(
                f"RefinementTree restored {restored} evaluated node(s) from TrialBank"
            )
        return restored

    @staticmethod
    def _restored_status(trial: Any) -> NodeStatus:
        """Rebuild verifier terminal state without trusting LLM judgment."""
        report = getattr(trial, "verification_report", None)
        if str(getattr(report, "decision", "") or "") == "prune":
            return NodeStatus.PRUNED
        if bool(getattr(trial, "is_feasible", False)):
            return NodeStatus.FEASIBLE
        return NodeStatus.INFEASIBLE

    def expand(
        self,
        parent_id: str,
        children: List[SolutionVariant],
        priors: Optional[List[float]] = None,
    ) -> List[str]:
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node '{parent_id}' not in tree.")
        parent = self._nodes[parent_id]
        if self._is_in_pruned_branch(parent):
            raise ValueError(
                f"cannot expand verifier-pruned branch: {parent_id}"
            )
        ids: List[str] = []
        for i, child_variant in enumerate(children):
            child_variant.parent_trial_id = parent_id
            p = priors[i] if priors and i < len(priors) else 0.5
            node = TreeNode(
                node_id=child_variant.trial_id,
                variant=child_variant,
                parent_id=parent_id,
                depth=parent.depth + 1,
                prior_score=float(p),
            )
            self._nodes[child_variant.trial_id] = node
            parent.children_ids.append(child_variant.trial_id)
            ids.append(child_variant.trial_id)
            logger.debug(
                f"RefinementTree expand: {parent_id} → {child_variant.trial_id} "
                f"(prior={p:.2f}, depth={node.depth})"
            )
        return ids

    # ------------------------------------------------------------------
    # Selection — score-driven exploitation with early novelty pressure
    # ------------------------------------------------------------------

    def _selection_stage(self) -> str:
        budget_hint = max(4, 3 * max(1, self.branching_factor))
        ratio = min(1.0, self.total_evaluations / float(budget_hint))
        if ratio < 0.33:
            return "explore"
        if ratio < 0.67:
            return "mixed"
        return "exploit"

    def _signature_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self._nodes.values():
            if not node.is_evaluated:
                continue
            sig = node.variant.novelty_signature()
            counts[sig] = counts.get(sig, 0) + 1
        return counts

    def _has_exploit_child(self, node: TreeNode) -> bool:
        for child_id in node.children_ids:
            child = self._nodes.get(child_id)
            if child is None:
                continue
            intent = (getattr(child.variant, "search_intent", "") or "").lower()
            if intent in {"exploit", "repair"}:
                return True
        return False

    def _needs_exploit_child(self, node: TreeNode) -> bool:
        return (
            node.status == NodeStatus.FEASIBLE
            and node.depth < self.max_depth
            and len(node.children_ids) < self.branching_factor
            and not self._has_exploit_child(node)
        )

    def expansion_candidates(self) -> List[TreeNode]:
        """Return the current live set without choosing a parent.

        Selection authority is deliberately separate from eligibility.  The
        paper profile passes this live set to the LLM branch selector; the
        offline profile may rank the same set with the deterministic policy.
        """
        def _has_search_evidence(node: TreeNode) -> bool:
            return bool(node.metrics) or float(node.score or 0.0) > 0.0

        def _judged_expandable(node: TreeNode) -> bool:
            return bool(node.branch_judgment and node.branch_judgment.should_expand)

        def _root_has_remaining_slots(node: TreeNode) -> bool:
            return (
                node.node_id == self._root_id
                and len(node.children_ids) < self.branching_factor
            )

        base_nodes = [
            node
            for node in self._nodes.values()
            if node.depth < self.max_depth
            and len(node.children_ids) < self.branching_factor
            and not self._is_in_pruned_branch(node)
        ]
        root_node = self._nodes.get(self._root_id or "")
        if (
            root_node is not None
            and root_node in base_nodes
            and 0 < len(root_node.children_ids) < self.branching_factor
        ):
            # Complete the initial breadth before comparing measured branches.
            return [root_node]
        return [
            node
            for node in base_nodes
            if (
                (node.node_id == self._root_id and not node.children_ids)
                or (_has_llm_judgment(node) and node.branch_judgment.should_expand)
                or (
                    not _has_llm_judgment(node)
                    and (
                        _root_has_remaining_slots(node)
                        or node.status == NodeStatus.FEASIBLE
                        or (
                            node.status == NodeStatus.INFEASIBLE
                            and (_has_search_evidence(node) or _judged_expandable(node))
                        )
                    )
                )
            )
        ]

    def _selection_value(self, node: TreeNode, signature_counts: Dict[str, int], stage: str) -> float:
        if node.node_id == self._root_id and node.children_ids:
            # The planning root may be used to fill missing initial breadth, but
            # it should not outrank measured code branches once any child has
            # produced evidence.
            return -1.0
        if _has_llm_judgment(node):
            # With LLM branch judgment, measured score and hand-written search
            # bonuses are evidence, not the branch decision.  Keep only tiny
            # deterministic tie-breakers so selection stays stable.
            sig_count = signature_counts.get(node.variant.novelty_signature(), 0)
            novelty = 1.0 / float(1 + sig_count)
            return float(node.branch_judgment.priority or 0.0) + 0.001 * novelty + 0.0001 * node.depth

        sig_count = signature_counts.get(node.variant.novelty_signature(), 0)
        novelty = 1.0 / float(1 + sig_count)
        intent = (getattr(node.variant, "search_intent", "") or "").lower()
        intent_bonus = 0.0
        if stage == "explore" and intent == "explore":
            intent_bonus = 0.10
        elif stage == "exploit" and intent in {"exploit", "repair"}:
            intent_bonus = 0.05
        if stage == "explore":
            novelty_weight = 0.35
        elif stage == "mixed":
            novelty_weight = 0.15
        else:
            novelty_weight = 0.03
        branch_priority = None
        if node.branch_judgment is not None and node.branch_judgment.source != "disabled":
            branch_priority = float(node.branch_judgment.priority or 0.0)
        # Once a branch has produced a feasible artifact, reserve at least one
        # follow-up expansion for exploitation/repair. This keeps early search
        # broad while preventing the tree from endlessly sampling new families
        # after it has found a measured foothold.
        exploit_reserve_bonus = 0.0
        if self._needs_exploit_child(node):
            exploit_reserve_bonus = 0.30 if stage in {"explore", "mixed"} else 0.12
        depth_bonus = 0.01 * min(node.depth, self.max_depth)
        base_value = branch_priority if branch_priority is not None else float(node.score or 0.0)
        return (
            base_value
            + novelty_weight * novelty
            + intent_bonus
            + exploit_reserve_bonus
            + depth_bonus
        )

    def select_for_expansion(self, preferred_node_id: Optional[str] = None) -> Optional[str]:
        """Return the node_id whose subtree should be refined next.

        This is the deterministic selector used by the offline profile. It
        ranks only nodes returned by :meth:`expansion_candidates`; the paper
        profile selects from that same live set in ``select_live_branch``.

        With probability ``epsilon`` and when ≥ 2 candidates exist, pick a
        non-best candidate to escape local optima.

        ``preferred_node_id`` is a one-shot continuation control used by
        reproducible evaluation runs. It bypasses ranking, but never bypasses
        the tree's depth or branching limits.
        """
        if preferred_node_id:
            preferred = self._nodes.get(preferred_node_id)
            if preferred is None:
                raise ValueError(f"continuation parent is absent from the tree: {preferred_node_id}")
            if self._is_in_pruned_branch(preferred):
                raise ValueError(
                    f"continuation parent belongs to a branch pruned by verifier authority: {preferred_node_id}"
                )
            if preferred.depth >= self.max_depth:
                raise ValueError(f"continuation parent reached max depth: {preferred_node_id}")
            if len(preferred.children_ids) >= self.branching_factor:
                raise ValueError(f"continuation parent has no child slot: {preferred_node_id}")
            logger.info(f"RefinementTree select (explicit-continuation): {preferred_node_id}")
            return preferred_node_id

        candidates = self.expansion_candidates()
        if (
            len(candidates) == 1
            and candidates[0].node_id == self._root_id
            and candidates[0].children_ids
        ):
            root_node = candidates[0]
            logger.debug(
                f"RefinementTree select (root-width): {root_node.node_id} "
                f"children={len(root_node.children_ids)}/{self.branching_factor}"
            )
            return root_node.node_id
        if not candidates:
            logger.debug("RefinementTree.select_for_expansion: no eligible parent.")
            return None
        stage = self._selection_stage()
        signature_counts = self._signature_counts()
        candidates.sort(
            key=lambda n: (self._selection_value(n, signature_counts, stage), n.score, n.depth),
            reverse=True,
        )
        if len(candidates) >= 2 and self._rng.random() < self.epsilon:
            chosen = self._rng.choice(candidates[1:])
            logger.debug(
                f"RefinementTree select (epsilon-{stage}): {chosen.node_id} "
                f"score={chosen.score:.4f} value={self._selection_value(chosen, signature_counts, stage):.4f} "
            f"judgment={(chosen.branch_judgment.judgment if chosen.branch_judgment else 'none')}"
            )
            return chosen.node_id
        chosen = candidates[0]
        logger.debug(
            f"RefinementTree select ({stage}): {chosen.node_id} "
            f"score={chosen.score:.4f} value={self._selection_value(chosen, signature_counts, stage):.4f} "
            f"judgment={(chosen.branch_judgment.judgment if chosen.branch_judgment else 'none')}"
        )
        return chosen.node_id

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_result(
        self,
        node_id: str,
        score: float,
        feasible: bool,
        pruned: bool = False,
        metrics: Optional[Dict[str, float]] = None,
        branch_judgment: Optional[BranchJudgment] = None,
    ) -> None:
        """Store measured outcome and authoritative prune state for a node.

        No upward propagation: in this design each node's measurement is the
        real signal for that node, not a Monte-Carlo estimate.
        """
        node = self._nodes.get(node_id)
        if node is None:
            logger.warning(f"record_result: unknown node '{node_id}'")
            return
        node.score = float(score)
        node.metrics = dict(metrics or {})
        node.branch_judgment = branch_judgment
        node.status = (
            NodeStatus.PRUNED
            if pruned
            else NodeStatus.FEASIBLE if feasible else NodeStatus.INFEASIBLE
        )
        self.total_evaluations += 1
        logger.debug(
            f"RefinementTree record_result {node_id}: score={score:.4f} "
            f"feasible={feasible} pruned={pruned} tree_size={len(self._nodes)}"
        )

    # ------------------------------------------------------------------
    # Queries used by PromptSampler / Reflector
    # ------------------------------------------------------------------

    def _is_in_pruned_branch(self, node: TreeNode) -> bool:
        """Return whether this node or any ancestor is verifier-pruned."""
        current: Optional[TreeNode] = node
        seen: set[str] = set()
        while current is not None and current.node_id not in seen:
            seen.add(current.node_id)
            if current.status == NodeStatus.PRUNED:
                return True
            current = self._nodes.get(current.parent_id or "")
        return False

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self._nodes.get(node_id)

    def get_path_to_root(self, node_id: str) -> List[TreeNode]:
        path: List[TreeNode] = []
        current_id: Optional[str] = node_id
        while current_id is not None and current_id in self._nodes:
            path.append(self._nodes[current_id])
            current_id = self._nodes[current_id].parent_id
        path.reverse()
        return path

    def get_siblings(self, node_id: str) -> List[TreeNode]:
        node = self._nodes.get(node_id)
        if node is None or node.parent_id is None:
            return []
        parent = self._nodes.get(node.parent_id)
        if parent is None:
            return []
        return [
            self._nodes[cid]
            for cid in parent.children_ids
            if cid != node_id and self._nodes[cid].is_evaluated
        ]

    def feasible_siblings(self, node_id: str) -> List[TreeNode]:
        return [s for s in self.get_siblings(node_id) if s.status == NodeStatus.FEASIBLE]

    def failed_siblings(self, node_id: str) -> List[TreeNode]:
        return [
            s for s in self.get_siblings(node_id)
            if s.status in {NodeStatus.INFEASIBLE, NodeStatus.PRUNED}
        ]

    # ------------------------------------------------------------------
    # Hierarchical search-dimension tracking (for prompt diversity hints)
    # ------------------------------------------------------------------

    def layer_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for node in self._nodes.values():
            dim = getattr(node.variant, "search_dimension", "initial")
            dist[dim] = dist.get(dim, 0) + 1
        return dist

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._nodes)

    def best_node(self) -> Optional[TreeNode]:
        evaluated = [n for n in self._nodes.values()
                     if n.status == NodeStatus.FEASIBLE]
        if not evaluated:
            return None
        return max(evaluated, key=lambda n: n.score)

    def summary(self) -> str:
        best = self.best_node()
        best_str = f"{best.node_id} score={best.score:.4f}" if best else "none"
        return (
            f"RefinementTree | nodes={self.size()} "
            f"evaluated={self.total_evaluations} best={best_str}"
        )


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (will be removed once all call sites are
# updated to the new names; kept for one transition window).
# ---------------------------------------------------------------------------

MCTSSearchTree = RefinementTree
MCTSNode = TreeNode


# ---------------------------------------------------------------------------
# Compatibility shim: old code calls ``backpropagate(node_id, score)`` whereas
# new code should call ``record_result(...)`` with explicit feasibility.  Keep
# the old method around so the reflector / scorer can be updated separately.
# ---------------------------------------------------------------------------

def _backpropagate(self: RefinementTree, node_id: str, score: float) -> None:
    self.record_result(node_id, score=score, feasible=score > 0.0)


RefinementTree.backpropagate = _backpropagate  # type: ignore[attr-defined]


# Old single-method alias retained for proposal_generator that still calls
# ``select()``.  Drop after the call site is updated.
RefinementTree.select = RefinementTree.select_for_expansion  # type: ignore[attr-defined]
