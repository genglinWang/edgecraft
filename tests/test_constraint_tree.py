import pytest

from edgecraft.agent.search.branch_judgment import (
    BranchJudgment,
    validate_branch_selection,
)
from edgecraft.agent.search.refinement_tree import NodeStatus, RefinementTree
from edgecraft.agent.search.solution_variant import SolutionVariant, default_variant
from edgecraft.agent.workspace.manager import WorkspaceManager
from edgecraft.core.modality import Modality, TaskType


def variant(trial_id: str, *, intent: str = "explore") -> SolutionVariant:
    return SolutionVariant(
        trial_id=trial_id,
        modality=Modality.VISION,
        task_type=TaskType.CLASSIFICATION,
        model_name=trial_id,
        model_family="test_family",
        train_code="print('train')\n",
        infer_code="print('infer')\n",
        search_intent=intent,
    )


def llm_judgment(*, expand: bool, priority: float) -> BranchJudgment:
    return BranchJudgment(
        judgment="promising_success" if expand else "dead_end",
        should_expand=expand,
        expand_mode="exploit" if expand else "abandon",
        priority=priority,
        evidence_cited=["ev_a"],
        source="llm",
    )


def test_live_set_respects_branch_limits_and_llm_branch_judgment() -> None:
    tree = RefinementTree(max_depth=3, branching_factor=2, epsilon=0.0)
    root = default_variant(Modality.VISION, TaskType.CLASSIFICATION)
    root.trial_id = "root"
    tree.set_root(root)
    tree.expand("root", [variant("keep"), variant("stop")])
    tree.record_result(
        "keep",
        score=0.4,
        feasible=False,
        metrics={"Accuracy": 0.4},
        branch_judgment=llm_judgment(expand=True, priority=0.8),
    )
    tree.record_result(
        "stop",
        score=0.9,
        feasible=False,
        metrics={"Accuracy": 0.9},
        branch_judgment=llm_judgment(expand=False, priority=1.0),
    )

    assert [node.node_id for node in tree.expansion_candidates()] == ["keep"]
    assert tree.select_for_expansion() == "keep"


def test_authoritatively_pruned_branch_cannot_reenter_live_set() -> None:
    tree = RefinementTree(max_depth=3, branching_factor=2, epsilon=0.0)
    root = default_variant(Modality.VISION, TaskType.CLASSIFICATION)
    root.trial_id = "root"
    tree.set_root(root)
    tree.expand("root", [variant("verifier-pruned"), variant("p2-repairable")])
    tree.expand("verifier-pruned", [variant("stale-descendant")])
    tree.record_result(
        "verifier-pruned",
        score=0.9,
        feasible=False,
        pruned=True,
        metrics={"Latency": 20.0},
        branch_judgment=llm_judgment(expand=True, priority=1.0),
    )
    tree.record_result(
        "p2-repairable",
        score=0.4,
        feasible=False,
        metrics={"Latency": 12.0},
        branch_judgment=llm_judgment(expand=True, priority=0.5),
    )

    assert tree.get_node("verifier-pruned").status == NodeStatus.PRUNED
    assert [node.node_id for node in tree.expansion_candidates()] == ["p2-repairable"]
    with pytest.raises(ValueError, match="branch pruned by verifier"):
        tree.select_for_expansion("verifier-pruned")
    with pytest.raises(ValueError, match="verifier-pruned branch"):
        tree.expand("stale-descendant", [variant("must-not-exist")])


def test_offline_selection_uses_llm_priority_not_raw_metric_score() -> None:
    tree = RefinementTree(max_depth=3, branching_factor=2, epsilon=0.0)
    root = default_variant(Modality.VISION, TaskType.CLASSIFICATION)
    root.trial_id = "root"
    tree.set_root(root)
    tree.expand("root", [variant("high-priority"), variant("high-score")])
    tree.record_result(
        "high-priority",
        score=0.2,
        feasible=True,
        metrics={"Accuracy": 0.2},
        branch_judgment=llm_judgment(expand=True, priority=0.9),
    )
    tree.record_result(
        "high-score",
        score=0.95,
        feasible=True,
        metrics={"Accuracy": 0.95},
        branch_judgment=llm_judgment(expand=True, priority=0.4),
    )

    assert tree.select_for_expansion() == "high-priority"


def test_llm_selection_is_confined_to_live_set_and_visible_evidence() -> None:
    selection = validate_branch_selection(
        {
            "selected_node_id": "node-a",
            "reasoning": "Closes the measured latency gap.",
            "evidence_cited": ["ev_latency"],
        },
        candidate_ids=["node-a", "node-b"],
        evidence_by_node={"node-a": {"ev_latency"}, "node-b": {"ev_quality"}},
    )
    assert selection.selected_node_id == "node-a"

    with pytest.raises(ValueError, match="outside"):
        validate_branch_selection(
            {"selected_node_id": "node-c", "evidence_cited": []},
            candidate_ids=["node-a", "node-b"],
            evidence_by_node={},
        )
    with pytest.raises(ValueError, match="Evidence IDs"):
        validate_branch_selection(
            {"selected_node_id": "node-a", "evidence_cited": ["ev_invented"]},
            candidate_ids=["node-a"],
            evidence_by_node={"node-a": {"ev_latency"}},
        )
    with pytest.raises(ValueError, match="Evidence IDs"):
        validate_branch_selection(
            {"selected_node_id": "node-a", "evidence_cited": ["ev_invented"]},
            candidate_ids=["node-a"],
            evidence_by_node={"node-a": set()},
        )


def test_declared_parent_components_are_inherited_byte_for_byte(tmp_path) -> None:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "loader.py").write_text("LOADER = 'parent'\n", encoding="utf-8")
    (parent / "train.py").write_text("TRAIN = 'parent'\n", encoding="utf-8")
    candidate = variant("child")
    candidate.parent_trial_id = "parent"
    candidate.inherited_components = ["loader.py"]
    candidate.changed_components = ["training recipe (train.py)"]

    parent_path = WorkspaceManager._parent_workspace(child, candidate)
    assert parent_path == parent
    assert WorkspaceManager._inherited_component_code(
        parent_path, candidate, "loader.py"
    ) == "LOADER = 'parent'\n"
    assert WorkspaceManager._inherited_component_code(
        parent_path, candidate, "train.py"
    ) is None
