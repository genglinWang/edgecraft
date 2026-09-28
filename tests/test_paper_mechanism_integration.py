from edgecraft.agent.search.branch_judgment import BranchJudgment
from edgecraft.agent.search.refinement_tree import NodeStatus, RefinementTree
from edgecraft.agent.search.scorer import Scorer
from edgecraft.agent.search.solution_variant import SolutionVariant, default_variant
from edgecraft.agent.search.trial_bank import TrialBank
from edgecraft.agent.search.trial_result import EdgeMetrics, StageReached, TrialResult
from edgecraft.agent.search.verification import Evidence, VerificationReport, VerifierPolicy
from edgecraft.config.profiles import profile_settings
from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import Constraint, UserSpec
from edgecraft.knowledge.calibration_store import (
    CalibrationPair,
    CalibrationStore,
    calibration_protocol_fingerprint,
)


def _variant(trial_id: str) -> SolutionVariant:
    return SolutionVariant(
        trial_id=trial_id,
        modality=Modality.VISION,
        task_type=TaskType.CLASSIFICATION,
        model_name=trial_id,
        model_family="review",
        train_code="print('train')\n",
        infer_code="print('infer')\n",
    )


def _llm_expand_judgment() -> BranchJudgment:
    return BranchJudgment(
        judgment="promising_failure",
        should_expand=True,
        expand_mode="repair",
        priority=1.0,
        evidence_cited=["ev_latency"],
        source="llm",
    )


def test_paper_path_composes_prune_tree_and_p2_selection(tmp_path) -> None:
    profile = profile_settings("paper")
    assert profile["VERIFIER_MODE"] == "ladder"
    assert profile["EXPANSION_MODE"] == "constraint_directed"
    assert profile["EXECUTION_BACKEND"] == "scheduler"
    assert profile["L1_FALSE_PRUNE_SAMPLE"] is False

    user_spec = UserSpec(
        description="review composition",
        constraints=[Constraint(metric="Latency", comparison="lte", target=10.0)],
    )
    verifier = VerifierPolicy(kappa=2.0)
    tree = RefinementTree(max_depth=3, branching_factor=2, epsilon=0.0)
    root = default_variant(Modality.VISION, TaskType.CLASSIFICATION)
    root.trial_id = "root"
    tree.set_root(root)
    tree.expand("root", [_variant("p1-pruned"), _variant("p2-accepted")])
    bank = TrialBank("review-run", persist_path=tmp_path / "trial_bank.json")

    p1 = Evidence(
        id="ev_latency",
        probe_id="efficiency",
        quantity="Latency",
        fidelity="measured_congruent",
        outcome="pass",
        value=15.0,
        sigma=1.0,
        artifact_fingerprint={"graph_hash": "graph-a"},
        environment_fingerprint="environment-a",
        protocol={
            "decision_authority": "p1_prune",
            "congruence_status": "calibrated_exact_graph",
            "runtime_exact": True,
            "fingerprint_verified": True,
            "calibration_error": 1.0,
            "calibration_snapshot_id": "snapshot-a",
            "calibration_pair_ids": ["pair-a"],
            "calibration_protocol_fingerprint": "protocol-a",
            "measurement_sessions": 3,
            "latency_statistic": "p95",
            "latency_uncertainty": "sample_std_across_session_p95",
            "p1_construction_guard": "edgecraft_p1_guard_v1",
            "p1_construction_guard_status": "passed",
            "p1_candidate_source_sha256": "candidate-source-sha256",
            "p1_guard_source_sha256": "guard-source-sha256",
            "p1_candidate_source_unchanged": True,
            "p1_guard_source_unchanged": True,
            "p1_process_spawn_policy": "blocked",
            "p1_training_mutation_count": 0,
        },
    )
    prune = verifier.decide([p1], user_spec)
    assert prune.action == "prune"
    pruned_trial = TrialResult(
        trial_id="p1-pruned",
        variant=_variant("p1-pruned"),
        verification_report=VerificationReport(
            level="L1",
            status="fail",
            decision=prune.action,
            evidence=[p1],
            decision_basis=prune.basis,
        ),
        branch_judgment=_llm_expand_judgment(),
    )
    bank.add(pruned_trial)
    tree.record_result(
        "p1-pruned",
        score=0.9,
        feasible=False,
        pruned=True,
        branch_judgment=pruned_trial.branch_judgment,
    )

    p2 = Evidence(
        probe_id="full",
        quantity="Latency",
        fidelity="measured_congruent",
        outcome="pass",
        value=9.0,
        artifact_fingerprint={"graph_hash": "graph-a"},
        environment_fingerprint="environment-a",
        protocol={
            "decision_authority": "p2_accept",
            "artifact_executed": True,
            "runtime_exact": True,
            "fingerprint_verified": True,
            "evaluation_valid": True,
            "measurement_sessions": 3,
            "latency_statistic": "p95",
        },
    )
    accept = verifier.decide([p2], user_spec)
    assert accept.action == "accept"
    accepted_artifact = tmp_path / "p2-accepted.onnx"
    accepted_artifact.write_bytes(b"synthetic-review-artifact")
    accepted_trial = TrialResult(
        trial_id="p2-accepted",
        variant=_variant("p2-accepted"),
        stage_reached=StageReached.EDGE_BENCHMARK,
        score=0.8,
        edge_metrics=EdgeMetrics(
            latency_ms=9.0,
            latency_mean_ms=7.0,
            latency_p95_ms=9.0,
            all_metrics={"Latency": 9.0, "Latency_mean": 7.0},
        ),
        artifact_paths={"artifact": str(accepted_artifact)},
        observations={"evaluation_validity": {"evaluation_valid": True}},
        verification_report=VerificationReport(
            level="L2",
            status="pass",
            decision=accept.action,
            evidence=[p2],
            decision_basis=accept.basis,
        ),
    )
    scorer = Scorer()
    accepted_trial.gap_slack = scorer.build_gap_slack(accepted_trial, user_spec)
    assert accepted_trial.gap_slack[0].signed_gap == -1.0
    accepted_trial.is_feasible = scorer.is_feasible(accepted_trial, user_spec)
    assert accepted_trial.is_feasible is True
    bank.add(accepted_trial)
    tree.record_result("p2-accepted", score=0.8, feasible=True)

    assert tree.get_node("p1-pruned").status == NodeStatus.PRUNED
    assert "p1-pruned" not in {
        node.node_id for node in tree.expansion_candidates()
    }
    assert bank.get_best_feasible(require_evaluation_valid=True).trial_id == "p2-accepted"
    assert bank.get_best_verified(user_spec).trial_id == "p2-accepted"


def test_prior_p1_p2_pair_authorizes_only_a_later_candidate(tmp_path) -> None:
    user_spec = UserSpec(
        description="prequential calibration",
        constraints=[Constraint(metric="Latency", comparison="lte", target=10.0)],
    )
    protocol = {
        "runtime": "onnxruntime",
        "precision": "fp16",
        "repetitions": 5,
        "measurement_sessions": 3,
        "latency_statistic": "p95",
        "latency_uncertainty": "sample_std_across_session_p95",
        "measurement_contract_version": "edge_efficiency_v3",
    }
    protocol_fingerprint = calibration_protocol_fingerprint(protocol)
    store = CalibrationStore(tmp_path / "calibration.sqlite3")
    cold = store.snapshot(
        environment_fingerprint="environment-a",
        protocol_fingerprint=protocol_fingerprint,
        graph_hash="graph-a",
    )
    assert cold.pair_count == 0

    first_p1 = Evidence(
        probe_id="efficiency",
        quantity="Latency",
        fidelity="proxy",
        outcome="pass",
        value=14.0,
        sigma=1.0,
        artifact_fingerprint={"graph_hash": "graph-a"},
        environment_fingerprint="environment-a",
        protocol={**protocol, "decision_authority": "audit_only"},
    )
    assert VerifierPolicy(kappa=2.0).decide([first_p1], user_spec).action == "escalate"

    pair = CalibrationPair(
        environment_fingerprint="environment-a",
        protocol_fingerprint=protocol_fingerprint,
        quantity="Latency",
        cheap_value=11.0,
        full_value=12.0,
        cheap_sigma=1.0,
        full_sigma=1.0,
        graph_hash="graph-a",
        cheap_evidence_id="ev-prior-p1",
        full_evidence_id="ev-prior-p2",
        cheap_protocol=protocol,
        full_protocol=protocol,
    )
    store.append(pair)
    prior = store.snapshot(
        environment_fingerprint="environment-a",
        protocol_fingerprint=protocol_fingerprint,
        graph_hash="graph-a",
    )
    assert prior.pair_ids == [pair.pair_id]

    later_p1 = first_p1.model_copy(
        update={
            "id": "ev-later-p1",
            "fidelity": "measured_congruent",
            "protocol": {
                **protocol,
                "decision_authority": "p1_prune",
                "congruence_status": "calibrated_exact_graph",
                "runtime_exact": True,
                "fingerprint_verified": True,
                "calibration_error": prior.errors["Latency"],
                "calibration_snapshot_id": prior.snapshot_id,
                "calibration_pair_ids": prior.pair_ids,
                "calibration_protocol_fingerprint": protocol_fingerprint,
                "p1_construction_guard": "edgecraft_p1_guard_v1",
                "p1_construction_guard_status": "passed",
                "p1_candidate_source_sha256": "candidate-source-sha256",
                "p1_guard_source_sha256": "guard-source-sha256",
                "p1_candidate_source_unchanged": True,
                "p1_guard_source_unchanged": True,
                "p1_process_spawn_policy": "blocked",
                "p1_training_mutation_count": 0,
            },
        }
    )
    assert VerifierPolicy(kappa=2.0).decide([later_p1], user_spec).action == "prune"
