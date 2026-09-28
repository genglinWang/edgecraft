from types import SimpleNamespace

from edgecraft.agent.metrics import MetricRegistry
from edgecraft.agent.search.scorer import Scorer
from edgecraft.agent.search.verification import Evidence, VerificationReport, VerifierPolicy
from edgecraft.core.task import Constraint, UserSpec


def spec() -> UserSpec:
    return UserSpec(
        description="test",
        constraints=[Constraint(metric="Latency", comparison="lte", target=10.0)],
    )


def test_p1_needs_calibration_before_it_can_prune() -> None:
    evidence = Evidence(
        probe_id="efficiency",
        quantity="Latency",
        fidelity="measured_congruent",
        outcome="pass",
        value=15.0,
        sigma=1.0,
        artifact_fingerprint={"graph_hash": "graph-a"},
        environment_fingerprint="environment-a",
    )
    assert VerifierPolicy().decide([evidence], spec()).action == "escalate"
    evidence.protocol.update(
        {
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
        }
    )
    assert VerifierPolicy(kappa=2.0).decide([evidence], spec()).action == "prune"


def test_only_full_congruent_evidence_accepts() -> None:
    evidence = Evidence(
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
    assert VerifierPolicy().decide([evidence], spec()).action == "accept"


def test_full_metric_without_execution_attestation_cannot_accept() -> None:
    evidence = Evidence(
        probe_id="full",
        quantity="Latency",
        fidelity="measured_congruent",
        outcome="pass",
        value=9.0,
    )
    assert VerifierPolicy().decide([evidence], spec()).action == "escalate"


def test_runtime_fallback_evidence_cannot_accept_requested_runtime() -> None:
    evidence = Evidence(
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
            "runtime_exact": False,
            "fingerprint_verified": True,
            "evaluation_valid": True,
            "fallback_reason": "requested runtime unavailable",
            "measurement_sessions": 3,
            "latency_statistic": "p95",
        },
    )
    assert VerifierPolicy().decide([evidence], spec()).action == "escalate"


def test_static_failure_requires_verified_rule() -> None:
    evidence = Evidence(
        probe_id="static",
        quantity="compatibility",
        fidelity="static",
        outcome="fail",
    )
    assert VerifierPolicy().decide([evidence], spec()).action == "escalate"
    evidence.protocol["verified_rule"] = True
    evidence.protocol["rule_id"] = "rule-a"
    assert VerifierPolicy().decide([evidence], spec()).action == "prune"


def test_scorer_recomputes_authority_instead_of_trusting_decision_string() -> None:
    result = SimpleNamespace(
        verification_report=VerificationReport(decision="accept"),
        observations={},
        local_metrics=None,
        edge_metrics=SimpleNamespace(
            latency_ms=9.0,
            memory_mb=None,
            all_metrics={"Latency": 9.0},
        ),
        stage_reached=SimpleNamespace(value="edge_benchmark"),
    )
    assert Scorer().is_feasible(result, spec()) is False


def test_scorer_accepts_only_recomputable_full_evidence() -> None:
    evidence = Evidence(
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
    result = SimpleNamespace(
        verification_report=VerificationReport(
            decision="accept",
            admission_status="admitted",
            evidence=[evidence],
        ),
        observations={"evaluation_validity": {"evaluation_valid": True}},
        local_metrics=None,
        edge_metrics=SimpleNamespace(
            latency_ms=9.0,
            memory_mb=None,
            all_metrics={"Latency": 9.0},
        ),
        stage_reached=SimpleNamespace(value="edge_benchmark"),
    )
    assert Scorer().is_feasible(result, spec()) is True


def test_latency_slo_uses_p95_when_mean_would_pass() -> None:
    metrics = MetricRegistry.normalize_metric_dict(
        {"latency_ms": 8.0, "latency_p95_ms": 12.0}
    )
    assert metrics["Latency_mean"] == 8.0
    assert metrics["Latency"] == 12.0
    evidence = Evidence(
        probe_id="full",
        quantity="Latency",
        fidelity="measured_congruent",
        outcome="pass",
        value=metrics["Latency"],
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
    assert VerifierPolicy().decide([evidence], spec()).action == "escalate"


def test_latency_evidence_without_p95_protocol_has_no_authority() -> None:
    evidence = Evidence(
        probe_id="full",
        quantity="Latency",
        fidelity="measured_congruent",
        outcome="pass",
        value=8.0,
        artifact_fingerprint={"graph_hash": "graph-a"},
        environment_fingerprint="environment-a",
        protocol={
            "decision_authority": "p2_accept",
            "artifact_executed": True,
            "runtime_exact": True,
            "fingerprint_verified": True,
            "evaluation_valid": True,
        },
    )
    assert VerifierPolicy().decide([evidence], spec()).action == "escalate"


def test_p1_candidate_zero_step_declaration_needs_controller_guard() -> None:
    evidence = Evidence(
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
            "training_steps": 0,
        },
    )
    assert VerifierPolicy(kappa=2.0).decide([evidence], spec()).action == "escalate"
