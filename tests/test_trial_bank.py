import pytest

from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.agent.search.trial_bank import TrialBank
from edgecraft.agent.search.trial_result import StageReached, TrialResult
from edgecraft.agent.search.verification import Evidence, GapSlackBrief, VerificationReport
from edgecraft.core.modality import Modality, TaskType


def _verified_trial(
    trial_id: str,
    score: float,
    gap: float,
    artifact_path,
) -> TrialResult:
    variant = SolutionVariant(
        trial_id=trial_id,
        modality=Modality.VISION,
        task_type=TaskType.CLASSIFICATION,
        model_name=trial_id,
        model_family="review",
        train_code="print('train')\n",
        infer_code="print('infer')\n",
    )
    evidence = Evidence(
        probe_id="full",
        quantity="Latency",
        fidelity="measured_congruent",
        outcome="pass",
        value=12.0,
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
    return TrialResult(
        trial_id=trial_id,
        variant=variant,
        stage_reached=StageReached.EDGE_BENCHMARK,
        score=score,
        is_feasible=False,
        artifact_paths={"artifact": str(artifact_path)},
        verification_report=VerificationReport(
            level="L2",
            status="pass",
            decision="escalate",
            evidence=[evidence],
        ),
        gap_slack=[
            GapSlackBrief(
                metric="Latency",
                comparison="lte",
                target=10,
                value=10 + gap,
                known=True,
                signed_gap=gap,
                gap=gap,
                slack=0,
                normalized_gap_high=gap / 10,
            )
        ],
    )


def test_best_verified_fallback_keeps_artifact_and_remaining_gap(tmp_path) -> None:
    bank = TrialBank("run-a", persist_path=tmp_path / "trial_bank.json")
    lower_artifact = tmp_path / "lower-quality.onnx"
    higher_artifact = tmp_path / "higher-quality.onnx"
    lower_artifact.write_bytes(b"synthetic-lower")
    higher_artifact.write_bytes(b"synthetic-higher")
    bank.add(_verified_trial("lower-quality", 0.4, 0.5, lower_artifact))
    bank.add(_verified_trial("higher-quality", 0.8, 2.0, higher_artifact))

    selected = bank.get_best_verified()
    assert selected is not None
    assert selected.trial_id == "higher-quality"
    assert selected.artifact_paths["artifact"].endswith("higher-quality.onnx")
    assert selected.gap_slack[0].signed_gap == 2.0
    assert selected.gap_slack[0].gap == 2.0
    assert not list(tmp_path.glob(".*.tmp"))


def test_verified_fallback_rejects_missing_artifact_or_incomplete_p2(tmp_path) -> None:
    bank = TrialBank("run-a")
    missing = _verified_trial("missing", 0.9, 1.0, tmp_path / "missing.onnx")
    bank.add(missing)
    incomplete_artifact = tmp_path / "incomplete.onnx"
    incomplete_artifact.write_bytes(b"synthetic")
    incomplete = _verified_trial("incomplete", 0.8, 1.0, incomplete_artifact)
    incomplete.verification_report.evidence[0].environment_fingerprint = ""
    bank.add(incomplete)
    assert bank.get_best_verified() is None


def test_trial_bank_refuses_to_silently_reset_corrupt_history(tmp_path) -> None:
    path = tmp_path / "trial_bank.json"
    path.write_text("{interrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to discard history"):
        TrialBank("run-a", persist_path=path)
