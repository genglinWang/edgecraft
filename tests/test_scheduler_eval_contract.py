from types import SimpleNamespace
import time

import pytest

pytest.importorskip("langchain_openai")

from edgecraft.agent.nodes import pipeline_executor as executor
from edgecraft.agent.search.solution_variant import default_variant
from edgecraft.agent.search.trial_result import EdgeMetrics, TrialResult
from edgecraft.core.modality import Modality, TaskType
from edgecraft.scheduler.types import JobResult, JobStatus


def test_scheduler_trial_keeps_metrics_in_verifier_envelope(tmp_path, monkeypatch):
    variant = default_variant(Modality.STRUCTURED, TaskType.CLASSIFICATION)
    trial = TrialResult(trial_id=variant.trial_id, variant=variant)
    metrics = {"AUC": 0.75, "Latency_p95": 2.0}
    job = JobResult(job_id="example", status=JobStatus.SUCCESS, stage_reached="edge",
                    edge_metrics=metrics,
                    diagnostics={"edge_result": {"runtime_used": "onnxruntime"}})
    monkeypatch.setattr(executor, "_record_workspace_artifacts", lambda *a: None)
    result = executor._job_result_to_trial(
        job_result=job, trial=trial, ws=tmp_path, variant=variant, start_time=time.time(),
    )
    envelope = result.observations["scheduler_edge_result"]
    assert envelope["metrics"] == metrics
    assert envelope["status"] == "success"
    assert envelope["runtime_used"] == "onnxruntime"
    assert result.edge_metrics.latency_p95_ms == 2.0


@pytest.mark.parametrize("host_status", ["success", "error"])
def test_scheduler_respects_executed_payload_adapter(tmp_path, monkeypatch, host_status):
    variant = default_variant(Modality.STRUCTURED, TaskType.CLASSIFICATION)
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"test-interface")
    trial = TrialResult(trial_id=variant.trial_id, variant=variant,
                        edge_metrics=EdgeMetrics(latency_ms=1.0))
    trial.failure_context["scheduler"] = {"status": "success", "stage_reached": "edge"}
    monkeypatch.setattr(executor, "_record_stage_execution_evidence", lambda *a, **kw: None)
    monkeypatch.setattr(executor, "_primary_deploy_artifact", lambda *a: str(artifact))
    monkeypatch.setattr(executor, "build_artifact_fingerprint",
                        lambda *a: SimpleNamespace(model_dump=lambda **kw: {}))
    monkeypatch.setattr(executor, "_record_declared_artifacts", lambda *a: None)
    monkeypatch.setattr(executor, "_audit_edge_eval_bundle", lambda *a: None)
    monkeypatch.setattr(executor, "_strict_edge_eval_bundle_error", lambda *a: "")
    monkeypatch.setattr(executor, "_artifact_eval_payload_contract",
                        lambda **kw: SimpleNamespace(outcome="fail"))
    monkeypatch.setattr(executor, "_declared_artifact_path", lambda *a: artifact)
    monkeypatch.setattr(executor, "_apply_verifier_policy", lambda *a, **kw: None)

    def host_probe(**kwargs):
        trial.observations["component_roundtrip"] = {"status": host_status}
        return SimpleNamespace(outcome="pass" if host_status == "success" else "fail")

    monkeypatch.setattr(executor, "_run_host_infer_contract_probe", host_probe)

    class ReachedFurtherVerification(Exception):
        pass

    def continue_verification(**kwargs):
        raise ReachedFurtherVerification

    monkeypatch.setattr(executor, "_run_evaluation_metric_contract", continue_verification)
    kwargs = dict(trial=trial, state={}, variant=variant, user_spec=None, ws=tmp_path)
    if host_status == "success":
        with pytest.raises(ReachedFurtherVerification):
            executor._finalize_scheduler_verification(**kwargs)
        assert trial.error is None
    else:
        executor._finalize_scheduler_verification(**kwargs)
        assert trial.error_stage == "edge_eval_contract"
        assert "interface mismatch" in trial.error
