import json

import pytest

from edgecraft.evaluation.resource_accounting import (
    compact_candidate_failure,
    cost_metrics_complete,
    summarize_candidate_llm_usage,
    summarize_stage_resources,
    summarize_token_telemetry,
)


def completed(**overrides):
    row = {
        "status": "completed",
        "provider": "provider-a",
        "model": "model-a",
        "experiment_run_id": "run-a",
        "provider_request_id": "request-a",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "request_started_at": "start",
        "response_received_at": "end",
    }
    row.update(overrides)
    return row


def test_token_summary_validates_totals_and_counts_failures(tmp_path) -> None:
    path = tmp_path / "tokens.jsonl"
    failed = completed(
        status="failed",
        provider_request_id=None,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
    )
    rows = [completed(), failed]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = summarize_token_telemetry(
        path,
        expected_provider="provider-a",
        expected_model="model-a",
        expected_run_id="run-a",
    )
    assert summary["total_tokens"] == 15
    assert summary["failed_request_count"] == 1
    assert summary["unmetered_failed_request_count"] == 1
    assert summary["token_count_complete"] is False
    assert summary["reported_totals_are_lower_bound"] is True
    path.write_text(json.dumps(completed(total_tokens=16)) + "\n")
    with pytest.raises(ValueError, match="inconsistent"):
        summarize_token_telemetry(path)


def test_resources_include_failed_candidate_stages() -> None:
    trials = [
        {"observations": {"stage_costs": {"loader": {"elapsed_s": 2}, "edge": {"elapsed_s": 3}}}},
        {"error": "failed", "observations": {"stage_costs": {"export": {"elapsed_s": 4}, "efficiency": {"elapsed_s": 5}}}},
    ]
    resources = summarize_stage_resources(trials, request_wall_s=20, allocated_gpu_count=2)
    assert resources["gpu_s"] == 40
    assert resources["device_s"] == 8
    assert resources["export_s"] == 4
    assert resources["compile_s"] == 0
    assert resources["accounted_trial_count"] == 2


def test_full_train_export_wall_time_is_not_misclassified_as_compile() -> None:
    resources = summarize_stage_resources(
        [
            {
                "observations": {
                    "stage_costs": {
                        "full_train_export": {
                            "elapsed_s": 100,
                            "resource": "gpu",
                        },
                        "target_compile": {"elapsed_s": 4, "resource": "device"},
                    }
                }
            }
        ],
        request_wall_s=110,
    )
    assert resources["train_s"] == 100
    assert resources["compile_s"] == 4


def test_incomplete_candidate_usage_fails_closed() -> None:
    summary = summarize_candidate_llm_usage(
        [{"candidate_id": "one", "llm_usage": {"token_count_complete": False}}]
    )
    assert summary["token_count_complete"] is False
    assert summary["total_tokens"] is None


def test_compact_failure_remains_auditable() -> None:
    row = compact_candidate_failure(
        {"metadata": {"export_format": "onnx"}},
        candidate_id="candidate-a",
        error=ValueError("invalid contract"),
        active_slo={"latency_ms": 10},
    )
    assert row["evaluation_valid"] is False
    assert row["measured_gap"] == {"latency_ms": None}
    record = {
        "llm_usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "token_count_complete": True,
        },
        "resources": {"wall_s": 2, "gpu_s": 2, "device_s": 0},
    }
    assert cost_metrics_complete(record)
