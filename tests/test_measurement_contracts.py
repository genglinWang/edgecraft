import pytest

from edgecraft.knowledge.compatibility.repro import write_runtime_repro_infer
from edgecraft.tools.deploy.energy_measurement import (
    POWER_IDLE_END_PREFIX,
    POWER_SAMPLE_PREFIX,
    merge_trusted_energy,
)


def _sample(timestamp_ns: int, power_w: float) -> str:
    return f"{POWER_SAMPLE_PREFIX}{timestamp_ns}\t{power_w}\tsensor\tboard"


def test_energy_requires_three_evaluator_owned_active_samples() -> None:
    result = {
        "metrics": {"Latency": 2.0, "Energy_mj": 999.0},
        "measurement_protocol": {
            "benchmark_start_ns": 200_000_000,
            "benchmark_end_ns": 500_000_000,
            "actual_repetitions": 3,
            "benchmark_scope": "inference_loop",
        },
    }
    insufficient = "\n".join(
        [
            f"{POWER_IDLE_END_PREFIX}150000000",
            _sample(100_000_000, 2.0),
            _sample(250_000_000, 3.0),
            _sample(350_000_000, 5.0),
        ]
    )
    rejected = merge_trusted_energy(result, insufficient)
    assert "Energy_mj" not in rejected["metrics"]
    assert rejected["measurement_protocol"]["energy_status"] == "unknown"

    sufficient = insufficient + "\n" + _sample(450_000_000, 4.0)
    measured = merge_trusted_energy(result, sufficient)
    assert measured["metrics"]["Power_w"] == 4.0
    assert measured["metrics"]["Energy_mj"] == pytest.approx(400.0)
    assert measured["measurement_protocol"]["energy_active_sample_count"] == 3
    assert measured["measurement_protocol"]["energy_trusted"] is True


def test_runtime_replay_emits_warmup_and_tail_latency_contract(tmp_path) -> None:
    path = write_runtime_repro_infer(
        str(tmp_path / "infer.py"),
        warmup=7,
        min_warmup_seconds=2.5,
        repetitions=11,
        min_measure_seconds=3.5,
        measurement_sessions=3,
    )
    text = (tmp_path / "infer.py").read_text(encoding="utf-8")
    assert path == str(tmp_path / "infer.py")
    assert "warmup = 7" in text
    assert "min_warmup_seconds = 2.5" in text
    assert "repetitions = 11" in text
    assert "min_measure_seconds = 3.5" in text
    assert "measurement_sessions = 3" in text
    assert "session_p95.append(percentile(samples, 0.95))" in text
    assert '"latency_statistic": "p95"' in text
    assert '"latency_p95_session_std_ms"' in text
