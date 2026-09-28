import json

import pytest

from edgecraft.knowledge.calibration_store import (
    CalibrationPair,
    CalibrationStore,
    calibration_protocol_fingerprint,
)


def test_calibration_is_append_only_and_exact_context(tmp_path) -> None:
    store = CalibrationStore(tmp_path / "calibration.sqlite3")
    pair = CalibrationPair(
        environment_fingerprint="env-a",
        protocol_fingerprint="protocol-a",
        quantity="Latency",
        cheap_value=10.0,
        full_value=12.5,
        graph_hash="graph-a",
        cheap_evidence_id="cheap-a",
        full_evidence_id="full-a",
    )
    store.append(pair)
    store.append(pair)
    snapshot = store.snapshot(
        environment_fingerprint="env-a",
        protocol_fingerprint="protocol-a",
        graph_hash="graph-a",
    )
    assert snapshot.pair_count == 1
    assert snapshot.errors["Latency"] == 2.5
    assert store.snapshot(
        environment_fingerprint="env-b",
        protocol_fingerprint="protocol-a",
        graph_hash="graph-a",
    ).pair_count == 0
    assert store.snapshot(
        environment_fingerprint="env-a",
        protocol_fingerprint="protocol-a",
        graph_hash="graph-b",
    ).pair_count == 0


def test_snapshot_refuses_graphless_context(tmp_path) -> None:
    with pytest.raises(ValueError, match="exact environment, protocol, and graph"):
        CalibrationStore(tmp_path / "calibration.sqlite3").snapshot(
            environment_fingerprint="env-a",
            protocol_fingerprint="protocol-a",
        )


def test_protocol_fingerprint_covers_runtime_measurement_conditions() -> None:
    base = {
        "runtime": "onnxruntime",
        "precision": "fp16",
        "actual_repetitions": 5,
        "measurement_sessions": 3,
        "latency_statistic": "p95",
        "thermal_state": "steady",
    }
    assert calibration_protocol_fingerprint(base) != calibration_protocol_fingerprint(
        {**base, "thermal_state": "warming"}
    )
    assert calibration_protocol_fingerprint(base) != calibration_protocol_fingerprint(
        {**base, "actual_repetitions": 6}
    )
    assert calibration_protocol_fingerprint(base) != calibration_protocol_fingerprint(
        {**base, "measurement_sessions": 4}
    )


def test_public_calibration_export_is_allowlisted(tmp_path) -> None:
    store = CalibrationStore(tmp_path / "calibration.sqlite3")
    protocol = {
        "runtime": "onnxruntime",
        "precision": "fp16",
        "actual_repetitions": 5,
        "private_dataset_path": "/private/data",
    }
    fingerprint = calibration_protocol_fingerprint(protocol)
    store.append(
        CalibrationPair(
            environment_fingerprint="env-a",
            protocol_fingerprint=fingerprint,
            quantity="Latency",
            cheap_value=10.0,
            full_value=11.0,
            graph_hash="graph-a",
            cheap_evidence_id="private-cheap-id",
            full_evidence_id="private-full-id",
            cheap_protocol=protocol,
            full_protocol=protocol,
        )
    )
    output = store.export_public_snapshot(tmp_path / "public.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert len(payload["pairs"]) == 1
    assert "private_dataset_path" not in serialized
    assert "private-cheap-id" not in serialized
    assert "created_at" not in serialized
