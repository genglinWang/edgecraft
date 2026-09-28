import json
import sqlite3

from edgecraft.knowledge.compatibility import CompatibilityRuleStore


def test_observation_needs_reproduction_and_matches_exact_context(tmp_path) -> None:
    store = CompatibilityRuleStore(tmp_path / "rules.sqlite3")
    fingerprint = {
        "artifact_format": "onnx",
        "artifact_properties": {"ir_version": 99},
        "op_set": [],
    }
    error = "Unsupported IR version: 99"
    observation = store.observe_failure(
        device="device-a",
        runtime="onnx",
        version="version-a",
        precision="fp16",
        error_text=error,
        environment_fingerprint="environment-a",
        artifact_fingerprint=fingerprint,
    )
    assert observation is not None
    assert store.exact_match(
        device="device-a",
        runtime="onnx",
        version="version-a",
        precision="fp16",
        environment_fingerprint="environment-a",
        artifact_fingerprint=fingerprint,
    ) == []
    rule = store.verify_observation(
        observation,
        reproduced_error_text=error,
        minimal_repro_path="minimal.onnx",
        pattern={
            "artifact_predicate": {
                "artifact_format": "onnx",
                "artifact_properties": {"ir_version": 99},
            }
        },
        protocol={
            "device": "device-a",
            "runtime": "onnx",
            "runtime_version": "version-a",
            "precision": "fp16",
            "environment_fingerprint": "environment-a",
            "reproduction_scope": "artifact_header",
            "gate_eligible": True,
        },
    )
    assert rule is not None and rule.can_hard_gate()
    snapshot = store.export_public_snapshot(tmp_path / "public-rules.json")
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert len(payload["verified_compatibility_rules"]) == 1
    assert "minimal.onnx" not in serialized
    assert error not in serialized
    assert "tenant_id" not in serialized
    assert "dataset_id" not in serialized
    assert len(
        store.exact_match(
            device="device-a",
            runtime="onnx",
            version="version-a",
            precision="fp16",
            environment_fingerprint="environment-a",
            artifact_fingerprint=fingerprint,
        )
    ) == 1
    assert store.exact_match(
        device="device-a",
        runtime="onnx",
        version="version-b",
        precision="fp16",
        environment_fingerprint="environment-b",
        artifact_fingerprint=fingerprint,
    ) == []
    assert store.exact_match(
        device="device-a",
        runtime="onnx",
        version="version-a",
        precision="fp16",
        environment_fingerprint="environment-b",
        artifact_fingerprint=fingerprint,
    ) == []
    assert store.exact_match(
        device="device-a",
        runtime="onnx",
        version="version-a",
        precision="int8",
        environment_fingerprint="environment-a",
        artifact_fingerprint=fingerprint,
    ) == []


def test_private_failure_observations_are_tenant_scoped_and_opaque(tmp_path) -> None:
    store = CompatibilityRuleStore(tmp_path / "rules.sqlite3")
    common = {
        "device": "device-a",
        "runtime": "onnx",
        "version": "version-a",
        "precision": "fp16",
        "error_text": "Unsupported IR version: 99",
        "environment_fingerprint": "environment-a",
        "artifact_fingerprint": {
            "artifact_format": "onnx",
            "artifact_properties": {"ir_version": 99},
        },
    }
    first = store.observe_failure(**common, tenant_id="tenant-a")
    second = store.observe_failure(**common, tenant_id="tenant-b")
    assert first is not None and second is not None and first.id != second.id
    assert len(store.list_observations(tenant_id="tenant-a")) == 1
    assert len(store.list_observations(tenant_id="tenant-b")) == 1
    assert store.observation_support(first) == 1
    assert store.observation_support(second) == 1
    raw = (tmp_path / "rules.sqlite3").read_bytes()
    assert b"tenant-a" not in raw
    assert b"tenant-b" not in raw


def test_verified_rules_from_two_environments_coexist(tmp_path) -> None:
    path = tmp_path / "rules.sqlite3"
    store = CompatibilityRuleStore(path)
    fingerprint = {
        "artifact_format": "onnx",
        "artifact_properties": {"ir_version": 99},
    }
    pattern = {
        "artifact_predicate": {
            "artifact_format": "onnx",
            "artifact_properties": {"ir_version": 99},
        }
    }
    error = "Unsupported IR version: 99"

    rules = []
    for environment in ("environment-a", "environment-b"):
        observation = store.observe_failure(
            device="device-a",
            runtime="onnx",
            version="version-a",
            precision="fp16",
            error_text=error,
            environment_fingerprint=environment,
            artifact_fingerprint=fingerprint,
        )
        assert observation is not None
        rule = store.verify_observation(
            observation,
            reproduced_error_text=error,
            minimal_repro_path="minimal.onnx",
            pattern=pattern,
            protocol={
                "device": "device-a",
                "runtime": "onnx",
                "runtime_version": "version-a",
                "precision": "fp16",
                "environment_fingerprint": environment,
                "reproduction_scope": "artifact_header",
                "gate_eligible": True,
            },
        )
        assert rule is not None
        rules.append(rule)

    assert rules[0].id != rules[1].id
    assert len(store.list_rules()) == 2
    with sqlite3.connect(path) as connection:
        raw_payload = connection.execute(
            "SELECT payload FROM rules WHERE id=?", (rules[0].id,)
        ).fetchone()[0]
        legacy_payload = json.loads(raw_payload)
        legacy_payload["id"] = "rule_legacy_environment_blind_id"
        connection.execute(
            "UPDATE rules SET id=?, payload=? WHERE id=?",
            (
                legacy_payload["id"],
                json.dumps(legacy_payload),
                rules[0].id,
            ),
        )
    reopened = CompatibilityRuleStore(path)
    assert len(reopened.list_rules()) == 2
    for environment, expected in zip(
        ("environment-a", "environment-b"), rules
    ):
        matches = reopened.exact_match(
            device="device-a",
            runtime="onnx",
            version="version-a",
            precision="fp16",
            environment_fingerprint=environment,
            artifact_fingerprint=fingerprint,
        )
        assert [item.id for item in matches] == [expected.id]
