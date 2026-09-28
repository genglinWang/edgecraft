#!/usr/bin/env python3
"""Small device-free demonstration of the three decision mechanisms."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from edgecraft.agent.search.verification import Evidence, VerifierPolicy
from edgecraft.core.task import Constraint, UserSpec
from edgecraft.knowledge.calibration_store import (
    CalibrationPair,
    CalibrationStore,
    calibration_protocol_fingerprint,
)
from edgecraft.knowledge.compatibility import CompatibilityRuleStore


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="edgecraft-demo-") as temporary:
        root = Path(temporary)
        protocol = {
            "runtime": "onnxruntime",
            "precision": "fp32",
            "repetitions": 5,
            "measurement_sessions": 3,
            "latency_statistic": "p95",
            "latency_uncertainty": "sample_std_across_session_p95",
            "measurement_contract_version": "edge_efficiency_v3",
        }
        protocol_fp = calibration_protocol_fingerprint(protocol)
        calibration = CalibrationStore(root / "calibration.sqlite3")
        calibration.append(
            CalibrationPair(
                environment_fingerprint="environment-a",
                protocol_fingerprint=protocol_fp,
                quantity="Latency",
                cheap_value=14.0,
                full_value=15.0,
                graph_hash="graph-a",
                cheap_evidence_id="cheap-1",
                full_evidence_id="full-1",
                cheap_protocol=protocol,
                full_protocol=protocol,
            )
        )
        snapshot = calibration.snapshot(
            environment_fingerprint="environment-a",
            protocol_fingerprint=protocol_fp,
            graph_hash="graph-a",
        )

        spec = UserSpec(
            description="offline verifier demo",
            constraints=[Constraint(metric="Latency", comparison="lte", target=10.0)],
        )
        p1 = Evidence(
            id="p1-latency",
            probe_id="efficiency",
            quantity="Latency",
            fidelity="measured_congruent",
            outcome="pass",
            value=14.0,
            sigma=1.0,
            artifact_fingerprint={"graph_hash": "graph-a"},
            environment_fingerprint="environment-a",
            protocol={
                **protocol,
                "decision_authority": "p1_prune",
                "congruence_status": "calibrated_exact_graph",
                "runtime_exact": True,
                "fingerprint_verified": True,
                "calibration_protocol_fingerprint": protocol_fp,
                "calibration_snapshot_id": snapshot.snapshot_id,
                "calibration_pair_ids": snapshot.pair_ids,
                "calibration_error": snapshot.errors["Latency"],
                "p1_construction_guard": "edgecraft_p1_guard_v1",
                "p1_construction_guard_status": "passed",
                "p1_candidate_source_sha256": "synthetic-candidate-source",
                "p1_guard_source_sha256": "synthetic-guard-source",
                "p1_candidate_source_unchanged": True,
                "p1_guard_source_unchanged": True,
                "p1_process_spawn_policy": "blocked",
                "p1_training_mutation_count": 0,
            },
        )
        p1_decision = VerifierPolicy(kappa=2.0).decide([p1], spec)

        fingerprint = {
            "artifact_format": "onnx",
            "artifact_properties": {"ir_version": 99},
            "op_set": [],
        }
        rules = CompatibilityRuleStore(root / "rules.sqlite3")
        error = "Unsupported IR version: 99"
        observation = rules.observe_failure(
            device="review-device",
            runtime="onnx",
            version="runtime-a",
            precision="fp32",
            error_text=error,
            environment_fingerprint="environment-a",
            artifact_fingerprint=fingerprint,
        )
        assert observation is not None
        rule = rules.verify_observation(
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
                "device": "review-device",
                "runtime": "onnx",
                "runtime_version": "runtime-a",
                "precision": "fp32",
                "environment_fingerprint": "environment-a",
                "reproduction_scope": "artifact_header",
                "gate_eligible": True,
            },
        )
        assert rule is not None
        hits = rules.exact_match(
            device="review-device",
            runtime="onnx",
            version="runtime-a",
            precision="fp32",
            environment_fingerprint="environment-a",
            artifact_fingerprint=fingerprint,
        )
        stale_hits = rules.exact_match(
            device="review-device",
            runtime="onnx",
            version="runtime-b",
            precision="fp32",
            environment_fingerprint="environment-b",
            artifact_fingerprint=fingerprint,
        )
        print(
            json.dumps(
                {
                    "calibration_pairs_visible": snapshot.pair_count,
                    "p1_action": p1_decision.action,
                    "verified_rule_matches": len(hits),
                    "changed_runtime_matches": len(stale_hits),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
