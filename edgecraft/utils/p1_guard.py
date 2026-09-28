"""Validation helpers for controller-owned P1 construction attestations."""
from __future__ import annotations

from typing import Any, Dict


P1_GUARD_SCHEMA = "edgecraft_p1_guard_v1"
P1_GUARD_PRODUCER = "edgecraft-controller"


def validate_p1_guard_attestation(
    attestation: Any,
    *,
    candidate_source_sha256: str,
    guard_source_sha256: str,
) -> Dict[str, Any]:
    """Check the controller runner's binding to one candidate source file."""
    record = dict(attestation) if isinstance(attestation, dict) else {}
    violations = []
    if record.get("schema_version") != P1_GUARD_SCHEMA:
        violations.append("missing controller P1 guard schema")
    if record.get("producer") != P1_GUARD_PRODUCER:
        violations.append("missing controller P1 guard producer")
    if record.get("status") != "passed":
        violations.append("controller P1 guard did not pass")
    if record.get("candidate_source_sha256") != candidate_source_sha256:
        violations.append("P1 guard is not bound to the executed candidate source")
    if record.get("guard_source_sha256") != guard_source_sha256:
        violations.append("P1 guard is not bound to the active controller runner")
    if record.get("process_spawn_policy") != "blocked":
        violations.append("P1 guard did not block child-process escape")
    if record.get("candidate_source_unchanged") is not True:
        violations.append("candidate source changed during P1 construction")
    if record.get("guard_source_unchanged") is not True:
        violations.append("controller guard source changed during P1 construction")
    calls = record.get("observed_training_mutations")
    if not isinstance(calls, list) or calls:
        violations.append("P1 guard observed a training mutation")
    mutation_count = record.get("training_mutation_count")
    if (
        not isinstance(mutation_count, int)
        or isinstance(mutation_count, bool)
        or mutation_count != 0
    ):
        violations.append("P1 guard reported a nonzero training mutation count")
    hooks = record.get("installed_hooks")
    installed_hooks = list(hooks) if isinstance(hooks, list) else []
    return {
        "valid": not violations,
        "violations": violations,
        "schema_version": record.get("schema_version"),
        "status": record.get("status"),
        "candidate_source_sha256": record.get("candidate_source_sha256"),
        "guard_source_sha256": record.get("guard_source_sha256"),
        "candidate_source_unchanged": record.get("candidate_source_unchanged"),
        "guard_source_unchanged": record.get("guard_source_unchanged"),
        "process_spawn_policy": record.get("process_spawn_policy"),
        "training_mutation_count": mutation_count,
        "installed_hooks": installed_hooks,
    }
