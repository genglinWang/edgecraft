"""Named, inspectable settings profiles for public EdgeCraft entry points.

Profiles select mechanism semantics only.  They never contain datasets,
credentials, device addresses, model paths, or measured paper results.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping


PAPER_PROFILE: Dict[str, Any] = {
    "PROFILE": "paper",
    "TEMPLATE_MODE": "guidance",
    "EXPANSION_MODE": "constraint_directed",
    "BRANCH_SELECTION_MODE": "llm_strict",
    "BRANCH_JUDGMENT_MODE": "llm_strict",
    "TREE_BRANCHING_FACTOR": 3,
    "MULTIFIDELITY_MODE": "ladder",
    "VERIFIER_MODE": "ladder",
    "VERIFIER_PROBES": "static_artifact,efficiency,full",
    "COMPAT_RULES": "enforce",
    "EXECUTION_BACKEND": "scheduler",
    "SCHEDULER_POLICY": "cost_aware",
    "CBR_SCOPE": "tenant",
    "REQUIRE_TOKEN_TELEMETRY": True,
    "REQUIRE_SPLIT_MANIFEST_CONSUMPTION": True,
    "REQUIRE_EDGE_QUALITY_PARITY": True,
    # Deployed policy: an authorized P1 rejection stops this candidate.
    "L1_FALSE_PRUNE_SAMPLE": False,
    "MAX_TRAIN_EPOCHS": 100,
}


PAPER_AUDIT_PROFILE: Dict[str, Any] = {
    **PAPER_PROFILE,
    "PROFILE": "paper-audit",
    # Counterfactual audit: execute P2 after every would-prune P1 decision so
    # false-prune rate and projected policy cost can be measured separately.
    "L1_FALSE_PRUNE_SAMPLE": True,
}


OFFLINE_PROFILE: Dict[str, Any] = {
    "PROFILE": "offline",
    "TEMPLATE_MODE": "guidance",
    "EXPANSION_MODE": "constraint_directed",
    "BRANCH_SELECTION_MODE": "heuristic",
    "BRANCH_JUDGMENT_MODE": "fallback",
    "TREE_BRANCHING_FACTOR": 3,
    "MULTIFIDELITY_MODE": "full",
    "VERIFIER_MODE": "full",
    "VERIFIER_PROBES": "static_artifact,full",
    "COMPAT_RULES": "off",
    "EXECUTION_BACKEND": "direct",
    "SCHEDULER_POLICY": "cost_aware",
    "CBR_SCOPE": "tenant",
    "REQUIRE_TOKEN_TELEMETRY": False,
    "REQUIRE_SPLIT_MANIFEST_CONSUMPTION": False,
    "REQUIRE_EDGE_QUALITY_PARITY": False,
    "L1_FALSE_PRUNE_SAMPLE": False,
    "MAX_TRAIN_EPOCHS": 1,
}


PROFILES: Mapping[str, Mapping[str, Any]] = {
    "paper": PAPER_PROFILE,
    "paper-audit": PAPER_AUDIT_PROFILE,
    "offline": OFFLINE_PROFILE,
}

PROFILE_DEFAULT_ITERATIONS = {"paper": 24, "paper-audit": 24, "offline": 2}


def _choices() -> str:
    return ", ".join(PROFILES)


def profile_settings(name: str) -> Dict[str, Any]:
    """Return a copy of a named profile or raise a reviewer-readable error."""
    key = str(name or "").strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown EdgeCraft profile {name!r}; choose {_choices()}")
    return dict(PROFILES[key])


def profile_default_iterations(name: str) -> int:
    key = str(name or "").strip().lower()
    if key not in PROFILE_DEFAULT_ITERATIONS:
        raise ValueError(f"unknown EdgeCraft profile {name!r}; choose {_choices()}")
    return int(PROFILE_DEFAULT_ITERATIONS[key])


def resolved_profile_manifest(settings_obj: Any, name: str) -> Dict[str, Any]:
    """Return only non-secret mechanism fields for review and run manifests."""
    configured = profile_settings(name)
    return {
        "schema_version": "edgecraft_profile_v1",
        "profile": str(name).strip().lower(),
        "default_iterations": profile_default_iterations(name),
        "settings": {
            key: getattr(settings_obj, key, configured[key])
            for key in sorted(configured)
        },
        "external_evidence_required_for_paper_numbers": True,
    }
