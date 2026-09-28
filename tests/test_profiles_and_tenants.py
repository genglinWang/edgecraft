import pytest

from edgecraft.agent.workspace.manager import (
    WorkspaceManager,
    ensure_private_run_workspace,
    run_workspace_path,
    tenant_workspace_key,
    validate_workspace_component,
)
from edgecraft.api.auth import (
    TenantAuthenticationError,
    authenticate_tenant,
    parse_tenant_tokens,
)
from edgecraft.api.task_manager import TaskManager, _TaskRecord
from edgecraft.config.profiles import profile_default_iterations, profile_settings
from edgecraft.config.settings import edgecraft_env
from edgecraft.knowledge.cbr.scope import case_store_namespace


def test_paper_profile_activates_claim_paths() -> None:
    profile = profile_settings("paper")
    assert profile_default_iterations("paper") == 24
    assert profile["EXPANSION_MODE"] == "constraint_directed"
    assert profile["BRANCH_SELECTION_MODE"] == "llm_strict"
    assert profile["TREE_BRANCHING_FACTOR"] == 3
    assert profile["BRANCH_JUDGMENT_MODE"] == "llm_strict"
    assert profile["VERIFIER_MODE"] == "ladder"
    assert profile["EXECUTION_BACKEND"] == "scheduler"
    assert profile["SCHEDULER_POLICY"] == "cost_aware"
    assert profile["REQUIRE_TOKEN_TELEMETRY"] is True
    assert profile["L1_FALSE_PRUNE_SAMPLE"] is False
    audit = profile_settings("paper-audit")
    assert profile_default_iterations("paper-audit") == 24
    assert audit["L1_FALSE_PRUNE_SAMPLE"] is True
    assert {
        key: value for key, value in audit.items() if key not in {"PROFILE", "L1_FALSE_PRUNE_SAMPLE"}
    } == {
        key: value for key, value in profile.items() if key not in {"PROFILE", "L1_FALSE_PRUNE_SAMPLE"}
    }


def test_environment_settings_override_defaults(monkeypatch) -> None:
    monkeypatch.delenv("EDGECRAFT_SAMPLE_SETTING", raising=False)
    assert edgecraft_env("SAMPLE_SETTING", "default") == "default"
    monkeypatch.setenv("EDGECRAFT_SAMPLE_SETTING", "public")
    assert edgecraft_env("SAMPLE_SETTING", "default") == "public"


def test_workspace_and_cbr_namespaces_are_tenant_private(tmp_path) -> None:
    first = run_workspace_path("run-a", tenant_id="tenant-a", base_dir=tmp_path)
    second = run_workspace_path("run-a", tenant_id="tenant-b", base_dir=tmp_path)
    assert first != second
    assert "tenant-a" not in str(first)
    assert "tenant-b" not in str(second)
    assert tenant_workspace_key("tenant-a") != tenant_workspace_key("tenant-b")
    assert case_store_namespace("tenant-a", scope="tenant") != case_store_namespace(
        "tenant-b", scope="tenant"
    )
    manager = WorkspaceManager(base_dir=tmp_path)
    assert manager._tenant_weights_cache("tenant-a") != manager._tenant_weights_cache(
        "tenant-b"
    )
    assert "tenant-a" not in str(manager._tenant_weights_cache("tenant-a"))


def test_workspace_component_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        validate_workspace_component("../other-tenant", label="run_id")


def test_created_tenant_run_directories_are_owner_only(tmp_path) -> None:
    run_path = ensure_private_run_workspace(
        "run-a",
        tenant_id="tenant-a",
        base_dir=tmp_path,
    )
    assert run_path.stat().st_mode & 0o077 == 0
    assert run_path.parent.stat().st_mode & 0o077 == 0


def test_optional_bearer_mapping_enforces_tenant_identity() -> None:
    raw = '{"0123456789abcdef":"tenant-a"}'
    assert parse_tenant_tokens(raw) == {"0123456789abcdef": "tenant-a"}
    assert (
        authenticate_tenant("Bearer 0123456789abcdef", token_config=raw)
        == "tenant-a"
    )
    with pytest.raises(TenantAuthenticationError, match="invalid"):
        authenticate_tenant("Bearer fedcba9876543210", token_config=raw)
    assert authenticate_tenant(None, token_config="") is None


def test_artifact_paths_cannot_escape_the_owning_run(tmp_path) -> None:
    run_root = tmp_path / "tenant-scope" / "run-a"
    record = _TaskRecord(
        task_id="task-a",
        request={},
        tenant_id="tenant-a",
        trial_bank_path=str(run_root / "trial_bank.json"),
    )
    assert TaskManager._path_belongs_to_record(
        record, run_root / "trial-a" / "outputs" / "best.onnx"
    )
    assert not TaskManager._path_belongs_to_record(
        record, tmp_path / "other-tenant" / "run-b" / "best.onnx"
    )
