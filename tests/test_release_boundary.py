import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from edgecraft.cli import main
from edgecraft.config.settings import settings
from edgecraft.utils.network import _ssh_base_args, validate_ssh_target


def test_default_device_manifest_contains_capabilities_not_connections() -> None:
    path = Path("edgecraft/config/device_runtime_docker.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["devices"]
    for facts in manifest["devices"].values():
        assert "ssh" not in facts
        assert "docker_image" not in facts
        assert "native_python" not in facts
        assert "native_envs" not in facts


def test_public_cli_is_named_edgecraft() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'edgecraft = "edgecraft.cli:main"' in project
    assert project.count('\nedgecraft = "') == 1
    assert Path("edgecraft/__main__.py").is_file()


def test_public_surfaces_do_not_advertise_deprecated_names() -> None:
    surfaces = [
        Path("README.md"),
        Path("REPRODUCIBILITY.md"),
        Path(".env.example"),
        Path("pyproject.toml"),
        Path("edgecraft/cli/commands.py"),
        *sorted(path for path in Path("edgecraft/tools/deploy/runner").rglob("*") if path.is_file()),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in surfaces)
    assert not re.search(r"\b(?:mo" + r"saas|MO" + r"SAAS)\b", text)
    assert "edgecraft synth" in Path("README.md").read_text(encoding="utf-8")


def test_model_registry_source_is_part_of_the_release_tree() -> None:
    # This is Python source required by the synthesis CLI, not a model weight.
    required = (
        Path("edgecraft/models/__init__.py"),
        Path("edgecraft/models/family_registry.py"),
        Path("edgecraft/models/specs.py"),
    )
    assert all(path.is_file() for path in required)


def test_public_cli_help_and_paper_profile_are_runnable() -> None:
    runner = CliRunner()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "synth" in help_result.output
    assert "profile" in help_result.output
    synth_help = runner.invoke(main, ["synth", "--help"])
    assert synth_help.exit_code == 0, synth_help.output
    assert "--ssh-key FILE" in synth_help.output
    assert "[required]" in synth_help.output
    profile_result = runner.invoke(main, ["profile", "show", "paper"])
    assert profile_result.exit_code == 0, profile_result.output
    assert '"profile": "paper"' in profile_result.output
    assert '"VERIFIER_MODE": "ladder"' in profile_result.output
    assert '"BRANCH_SELECTION_MODE": "llm_strict"' in profile_result.output
    assert '"REQUIRE_TOKEN_TELEMETRY": true' in profile_result.output


def test_serve_refuses_unauthenticated_non_loopback_bind(monkeypatch) -> None:
    monkeypatch.setattr(settings, "API_TENANT_TOKENS", "")
    result = CliRunner().invoke(main, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code != 0
    assert "requires EDGECRAFT_API_TENANT_TOKENS" in result.output


def test_edge_ssh_requires_one_explicit_identity() -> None:
    with pytest.raises(ValueError, match="explicit SSH private key"):
        _ssh_base_args("reviewer@device", None)

    args = _ssh_base_args("reviewer@device", "/review/key")
    assert args[:3] == ["ssh", "-i", "/review/key"]
    assert "IdentitiesOnly=yes" in args
    assert "BatchMode=yes" in args
    assert "UserKnownHostsFile=/dev/null" not in args
    assert "StrictHostKeyChecking=no" not in args
    assert validate_ssh_target("reviewer@device") == "reviewer@device"
    with pytest.raises(ValueError, match=r"\[user@\]host"):
        validate_ssh_target("-oProxyCommand=unexpected")


def test_edge_runner_transfer_scripts_pin_the_explicit_identity() -> None:
    scripts = (
        "collect_results.sh",
        "run_job_and_collect.sh",
        "stream_logs.sh",
        "submit_job.sh",
    )
    for name in scripts:
        text = Path("edgecraft/tools/deploy/runner/server", name).read_text(encoding="utf-8")
        assert '-i "$KEY"' in text
        assert "IdentitiesOnly=yes" in text
        assert "[user@]host form" in text
        assert "ACCEPT_NEW=0" in text

    push_runner = Path("edgecraft/tools/deploy/runner/edge/bin/edgecraft-edge-runner").read_text(
        encoding="utf-8"
    )
    assert '-i "$PUSH_SSH_KEY"' in push_runner
    assert "IdentitiesOnly=yes" in push_runner
    assert "PUSH_SSH_OPTS" not in push_runner
    assert "[user@]host:path form" in push_runner
