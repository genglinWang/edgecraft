from edgecraft.utils.subprocess_env import candidate_subprocess_env


def test_generated_process_environment_excludes_controller_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("EDGECRAFT_API_TENANT_TOKENS", "synthetic-token-map")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/synthetic-agent")
    monkeypatch.setenv("PATH", "/review/bin")

    env = candidate_subprocess_env(
        tmp_path,
        {"EDGECRAFT_TRAIN_SEED": 7},
    )

    assert env["PATH"] == "/review/bin"
    assert env["EDGECRAFT_TRAIN_SEED"] == "7"
    assert env["HOME"].startswith(str(tmp_path))
    assert "OPENAI_API_KEY" not in env
    assert "EDGECRAFT_API_TENANT_TOKENS" not in env
    assert "SSH_AUTH_SOCK" not in env
