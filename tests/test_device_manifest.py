import json

from edgecraft.config.device_manifest import load_declared_devices


def test_nested_runtime_paths_expand_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVICE_ENV_ROOT", "/opt/edgecraft")
    manifest = tmp_path / "devices.json"
    manifest.write_text(json.dumps({"devices": {"raspberry_pi_5": {
        "execution_mode": "native",
        "native_envs": {"pytorch": "${DEVICE_ENV_ROOT}/torch/bin/python"},
    }}}))
    result = load_declared_devices(manifest)
    assert result["raspberry_pi_5"]["native_envs"]["pytorch"] == "/opt/edgecraft/torch/bin/python"
