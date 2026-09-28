from pathlib import Path

from edgecraft.config.settings import settings
from edgecraft.tools.deploy.edge_runner import EdgeRunner


def test_runner_writes_to_job_workspace_not_installed_package(tmp_path, monkeypatch):
    package = tmp_path / "installed-runner"
    package.mkdir()
    (package / "check.sh").write_text("#!/bin/bash\npwd\nprintf 'result' > result.txt\n")
    workspace = tmp_path / "job"
    workspace.mkdir()
    monkeypatch.setattr(settings, "EDGECRAFT_ROOT", str(workspace))
    runner = EdgeRunner(edge_runner_path=str(package))
    result = runner._run_edge_script("check.sh", [], cwd=str(workspace))
    assert result["status"] == "success"
    assert Path(result["stdout"].strip()).resolve() == workspace.resolve()
    assert (workspace / "result.txt").read_text() == "result"
    assert not (package / "result.txt").exists()
