#!/usr/bin/env python3
"""Exercise an installed EdgeCraft wheel from outside its source checkout."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="edgecraft-install-check-") as directory:
        env = os.environ.copy()
        env["EDGECRAFT_ROOT"] = directory
        checks = (
            ["-c", "from edgecraft.config.device_manifest import load_declared_devices; assert load_declared_devices()"],
            ["-c", "from importlib.resources import files; root = files('edgecraft'); assert root.joinpath('config/crawl_sources.yaml').read_text(); assert root.joinpath('tools/deploy/benchmark/run.sh').read_text()"],
            ["-c", "from edgecraft.tools.deploy.edge_runner import EdgeRunner; from pathlib import Path; assert (Path(EdgeRunner().edge_runner_path) / 'run_job_and_collect.sh').is_file()"],
            ["-m", "edgecraft", "--help"],
            ["-m", "edgecraft", "synth", "--help"],
            ["-m", "edgecraft", "profile", "show", "paper"],
        )
        for arguments in checks:
            result = subprocess.run(
                [sys.executable, "-I", *arguments],
                cwd=directory, env=env, capture_output=True, text=True,
            )
            if result.returncode:
                print(result.stdout + result.stderr, file=sys.stderr)
                return result.returncode
        command = Path(sys.executable).parent / ("edgecraft.exe" if os.name == "nt" else "edgecraft")
        result = subprocess.run(
            [str(command), "--help"],
            cwd=directory, env=env, capture_output=True, text=True,
        )
        if result.returncode:
            print(result.stdout + result.stderr, file=sys.stderr)
            return result.returncode
    print("installed package passed (CLI, device manifest, crawler configuration, benchmark runner)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
