#!/usr/bin/env python3
"""Dependency-free structural check for an EdgeCraft source release.

It validates the public entry point, claim-relevant mechanism paths, source
syntax, synthetic fixture labels, documentation boundary, and release scan.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = (
    "edgecraft/__main__.py",
    "edgecraft/agent/search/refinement_tree.py",
    "edgecraft/agent/search/branch_judgment.py",
    "edgecraft/agent/search/verification.py",
    "edgecraft/agent/nodes/pipeline_executor.py",
    "edgecraft/knowledge/calibration_store.py",
    "edgecraft/knowledge/compatibility/rule_store.py",
    "edgecraft/scheduler/scheduler.py",
    "edgecraft/api/auth.py",
    "edgecraft/api/resources.py",
    "edgecraft/utils/p1_guard.py",
    "edgecraft/utils/p1_probe_runner.py",
    "edgecraft/utils/subprocess_env.py",
    "tests/test_api_tenant_routes.py",
    "tests/test_constraint_tree.py",
    "tests/test_paper_mechanism_integration.py",
    "tests/test_p1_construction_guard.py",
    "tests/test_verifier_policy.py",
    "tests/test_scheduler_policy.py",
    "tests/test_sdk_contract.py",
    "tests/test_trial_bank.py",
    "scripts/check_release.py",
    "scripts/check_installed_package.py",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "RELEASE.md",
    "PAPER_CODE_MAP.md",
    "REPRODUCIBILITY.md",
    "SECURITY.md",
)


def _check_python_syntax(errors: list[str]) -> int:
    checked = 0
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if any(part in {".git", ".venv", "venv", "__pycache__", "build", "dist"} for part in relative.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"invalid Python source {relative}: {exc}")
        checked += 1
    return checked


def _check_public_interface(errors: list[str]) -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'edgecraft = "edgecraft.cli:main"' not in project:
        errors.append("pyproject.toml does not expose the edgecraft command")
    if project.count('\nedgecraft = "') != 1:
        errors.append("pyproject.toml must expose one edgecraft command")

    profiles = (ROOT / "edgecraft/config/profiles.py").read_text(encoding="utf-8")
    required_switches = (
        '"BRANCH_SELECTION_MODE": "llm_strict"',
        '"VERIFIER_MODE": "ladder"',
        '"EXECUTION_BACKEND": "scheduler"',
        '"SCHEDULER_POLICY": "cost_aware"',
        '"REQUIRE_TOKEN_TELEMETRY": True',
        '"L1_FALSE_PRUNE_SAMPLE": False',
        '"paper-audit": PAPER_AUDIT_PROFILE',
        '"paper": 24',
    )
    for switch in required_switches:
        if switch not in profiles:
            errors.append(f"paper profile is missing {switch}")


def _check_fixtures(errors: list[str]) -> None:
    fixtures = {
        "fixtures/synthetic_verified_rules.json": "edgecraft_public_rules_v1",
        "fixtures/synthetic_calibration.json": "edgecraft_public_calibration_v1",
    }
    for relative, schema in fixtures.items():
        path = ROOT / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"unreadable fixture {relative}: {exc}")
            continue
        if payload.get("schema_version") != schema:
            errors.append(f"fixture {relative} has the wrong schema")
        if payload.get("synthetic") is not True:
            errors.append(f"fixture {relative} is not explicitly marked synthetic")


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            errors.append(f"required release path is missing: {relative}")

    python_count = _check_python_syntax(errors)
    _check_public_interface(errors)
    _check_fixtures(errors)

    release = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_release.py")],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if release.returncode != 0:
        detail = (release.stdout + release.stderr).strip()
        errors.append(f"release-boundary scan failed:\n{detail}")

    if errors:
        for item in errors:
            print(f"FAIL: {item}")
        print(f"release smoke check failed with {len(errors)} issue(s)", file=sys.stderr)
        return 1

    print(
        "release smoke check passed "
        f"({python_count} Python files; public CLI, paper profile, synthetic fixtures, "
        "and source release checked)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
