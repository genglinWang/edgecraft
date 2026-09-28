import hashlib
import json
from pathlib import Path
import subprocess
import sys

from edgecraft.utils.p1_guard import validate_p1_guard_attestation


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "edgecraft" / "utils" / "p1_probe_runner.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(candidate: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--script",
            str(candidate),
            "--",
            "--probe",
            "efficiency",
        ],
        cwd=str(candidate.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])
    return completed, payload


def test_controller_guard_attests_initialized_probe_source(tmp_path) -> None:
    candidate = tmp_path / "train.py"
    candidate.write_text(
        """import json
print(json.dumps({
    "status": "success",
    "probe": "efficiency",
    "training_steps": 0,
    "model_path": "outputs/initialized.onnx",
}))
""",
        encoding="utf-8",
    )
    completed, payload = _run(candidate)
    assert completed.returncode == 0
    attestation = payload["p1_guard_attestation"]
    checked = validate_p1_guard_attestation(
        attestation,
        candidate_source_sha256=_sha256(candidate),
        guard_source_sha256=_sha256(RUNNER),
    )
    assert checked["valid"] is True
    assert checked["training_mutation_count"] == 0


def test_controller_guard_blocks_training_call_in_efficiency_path(tmp_path) -> None:
    candidate = tmp_path / "train.py"
    candidate.write_text(
        """import json
class CandidateModel:
    def fit(self):
        return None

CandidateModel().fit()
print(json.dumps({"status": "success", "probe": "efficiency", "training_steps": 0}))
""",
        encoding="utf-8",
    )
    completed, payload = _run(candidate)
    assert completed.returncode != 0
    assert payload["status"] == "error"
    attestation = payload["p1_guard_attestation"]
    assert attestation["status"] == "blocked"
    assert attestation["training_mutation_count"] == 1
    checked = validate_p1_guard_attestation(
        attestation,
        candidate_source_sha256=_sha256(candidate),
        guard_source_sha256=_sha256(RUNNER),
    )
    assert checked["valid"] is False


def test_controller_guard_imports_candidate_sibling_modules(tmp_path) -> None:
    (tmp_path / "loader.py").write_text("def describe():\n    return 'local-data'\n")
    candidate = tmp_path / "train.py"
    candidate.write_text(
        "import json\nfrom loader import describe\n"
        "print(json.dumps({'status': 'success', 'probe': 'efficiency', "
        "'training_steps': 0, 'description': describe()}))\n"
    )
    completed, payload = _run(candidate)
    assert completed.returncode == 0
    assert payload["description"] == "local-data"
    assert payload["p1_guard_attestation"]["status"] == "passed"


def test_controller_guard_blocks_child_process_escape(tmp_path) -> None:
    candidate = tmp_path / "train.py"
    candidate.write_text(
        """import json
import subprocess
import sys
subprocess.run([sys.executable, "-c", "pass"], check=True)
print(json.dumps({"status": "success", "probe": "efficiency", "training_steps": 0}))
""",
        encoding="utf-8",
    )
    completed, payload = _run(candidate)
    assert completed.returncode != 0
    assert payload["p1_guard_attestation"]["status"] == "blocked"
    assert payload["p1_guard_attestation"]["observed_training_mutations"] == [
        "subprocess.run"
    ]


def test_controller_guard_binds_pre_execution_source(tmp_path) -> None:
    candidate = tmp_path / "train.py"
    candidate.write_text(
        """import json
from pathlib import Path
Path(__file__).write_text("# changed during construction\\n", encoding="utf-8")
print(json.dumps({"status": "success", "probe": "efficiency", "training_steps": 0}))
""",
        encoding="utf-8",
    )
    source_before = _sha256(candidate)
    completed, payload = _run(candidate)
    assert completed.returncode != 0
    attestation = payload["p1_guard_attestation"]
    assert attestation["candidate_source_sha256"] == source_before
    assert attestation["candidate_source_unchanged"] is False
    checked = validate_p1_guard_attestation(
        attestation,
        candidate_source_sha256=source_before,
        guard_source_sha256=_sha256(RUNNER),
    )
    assert checked["valid"] is False


def test_candidate_zero_step_json_cannot_replace_guard_attestation() -> None:
    checked = validate_p1_guard_attestation(
        {},
        candidate_source_sha256="candidate-source",
        guard_source_sha256="guard-source",
    )
    assert checked["valid"] is False
