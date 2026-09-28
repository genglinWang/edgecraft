# Contributing to EdgeCraft

Use an issue to describe a bug or a proposed change. For a bug report, include
the command, Python and runtime versions, device family, and a small example.
Remove credentials and private data from logs before sharing them.

## Development checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python scripts/release_smoke.py
python -m pytest -q
PYTHONPATH=. python examples/offline_mechanisms.py
git diff --check
```

Keep changes focused and add a regression test when changing behavior.
Contributions are provided under the project's AGPL-3.0-only license. Preserve
applicable copyright and third-party license notices.
Changes to synthesis or verification should retain the distinction between
LLM proposals, device measurements, and verifier decisions. Hardware tests
use locally supplied datasets, runtime images, and SSH credentials.

See [RELEASE.md](RELEASE.md) for the source and packaging checks and
[SECURITY.md](SECURITY.md) for execution assumptions.
