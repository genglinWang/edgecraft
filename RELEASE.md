# Release checks

EdgeCraft source releases include the implementation, tests, small examples,
configuration templates, and documentation. Keep credentials, device addresses,
private paths, datasets, trained models, and experiment outputs in local storage.

Run the source checks and tests before publishing:

```bash
python scripts/release_smoke.py
python -m pytest -q
git diff --check
```

The release scan checks common credential and infrastructure patterns and
excludes generated artifacts from the source tree. Review the staged diff
as well: a pattern scan is not a substitute for checking the intended files.

## Package verification

```bash
python -m pip install build setuptools wheel
python -m build --wheel --no-isolation
python -m pip install --no-deps dist/*.whl
python scripts/check_installed_package.py
```

The installation check runs from a temporary directory with Python's isolated
mode. It verifies the installed CLI and the device manifest, crawler
configuration, and benchmark runner bundled with the package. Install
`requirements-test.txt` first for this dependency-light check; full synthesis
uses the dependencies in `pyproject.toml`.

Published author names, repository URLs, normal commits, and release tags are
part of the public project metadata. Check new history for private files before
pushing; publish only the release branch.
