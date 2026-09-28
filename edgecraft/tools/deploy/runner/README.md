# Edge runner

This directory contains the controller and device-side scripts used by
EdgeCraft's hardware-in-the-loop verifier. The controller transfers a trusted
job bundle over SSH, optionally starts a one-shot runner, streams logs, and
collects a result archive.

Supply the target device and SSH key explicitly:

```bash
export DEVICE_HOST=user@device-host
export SSH_KEY_PATH=/path/to/private_key

bash edge-runner/server/run_job_and_collect.sh \
  --edge "$DEVICE_HOST" \
  --key "$SSH_KEY_PATH" \
  --job-dir ./job \
  --run-inline \
  --collect-extract \
  --collect-keep-job
```

The scripts always pass the selected identity with SSH's `-i` option and
`IdentitiesOnly=yes`. They do not contain a device address, account, key, or
container image. Normal host-key verification remains enabled; `--accept-new`
may be used for reviewed first contact. The default controller path preserves
existing remote jobs and results.

## Job bundle

A job directory contains:

- `run.sh`: required entry point;
- `meta.env`: optional trusted metadata, including `JOB_ID`, `TIMEOUT_SECS`,
  and `WORKDIR`;
- `payload/`: optional model, code, configuration, and evaluation data.

`meta.env` and `run.sh` are executed by the target shell. Only submit bundles
from a trusted controller and use a dedicated, least-privilege device account.

## Optional daemon installation

Linux devices with systemd can install the inbox watcher:

```bash
cd edge-runner/edge
sudo EDGE_USER=edgeuser bash install.sh
```

The daemon stores state below its configured `BASE_DIR`. The optional cleanup
timer removes stale runner-owned artifacts according to the retention settings
in the installed environment file. Review those settings before enabling the
timer on a shared device. The one-shot `--run-inline` flow does not require a
daemon installation. Optional push-back likewise requires an explicit
`PUSH_SSH_KEY` and uses `IdentitiesOnly=yes`.

## Result contract

Each result archive contains stdout, stderr, exit status, timestamps, and files
created in the job output directory. `collect_results.sh` validates archive
integrity before extraction. Collection keeps the remote job by default.

This is a research prototype, not a hardened multi-user execution service. See
the repository security notes before exposing it beyond a controlled lab.
