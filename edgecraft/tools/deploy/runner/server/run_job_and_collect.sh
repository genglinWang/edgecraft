#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: submit a job, wait for result tarball, collect it.

usage() {
  cat <<'EOF'
End-to-end helper: submit a job to an edge device, wait for completion,
then collect the result tarball locally.

Usage:
  bash run_job_and_collect.sh \
    --edge "$DEVICE_HOST" \
    --key "$SSH_KEY_PATH" \
    --job-dir /path/to/job_dir \
    [options]

Options:
  --remote-base DIR     Remote base dir (default: /var/lib/edgecraft-edge-runner)
  --wait-secs SEC       Max seconds to wait for result (default: 600)
  --poll-interval SEC   Interval between checks (default: 5)
  --collect-out DIR     Local dir to store results (default: ./collected_results)
  --collect-retries N   Retries for download/verify in collect step (default: 3)
  --collect-retry-delay SEC  Delay between collect retries (default: 2)
  --collect-extract     Extract tarball after download
  --collect-delete      Deprecated no-op; permanent remote deletion is disabled
  --preflight-cleanup   Deprecated no-op; permanent remote deletion is disabled
  --no-preflight-cleanup  Preserve all pre-existing remote jobs (default)
  --stream-grace-secs SEC  Max seconds to wait for log stream after result exists (default: 5)
  --accept-new          StrictHostKeyChecking=accept-new (default: off)
  --no-accept-new       Disable accept-new
  --stream              Stream logs in real-time while job is running
  --stream-stderr       Also stream stderr when using --stream
  --run-inline          Process the submitted job once over SSH instead of relying on a daemon
  --inline-min-free-mb N  Minimum free space for the inline runner (default: 2048)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EDGE=""
KEY=""
JOB_DIR=""
REMOTE_BASE="/var/lib/edgecraft-edge-runner"
WAIT_SECS=600
POLL_INTERVAL=5
COLLECT_OUT="./collected_results"
COLLECT_RETRIES=3
COLLECT_RETRY_DELAY=2
COLLECT_EXTRACT=0
COLLECT_DELETE=0
ACCEPT_NEW=0
COLLECT_PURGE=0
PREFLIGHT_CLEANUP=0
STREAM_LOGS=0
STREAM_STDERR=0
STREAM_GRACE_SECS=5
RUN_INLINE=0
INLINE_MIN_FREE_MB=2048

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge) EDGE="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --job-dir) JOB_DIR="${2:-}"; shift 2 ;;
    --remote-base) REMOTE_BASE="${2:-}"; shift 2 ;;
    --wait-secs) WAIT_SECS="${2:-}"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="${2:-}"; shift 2 ;;
    --collect-out) COLLECT_OUT="${2:-}"; shift 2 ;;
    --collect-retries) COLLECT_RETRIES="${2:-}"; shift 2 ;;
    --collect-retry-delay) COLLECT_RETRY_DELAY="${2:-}"; shift 2 ;;
    --collect-extract) COLLECT_EXTRACT=1; shift 1 ;;
    --collect-delete) echo "[run-job] permanent remote deletion disabled; keeping job" >&2; shift 1 ;;
    --preflight-cleanup) echo "[run-job] destructive preflight cleanup disabled" >&2; shift 1 ;;
    --no-preflight-cleanup) PREFLIGHT_CLEANUP=0; shift 1 ;;
    --stream-grace-secs) STREAM_GRACE_SECS="${2:-5}"; shift 2 ;;
    --collect-purge) echo "[run-job] permanent remote purge disabled" >&2; COLLECT_PURGE=0; shift 1 ;;
    --collect-keep-job|--collect-no-purge) COLLECT_PURGE=0; shift 1 ;;
    --accept-new) ACCEPT_NEW=1; shift 1 ;;
    --no-accept-new) ACCEPT_NEW=0; shift 1 ;;
    --stream) STREAM_LOGS=1; shift 1 ;;
    --stream-stderr) STREAM_STDERR=1; shift 1 ;;
    --run-inline) RUN_INLINE=1; shift 1 ;;
    --inline-min-free-mb) INLINE_MIN_FREE_MB="${2:-2048}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$EDGE" || -z "$KEY" || -z "$JOB_DIR" ]]; then
  echo "ERROR: --edge/--key/--job-dir are required" >&2
  usage
  exit 2
fi
if [[ ! -f "$KEY" ]]; then
  echo "ERROR: key not found: $KEY" >&2
  exit 2
fi
if [[ ! "$EDGE" =~ ^([A-Za-z0-9_][A-Za-z0-9._-]*@)?([A-Za-z0-9_][A-Za-z0-9._-]*|\[[0-9A-Fa-f:.]+\])$ ]]; then
  echo "ERROR: --edge must use the [user@]host form" >&2
  exit 2
fi

ssh_opts=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes)
if [[ "$ACCEPT_NEW" == "1" ]]; then
  ssh_opts+=(-o StrictHostKeyChecking=accept-new)
fi

# Submit job
submit_cmd=(
  bash "$SCRIPT_DIR/submit_job.sh"
  --edge "$EDGE"
  --key "$KEY"
  --job-dir "$JOB_DIR"
  --remote-base "$REMOTE_BASE"
  --no-preflight-cleanup
)
if [[ "$ACCEPT_NEW" != "1" ]]; then
  submit_cmd+=(--no-accept-new)
fi

remote_path="$("${submit_cmd[@]}")"
job_file="$(basename "$remote_path")"
job_id="${job_file%.tar.gz}"

echo "[run-job] Submitted job_id=$job_id (remote: $remote_path)"

if [[ "$RUN_INLINE" == "1" ]]; then
  runner_src="$(cd "$SCRIPT_DIR/.." && pwd)/edge/bin/edgecraft-edge-runner"
  if [[ ! -f "$runner_src" ]]; then
    echo "[run-job] ERROR: inline runner not found: $runner_src" >&2
    exit 2
  fi
  remote_runner="${REMOTE_BASE%/}/tmp/edgecraft-edge-runner-inline"
  scp "${ssh_opts[@]}" "$runner_src" "$EDGE:$remote_runner"
  ssh "${ssh_opts[@]}" "$EDGE" "chmod +x '$remote_runner' && mkdir -p '${REMOTE_BASE%/}/tmp' && TMPDIR='${REMOTE_BASE%/}/tmp' RUNNER_MIN_FREE_MB='$INLINE_MIN_FREE_MB' BASE_DIR='${REMOTE_BASE%/}' '$remote_runner' process-inbox"
fi

# Start streaming logs in background if requested
STREAM_PID=""
if [[ "$STREAM_LOGS" == "1" ]]; then
  stream_cmd=(
    bash "$SCRIPT_DIR/stream_logs.sh"
    --edge "$EDGE"
    --key "$KEY"
    --job-id "$job_id"
    --remote-base "$REMOTE_BASE"
    --follow
    --timeout "$WAIT_SECS"
  )
  if [[ "$STREAM_STDERR" == "1" ]]; then
    stream_cmd+=(--stderr)
  fi
  if [[ "$ACCEPT_NEW" != "1" ]]; then
    stream_cmd+=(--no-accept-new)
  fi
  echo "[run-job] Starting log stream..."
  "${stream_cmd[@]}" &
  STREAM_PID=$!
fi

# Cleanup function to kill stream on exit
cleanup_stream() {
  if [[ -n "$STREAM_PID" ]] && kill -0 "$STREAM_PID" 2>/dev/null; then
    kill "$STREAM_PID" 2>/dev/null || true
    wait "$STREAM_PID" 2>/dev/null || true
  fi
}
trap cleanup_stream EXIT

remote_outbox="${REMOTE_BASE%/}/outbox"
deadline=$((SECONDS + WAIT_SECS))

if [[ "$STREAM_LOGS" != "1" ]]; then
  echo "[run-job] Waiting for result in $remote_outbox/$job_file ..."
fi
while true; do
  if ssh "${ssh_opts[@]}" "$EDGE" "test -f '$remote_outbox/$job_file'"; then
    if [[ "$STREAM_LOGS" != "1" ]]; then
      echo "[run-job] Result available."
    fi
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[run-job] Timeout after ${WAIT_SECS}s; result not ready yet." >&2
    echo "You can run collect_results.sh later with --job-id $job_id." >&2
    exit 1
  fi
  sleep "$POLL_INTERVAL"
done

# Wait briefly for stream to finish (it will exit when exit_code file appears).
# Some edge jobs produce the result tarball before the log-follow SSH session
# notices completion. Do not let log streaming block result collection forever.
if [[ -n "$STREAM_PID" ]] && kill -0 "$STREAM_PID" 2>/dev/null; then
  stream_deadline=$((SECONDS + STREAM_GRACE_SECS))
  while kill -0 "$STREAM_PID" 2>/dev/null; do
    if (( SECONDS >= stream_deadline )); then
      echo "[run-job] Log stream still active after ${STREAM_GRACE_SECS}s; continuing to collect." >&2
      kill "$STREAM_PID" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  wait "$STREAM_PID" 2>/dev/null || true
fi

collect_cmd=(
  bash "$SCRIPT_DIR/collect_results.sh"
  --edge "$EDGE"
  --key "$KEY"
  --remote-base "$REMOTE_BASE"
  --out "$COLLECT_OUT"
  --job-id "$job_id"
  --retries "$COLLECT_RETRIES"
  --retry-delay "$COLLECT_RETRY_DELAY"
)
if [[ "$COLLECT_EXTRACT" == "1" ]]; then
  collect_cmd+=(--extract)
fi
if [[ "$COLLECT_DELETE" == "1" ]]; then
  collect_cmd+=(--delete)
fi
if [[ "$COLLECT_PURGE" != "1" ]]; then
  collect_cmd+=(--keep-job)
fi
if [[ "$ACCEPT_NEW" != "1" ]]; then
  collect_cmd+=(--no-accept-new)
fi

"${collect_cmd[@]}"

echo "[run-job] Job $job_id finished. Results in $COLLECT_OUT"
