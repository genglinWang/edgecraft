#!/usr/bin/env bash
set -euo pipefail

# Stream logs in real-time from an edge device job.

usage() {
  cat <<'EOF'
Stream real-time logs from an edge device job (MVP).

Usage:
  bash stream_logs.sh --edge "$DEVICE_HOST" --key "$SSH_KEY_PATH" --job-id <job_id> [options]

Required:
  --edge         SSH target, e.g. user@device-host
  --key          SSH private key path (IdentityFile)
  --job-id       The job ID to stream logs for

Options:
  --remote-base  Remote base dir on edge (default: /var/lib/edgecraft-edge-runner)
  --stderr       Also stream stderr.log (merged with stdout)
  --follow       Keep streaming until job completes (exit_code file appears)
  --tail LINES   Start with last N lines (default: 0, meaning from beginning)
  --timeout SEC  Max seconds to wait for log file to appear (default: 60)
  --accept-new   Use StrictHostKeyChecking=accept-new (default: off)

Output:
  Prints log content to stdout. Exits with 0 when job completes or stream ends.
EOF
}

EDGE=""
KEY=""
JOB_ID=""
REMOTE_BASE="/var/lib/edgecraft-edge-runner"
STREAM_STDERR=0
FOLLOW=0
TAIL_LINES=0
WAIT_TIMEOUT=60
ACCEPT_NEW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge) EDGE="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --job-id) JOB_ID="${2:-}"; shift 2 ;;
    --remote-base) REMOTE_BASE="${2:-}"; shift 2 ;;
    --stderr) STREAM_STDERR=1; shift 1 ;;
    --follow) FOLLOW=1; shift 1 ;;
    --tail) TAIL_LINES="${2:-0}"; shift 2 ;;
    --timeout) WAIT_TIMEOUT="${2:-60}"; shift 2 ;;
    --accept-new) ACCEPT_NEW=1; shift 1 ;;
    --no-accept-new) ACCEPT_NEW=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$EDGE" || -z "$KEY" || -z "$JOB_ID" ]]; then
  echo "ERROR: --edge/--key/--job-id are required" >&2
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

remote_outbox="${REMOTE_BASE%/}/outbox"
remote_job_dir="$remote_outbox/$JOB_ID"
stdout_log="$remote_job_dir/stdout.log"
stderr_log="$remote_job_dir/stderr.log"
exit_code_file="$remote_job_dir/exit_code"

# Wait for the log file to appear (job may not have started yet)
echo "[stream] Waiting for job $JOB_ID to start..." >&2
deadline=$((SECONDS + WAIT_TIMEOUT))
while true; do
  if ssh "${ssh_opts[@]}" "$EDGE" "test -f '$stdout_log'" 2>/dev/null; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[stream] ERROR: Timeout waiting for log file: $stdout_log" >&2
    exit 1
  fi
  sleep 1
done
echo "[stream] Job started, streaming logs..." >&2

# Build the remote command
if [[ "$FOLLOW" == "1" ]]; then
  # Follow mode: first dump existing content, then tail -f for new content
  # This ensures we don't miss logs even if job completes quickly
  if [[ "$STREAM_STDERR" == "1" ]]; then
    remote_cmd="
# First, output existing content (handles fast-completing jobs)
if [ $TAIL_LINES -gt 0 ]; then
  tail -n $TAIL_LINES '$stdout_log' '$stderr_log' 2>/dev/null || true
else
  cat '$stdout_log' '$stderr_log' 2>/dev/null || true
fi

# If job already completed, we're done
if [ -f '$exit_code_file' ]; then
  exit 0
fi

# Otherwise, follow new content from current position
tail -n 0 -f '$stdout_log' '$stderr_log' &
TAIL_PID=\$!
while true; do
  if [ -f '$exit_code_file' ]; then
    sleep 0.3
    kill \$TAIL_PID 2>/dev/null || true
    break
  fi
  sleep 0.5
done
wait \$TAIL_PID 2>/dev/null || true
"
  else
    remote_cmd="
# First, output existing content (handles fast-completing jobs)
if [ $TAIL_LINES -gt 0 ]; then
  tail -n $TAIL_LINES '$stdout_log' 2>/dev/null || true
else
  cat '$stdout_log' 2>/dev/null || true
fi

# If job already completed, we're done
if [ -f '$exit_code_file' ]; then
  exit 0
fi

# Otherwise, follow new content from current position
tail -n 0 -f '$stdout_log' &
TAIL_PID=\$!
while true; do
  if [ -f '$exit_code_file' ]; then
    sleep 0.3
    kill \$TAIL_PID 2>/dev/null || true
    break
  fi
  sleep 0.5
done
wait \$TAIL_PID 2>/dev/null || true
"
  fi
else
  # Non-follow mode: just cat the current content
  if [[ "$STREAM_STDERR" == "1" ]]; then
    if [[ "$TAIL_LINES" -gt 0 ]]; then
      remote_cmd="tail -n $TAIL_LINES '$stdout_log' '$stderr_log' 2>/dev/null || true"
    else
      remote_cmd="cat '$stdout_log' '$stderr_log' 2>/dev/null || true"
    fi
  else
    if [[ "$TAIL_LINES" -gt 0 ]]; then
      remote_cmd="tail -n $TAIL_LINES '$stdout_log' 2>/dev/null || true"
    else
      remote_cmd="cat '$stdout_log' 2>/dev/null || true"
    fi
  fi
fi

# Execute the remote command
# Use -t for pseudo-terminal to allow Ctrl+C to propagate
ssh "${ssh_opts[@]}" -t "$EDGE" "$remote_cmd" 2>/dev/null || true

# Check if job completed
if ssh "${ssh_opts[@]}" "$EDGE" "test -f '$exit_code_file'" 2>/dev/null; then
  exit_code=$(ssh "${ssh_opts[@]}" "$EDGE" "cat '$exit_code_file'" 2>/dev/null || echo "?")
  echo "" >&2
  echo "[stream] Job $JOB_ID completed with exit_code=$exit_code" >&2
else
  echo "" >&2
  echo "[stream] Stream ended (job may still be running)" >&2
fi
