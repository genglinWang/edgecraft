#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Submit a job bundle to an edge device inbox (MVP).

Usage:
  bash submit_job.sh --edge "$DEVICE_HOST" --key "$SSH_KEY_PATH" --job-dir /path/to/job_dir [options]

Required:
  --edge        SSH target, e.g. user@device-host
  --key         SSH private key path (IdentityFile)
  --job-dir     Directory containing at least run.sh, optionally meta.env and payload/

Options:
  --job-id      Override JOB_ID (also used for the tarball file name)
  --remote-base Remote base dir on edge (default: /var/lib/edgecraft-edge-runner)
  --preflight-cleanup / --no-preflight-cleanup
                Permanent cleanup is disabled; preserve prior jobs (default)
  --keep-recent-outbox N
                Keep N newest edgecraft_* tarballs in outbox (default: 20)
  --cleanup-age-min M
                Remove stale profiling payloads older than M minutes (default: 120)
  --accept-new  Use StrictHostKeyChecking=accept-new (default: off)

Output:
  Prints the remote tarball path on success.
EOF
}

EDGE=""
KEY=""
JOB_DIR=""
JOB_ID=""
REMOTE_BASE="/var/lib/edgecraft-edge-runner"
ACCEPT_NEW=0
PREFLIGHT_CLEANUP=0
KEEP_RECENT_OUTBOX=20
CLEANUP_AGE_MIN=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge) EDGE="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --job-dir) JOB_DIR="${2:-}"; shift 2 ;;
    --job-id) JOB_ID="${2:-}"; shift 2 ;;
    --remote-base) REMOTE_BASE="${2:-}"; shift 2 ;;
    --preflight-cleanup) echo "WARN: destructive preflight cleanup disabled" >&2; PREFLIGHT_CLEANUP=0; shift 1 ;;
    --no-preflight-cleanup) PREFLIGHT_CLEANUP=0; shift 1 ;;
    --keep-recent-outbox) KEEP_RECENT_OUTBOX="${2:-}"; shift 2 ;;
    --cleanup-age-min) CLEANUP_AGE_MIN="${2:-}"; shift 2 ;;
    --accept-new) ACCEPT_NEW=1; shift 1 ;;
    --no-accept-new) ACCEPT_NEW=0; shift 1 ;;
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
if [[ ! -d "$JOB_DIR" ]]; then
  echo "ERROR: job-dir not found: $JOB_DIR" >&2
  exit 2
fi
if [[ ! -f "$JOB_DIR/run.sh" ]]; then
  echo "ERROR: missing $JOB_DIR/run.sh" >&2
  exit 2
fi

ssh_opts=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes)
if [[ "$ACCEPT_NEW" == "1" ]]; then
  ssh_opts+=(-o StrictHostKeyChecking=accept-new)
fi

tmp_root="$(mktemp -d)"
cleanup() { rm -rf "$tmp_root"; }
trap cleanup EXIT

bundle_dir="$tmp_root/bundle"
mkdir -p "$bundle_dir"

# Copy user content.
cp -a "$JOB_DIR/run.sh" "$bundle_dir/run.sh"
if [[ -f "$JOB_DIR/meta.env" ]]; then
  cp -a "$JOB_DIR/meta.env" "$bundle_dir/meta.env"
fi
if [[ -d "$JOB_DIR/payload" ]]; then
  mkdir -p "$bundle_dir/payload"
  cp -a "$JOB_DIR/payload/." "$bundle_dir/payload/"
fi

# Determine job_id
if [[ -z "$JOB_ID" && -f "$bundle_dir/meta.env" ]]; then
  # Best-effort parse JOB_ID=... from meta.env
  JOB_ID="$(grep -E '^JOB_ID=' "$bundle_dir/meta.env" | head -n 1 | cut -d= -f2- || true)"
fi
if [[ -z "$JOB_ID" ]]; then
  JOB_ID="job-$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM$RANDOM"
fi

# Ensure meta.env exists and contains JOB_ID.
if [[ ! -f "$bundle_dir/meta.env" ]]; then
  touch "$bundle_dir/meta.env"
fi
if ! grep -qE '^JOB_ID=' "$bundle_dir/meta.env"; then
  echo "JOB_ID=$JOB_ID" >> "$bundle_dir/meta.env"
fi

# Make sure run.sh is executable inside the bundle
chmod +x "$bundle_dir/run.sh" || true

tarball="$tmp_root/${JOB_ID}.tar.gz"
tar -C "$bundle_dir" -czf "$tarball" .

tar_listing="$(tar -tzf "$tarball")"

# Guardrail: ensure bundle contains run.sh at top-level before upload.  Avoid
# `tar | grep -q` under pipefail: grep can exit early and make tar report
# SIGPIPE even when the matching entry exists.
if ! grep -Eq '^(\./)?run\.sh$' <<< "$tar_listing"; then
  echo "ERROR: generated tarball does not contain top-level run.sh: $tarball" >&2
  echo "DEBUG: bundle_dir=$bundle_dir" >&2
  echo "DEBUG: bundle_dir listing:" >&2
  find "$bundle_dir" -maxdepth 2 -printf '%M %s %p\n' 2>/dev/null | head -120 >&2 || true
  echo "DEBUG: tarball listing:" >&2
  printf '%s\n' "$tar_listing" | head -120 >&2 || true
  exit 2
fi

remote_inbox="${REMOTE_BASE%/}/inbox"
remote_tmp="${REMOTE_BASE%/}/tmp"
remote_path="$remote_inbox/${JOB_ID}.tar.gz"
# Keep an in-progress upload outside the runner's job-tarball namespace.  The
# final rename is the only point at which the inbox may observe this bundle.
remote_tmp_path="$remote_tmp/.upload-${JOB_ID}.tar.gz.part"

# Ensure directories exist
ssh "${ssh_opts[@]}" "$EDGE" "mkdir -p \"$remote_inbox\" \"$remote_tmp\""

# Best-effort preflight cleanup to reduce "No space left on device" during scp.
# Cleans stale EdgeCraft artifacts (profile/deploy tarballs and extracted directories).
if [[ "$PREFLIGHT_CLEANUP" == "1" ]]; then
  remote_base="${REMOTE_BASE%/}"
  ssh "${ssh_opts[@]}" "$EDGE" "
set +e
BASE=\"$remote_base\"
export PATH=\"\$HOME/.local/edgecraft-safe-bin:\$PATH\"
mkdir -p \"\$BASE/outbox\" \"\$BASE/tmp\" \"\$BASE/inbox\" \"\$BASE/work\" \"\$BASE/done\" \"\$BASE/failed\"
rm -f \"\$BASE/tmp\"/edgecraft_*.tar.gz >/dev/null 2>&1 || true
ls -1t \"\$BASE/outbox\"/edgecraft_*.tar.gz 2>/dev/null | awk 'NR>${KEEP_RECENT_OUTBOX}' | xargs -r rm -f >/dev/null 2>&1 || true
find \"\$BASE/outbox\" -maxdepth 1 -type d \\( -name 'edgecraft_profile_*' -o -name 'edgecraft_deploy_*' \\) -mmin +${CLEANUP_AGE_MIN} -exec rm -rf {} + >/dev/null 2>&1 || true
find \"\$BASE/work\" \"\$BASE/done\" \"\$BASE/failed\" -maxdepth 1 -type d -name 'edgecraft_*' -mmin +${CLEANUP_AGE_MIN} -exec rm -rf {} + >/dev/null 2>&1 || true
find \"\$BASE/inbox\" -maxdepth 1 -type f -name 'edgecraft_*.tar.gz' -mmin +${CLEANUP_AGE_MIN} -exec rm -f {} + >/dev/null 2>&1 || true
"
fi

# Upload to tmp first, then atomically move to inbox
# This prevents the path watcher from triggering on incomplete files
scp "${ssh_opts[@]}" "$tarball" "$EDGE:$remote_tmp_path"
ssh "${ssh_opts[@]}" "$EDGE" "mv \"$remote_tmp_path\" \"$remote_path\""

echo "$remote_path"
