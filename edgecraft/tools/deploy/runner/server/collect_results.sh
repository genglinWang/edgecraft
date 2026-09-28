#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Collect result tarballs from an edge device outbox (MVP).

Usage:
  bash collect_results.sh --edge "$DEVICE_HOST" --key "$SSH_KEY_PATH" --out /path/to/output_dir [options]

Required:
  --edge         SSH target, e.g. user@device-host
  --key          SSH private key path (IdentityFile)
  --out          Local directory to store downloaded .tar.gz

Options:
  --remote-base  Remote base dir on edge (default: /var/lib/edgecraft-edge-runner)
  --job-id       Only collect a specific job_id tarball
  --delete       Deprecated no-op; permanent remote deletion is disabled
  --extract      Extract each tarball into --out/<job_id>/
  --keep-local N Keep only newest N local entries in --out (default: 20)
  --retries N    Download/verify retries per tarball (default: 3)
  --retry-delay  Seconds between retries (default: 2)
  --purge-job    Deprecated no-op; permanent remote deletion is disabled
  --keep-job     Keep job files on edge (default)
  --accept-new   Use StrictHostKeyChecking=accept-new (default: off)

EOF
}

EDGE=""
KEY=""
OUT=""
REMOTE_BASE="/var/lib/edgecraft-edge-runner"
JOB_ID=""
DELETE_REMOTE=0
EXTRACT=0
ACCEPT_NEW=0
PURGE_JOB=0
RETRIES=3
RETRY_DELAY=2
KEEP_LOCAL=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge) EDGE="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --remote-base) REMOTE_BASE="${2:-}"; shift 2 ;;
    --job-id) JOB_ID="${2:-}"; shift 2 ;;
    --delete) echo "WARN: permanent remote deletion disabled" >&2; DELETE_REMOTE=0; shift 1 ;;
    --extract) EXTRACT=1; shift 1 ;;
    --keep-local) KEEP_LOCAL="${2:-}"; shift 2 ;;
    --retries) RETRIES="${2:-}"; shift 2 ;;
    --retry-delay) RETRY_DELAY="${2:-}"; shift 2 ;;
    --purge-job) echo "WARN: permanent remote purge disabled" >&2; PURGE_JOB=0; shift 1 ;;
    --keep-job|--no-purge) PURGE_JOB=0; shift 1 ;;
    --accept-new) ACCEPT_NEW=1; shift 1 ;;
    --no-accept-new) ACCEPT_NEW=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$EDGE" || -z "$KEY" || -z "$OUT" ]]; then
  echo "ERROR: --edge/--key/--out are required" >&2
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

mkdir -p "$OUT"

ssh_opts=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes)
if [[ "$ACCEPT_NEW" == "1" ]]; then
  ssh_opts+=(-o StrictHostKeyChecking=accept-new)
fi

remote_outbox="${REMOTE_BASE%/}/outbox"

pattern="*.tar.gz"
if [[ -n "$JOB_ID" ]]; then
  pattern="${JOB_ID}.tar.gz"
fi

# List remote tarballs (may be empty)
remote_list="$(ssh "${ssh_opts[@]}" "$EDGE" "ls -1 \"$remote_outbox\"/$pattern 2>/dev/null || true")"
if [[ -z "$remote_list" ]]; then
  echo "No remote results found in $remote_outbox matching: $pattern"
  exit 0
fi

downloaded=0
failed=0
while IFS= read -r remote_file; do
  [[ -z "$remote_file" ]] && continue
  base="$(basename "$remote_file")"
  job_id="${base%.tar.gz}"
  local_path="$OUT/$base"

  ok=0
  attempt=1
  while (( attempt <= RETRIES )); do
    rm -f "$local_path"
    if ! scp "${ssh_opts[@]}" "$EDGE:$remote_file" "$local_path"; then
      echo "WARN: download failed ($job_id) attempt=$attempt/$RETRIES" >&2
      attempt=$((attempt + 1))
      sleep "$RETRY_DELAY"
      continue
    fi

    # Validate tar integrity before extraction.
    if ! tar -tzf "$local_path" >/dev/null 2>&1; then
      echo "WARN: tar validation failed ($job_id) attempt=$attempt/$RETRIES" >&2
      attempt=$((attempt + 1))
      sleep "$RETRY_DELAY"
      continue
    fi

    if [[ "$EXTRACT" == "1" ]]; then
      rm -rf "$OUT/$job_id"
      mkdir -p "$OUT/$job_id"
      if ! tar -C "$OUT/$job_id" -xzf "$local_path"; then
        echo "WARN: tar extract failed ($job_id) attempt=$attempt/$RETRIES" >&2
        attempt=$((attempt + 1))
        sleep "$RETRY_DELAY"
        continue
      fi
    fi

    ok=1
    break
  done

  if [[ "$ok" != "1" ]]; then
    echo "ERROR: failed to collect valid result tar for $job_id after $RETRIES attempts" >&2
    failed=$((failed + 1))
    continue
  fi

  downloaded=$((downloaded + 1))

  if [[ "$DELETE_REMOTE" == "1" ]]; then
    ssh "${ssh_opts[@]}" "$EDGE" "PATH=\"\$HOME/.local/edgecraft-safe-bin:\$PATH\" rm -f \"$remote_file\""
  fi

  if [[ "$PURGE_JOB" == "1" ]]; then
    remote_base="${REMOTE_BASE%/}"
    # Best-effort purge: do not fail collection if remote cleanup hits permissions.
    ssh "${ssh_opts[@]}" "$EDGE" "
set +e
export PATH=\"\$HOME/.local/edgecraft-safe-bin:\$PATH\"
rm -rf \
\"$remote_base/inbox/${job_id}.tar.gz\" \
\"$remote_base/tmp/${job_id}.tar.gz\" \
\"$remote_base/outbox/${job_id}.tar.gz\" \
\"$remote_base/outbox/$job_id\" \
\"$remote_base/work/${job_id}.tar.gz\" \
\"$remote_base/work/$job_id\" \
\"$remote_base/done/$job_id\" \
\"$remote_base/failed/$job_id\" >/dev/null 2>&1 || true
"
  fi
done <<< "$remote_list"

# Keep local collection directory bounded to avoid accumulating stale history.
if [[ "${KEEP_LOCAL:-0}" =~ ^[0-9]+$ ]] && (( KEEP_LOCAL > 0 )); then
  mapfile -t local_entries < <(ls -1dt "$OUT"/* 2>/dev/null || true)
  if (( ${#local_entries[@]} > KEEP_LOCAL )); then
    for stale_path in "${local_entries[@]:KEEP_LOCAL}"; do
      rm -rf "$stale_path"
    done
  fi
fi

echo "Downloaded $downloaded file(s) into: $OUT"
if (( failed > 0 )); then
  echo "Failed to collect $failed file(s)." >&2
  exit 1
fi
