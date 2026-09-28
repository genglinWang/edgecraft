#!/usr/bin/env bash
set -euo pipefail

# EdgeCraft Edge Runner MVP installer (run on edge device)
#
# Installs:
# - /usr/local/bin/edgecraft-edge-runner
# - /usr/local/bin/edgecraft-edge-cleanup
# - /etc/edgecraft-edge-runner.env
# - /etc/systemd/system/edgecraft-edge-runner.{service,path}
# - /etc/systemd/system/edgecraft-edge-cleanup.{service,timer}
# - creates /var/lib/edgecraft-edge-runner owned by EDGE_USER

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "ERROR: please run with sudo: sudo bash install.sh" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EDGE_USER="${EDGE_USER:-}"
if [[ -z "$EDGE_USER" ]]; then
  # Best-effort default: the non-root user who invoked sudo
  EDGE_USER="${SUDO_USER:-}"
fi
if [[ -z "$EDGE_USER" ]]; then
  echo "ERROR: EDGE_USER is empty. Run: sudo EDGE_USER=edgeuser bash install.sh" >&2
  exit 1
fi

EDGE_HOME="$(getent passwd "$EDGE_USER" | cut -d: -f6)"
if [[ -z "$EDGE_HOME" ]]; then
  echo "ERROR: cannot resolve home for user: $EDGE_USER" >&2
  exit 1
fi

BIN_SRC="$HERE/bin/edgecraft-edge-runner"
BIN_CLEANUP_SRC="$HERE/bin/edgecraft-edge-cleanup"
SERVICE_SRC="$HERE/systemd/edgecraft-edge-runner@.service"
PATH_SRC="$HERE/systemd/edgecraft-edge-runner@.path"
CLEANUP_SERVICE_SRC="$HERE/systemd/edgecraft-edge-cleanup.service"
CLEANUP_TIMER_SRC="$HERE/systemd/edgecraft-edge-cleanup.timer"

if [[ ! -f "$BIN_SRC" ]]; then
  echo "ERROR: missing $BIN_SRC" >&2
  exit 1
fi
if [[ ! -f "$BIN_CLEANUP_SRC" ]]; then
  echo "ERROR: missing $BIN_CLEANUP_SRC" >&2
  exit 1
fi

install -m 0755 "$BIN_SRC" /usr/local/bin/edgecraft-edge-runner
install -m 0755 "$BIN_CLEANUP_SRC" /usr/local/bin/edgecraft-edge-cleanup

if [[ ! -f /etc/edgecraft-edge-runner.env ]]; then
  cat > /etc/edgecraft-edge-runner.env <<EOF
# EdgeCraft edge runner config (MVP)
EDGE_USER=$EDGE_USER
BASE_DIR=/var/lib/edgecraft-edge-runner

# Optional: automatically push results back to controller via scp.
# Set PUSH_RESULTS=1 to enable.
PUSH_RESULTS=0

# If PUSH_RESULTS=1, you must set both:
# PUSH_DEST="user@host:/absolute/path/on/controller/"
PUSH_DEST=
# PUSH_SSH_KEY=/path/to/private_key
PUSH_SSH_KEY=
# Set to 1 only for first contact with a reviewed host.
PUSH_ACCEPT_NEW=0

# Daily cleanup policy (executed by systemd timer)
# Keep newest N edgecraft_*.tar.gz in outbox
CLEANUP_KEEP_RECENT_OUTBOX=5
# Remove stale artifacts older than N minutes
CLEANUP_AGE_MINUTES=1440

# Per-job runner cleanup policy (inside edgecraft-edge-runner process loop)
RUNNER_KEEP_RECENT_OUTBOX=5
RUNNER_CLEANUP_AGE_MINUTES=120
RUNNER_MIN_FREE_MB=2048
EOF
  chmod 0644 /etc/edgecraft-edge-runner.env
fi

install -m 0644 "$SERVICE_SRC" /etc/systemd/system/edgecraft-edge-runner@.service
install -m 0644 "$PATH_SRC" /etc/systemd/system/edgecraft-edge-runner@.path
install -m 0644 "$CLEANUP_SERVICE_SRC" /etc/systemd/system/edgecraft-edge-cleanup.service
install -m 0644 "$CLEANUP_TIMER_SRC" /etc/systemd/system/edgecraft-edge-cleanup.timer

# State dir
BASE_DIR="$(. /etc/edgecraft-edge-runner.env; echo "${BASE_DIR:-/var/lib/edgecraft-edge-runner}")"
mkdir -p "$BASE_DIR"/{inbox,work,outbox,done,failed,tmp}
chown -R "$EDGE_USER":"$EDGE_USER" "$BASE_DIR"
chmod 0755 "$BASE_DIR"

systemctl daemon-reload
systemctl enable --now "edgecraft-edge-runner@${EDGE_USER}.path"
systemctl enable --now edgecraft-edge-cleanup.timer

echo "OK: installed EdgeCraft edge runner MVP."
echo " - EDGE_USER: $EDGE_USER"
echo " - BASE_DIR : $BASE_DIR"
echo " - CLEANUP  : systemd timer enabled (daily at 03:00)"
echo
echo "Next:"
echo " - Put job bundles into: $BASE_DIR/inbox/"
echo " - Watch logs: journalctl -u edgecraft-edge-runner.service -f"
