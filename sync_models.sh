#!/usr/bin/env bash
# =============================================================================
# sync_models.sh — Transfer locally trained ML models to VPS
#
# Run from your LOCAL machine AFTER training (python main.py --train).
# Avoids re-training on VPS (saves 20–30 min and VPS RAM).
#
# Usage:
#   bash sync_models.sh <VPS_IP> [--user <username>] [--port <ssh_port>]
#
# Examples:
#   bash sync_models.sh 65.21.100.42
#   bash sync_models.sh 65.21.100.42 --user antho --port 2222
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GRN}[OK]${NC}    $*"; }
info() { echo -e "        $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Args ─────────────────────────────────────────────────────────────────────
VPS_IP="${1:-}"
VPS_USER="antho"
SSH_PORT="22"

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) VPS_USER="$2"; shift ;;
    --port) SSH_PORT="$2"; shift ;;
    *) ;;
  esac
  shift
done

[[ -n "$VPS_IP" ]] || die "Usage: bash sync_models.sh <VPS_IP> [--user <user>] [--port <port>]"

# ── Detect project root (script location) ────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${SCRIPT_DIR}/cache"
REMOTE_DIR="/home/${VPS_USER}/AlgoTrad/cache"
SSH_OPTS="-p ${SSH_PORT} -o StrictHostKeyChecking=accept-new"

# ── Check local models exist ──────────────────────────────────────────────────
MODEL_FILES=()
for f in \
  "${CACHE_DIR}/lstm_model.keras" \
  "${CACHE_DIR}/xgb_model.pkl" \
  "${CACHE_DIR}/lgb_model.pkl"
do
  if [[ -f "$f" ]]; then
    MODEL_FILES+=("$f")
    info "Found: $(basename "$f")  ($(du -sh "$f" | cut -f1))"
  else
    echo -e "${YLW}[WARN]${NC}  Missing: $(basename "$f") — run 'python main.py --train' first"
  fi
done

[[ ${#MODEL_FILES[@]} -gt 0 ]] || die "No model files found in ${CACHE_DIR}.\nRun: python main.py --train"

# ── Ensure remote cache dir exists ───────────────────────────────────────────
echo "Connecting to ${VPS_USER}@${VPS_IP}:${SSH_PORT}..."
ssh ${SSH_OPTS} "${VPS_USER}@${VPS_IP}" "mkdir -p ${REMOTE_DIR}"

# ── Transfer models ──────────────────────────────────────────────────────────
echo "Transferring ${#MODEL_FILES[@]} model file(s)..."
scp -P "${SSH_PORT}" "${MODEL_FILES[@]}" "${VPS_USER}@${VPS_IP}:${REMOTE_DIR}/"

ok "Models synced to ${VPS_USER}@${VPS_IP}:${REMOTE_DIR}"
echo ""
echo "  Next: restart service on VPS"
echo "  ssh ${VPS_USER}@${VPS_IP} 'sudo systemctl restart algotrad'"
