#!/usr/bin/env bash
# =============================================================================
# deploy.sh — AlgoTrad VPS deployment (Ubuntu 22.04 LTS)
# Usage: bash deploy.sh [--no-train] [--user <username>] [--repo <git-url>]
#
# Steps:
#   1. Checks (OS, RAM, root)
#   2. System packages (python3.11, git, curl, screen)
#   3. Clone / update repo
#   4. Python venv + pip install
#   5. Interactive .env setup
#   6. Systemd service install
#   7. Optional: ML model training (--train)
#   8. Optional: Streamlit dashboard (--dashboard)
#   9. Optional: daily backup cron
#  10. Health summary
# =============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DEPLOY_USER="${SUDO_USER:-${USER:-antho}}"
INSTALL_DIR="/home/${DEPLOY_USER}/AlgoTrad"
REPO_URL="https://github.com/${DEPLOY_USER}/AlgoTrad.git"   # override with --repo
RUN_TRAIN=true
RUN_DASHBOARD=false
PAPER_MODE="--paper"
SERVICE_NAME="algotrad"
MIN_RAM_MB=6000
PYTHON_BIN=""                        # resolved below

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${BLU}[INFO]${NC}  $*"; }
ok()    { echo -e "${GRN}[OK]${NC}    $*"; }
warn()  { echo -e "${YLW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

banner() {
  echo -e "${CYN}"
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║          AlgoTrad — VPS Deploy Script                ║"
  echo "║          Paper Trading  |  1-Month Test              ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-train)    RUN_TRAIN=false ;;
    --dashboard)   RUN_DASHBOARD=true ;;
    --user)        DEPLOY_USER="$2"; INSTALL_DIR="/home/${DEPLOY_USER}/AlgoTrad"; shift ;;
    --repo)        REPO_URL="$2"; shift ;;
    --dir)         INSTALL_DIR="$2"; shift ;;
    *) warn "Unknown arg: $1" ;;
  esac
  shift
done

# =============================================================================
# STEP 0 — Pre-flight checks
# =============================================================================
banner

info "Running pre-flight checks..."

# Must be root (for apt + systemd)
[[ $EUID -eq 0 ]] || die "Run as root: sudo bash deploy.sh"

# OS check
if ! grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
  warn "Not Ubuntu — script written for Ubuntu 22.04 LTS. Proceed carefully."
fi

# RAM check
TOTAL_RAM_MB=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
if [[ $TOTAL_RAM_MB -lt $MIN_RAM_MB ]]; then
  die "Insufficient RAM: ${TOTAL_RAM_MB}MB detected, minimum ${MIN_RAM_MB}MB required.\n  tensorflow + torch + qiskit need at least 6–8 GB."
fi
ok "RAM: ${TOTAL_RAM_MB}MB ✓"

# Disk check (need ~10 GB)
FREE_DISK_GB=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
if [[ $FREE_DISK_GB -lt 10 ]]; then
  die "Low disk space: ${FREE_DISK_GB}GB free, need at least 10GB (python packages are heavy)."
fi
ok "Disk: ${FREE_DISK_GB}GB free ✓"

# User exists
if ! id "${DEPLOY_USER}" &>/dev/null; then
  warn "User '${DEPLOY_USER}' not found. Creating..."
  useradd -m -s /bin/bash "${DEPLOY_USER}"
  ok "User '${DEPLOY_USER}' created"
fi

# =============================================================================
# STEP 1 — System packages
# =============================================================================
info "Installing system packages..."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  python3.11 python3.11-venv python3.11-dev \
  python3-pip \
  git curl wget screen htop \
  build-essential libssl-dev libffi-dev \
  libhdf5-dev pkg-config \
  nginx \
  2>/dev/null

# Resolve python binary
for py in python3.11 python3.10 python3; do
  if command -v "$py" &>/dev/null; then
    PYTHON_BIN=$(command -v "$py")
    PY_VERSION=$("$PYTHON_BIN" --version 2>&1)
    break
  fi
done
[[ -n "$PYTHON_BIN" ]] || die "No python3 found after install"
ok "Python: ${PY_VERSION} → ${PYTHON_BIN}"

# =============================================================================
# STEP 2 — Clone / update repo
# =============================================================================
info "Setting up repository at ${INSTALL_DIR}..."

if [[ -d "${INSTALL_DIR}/.git" ]]; then
  info "Repo exists — pulling latest..."
  sudo -u "${DEPLOY_USER}" git -C "${INSTALL_DIR}" pull --ff-only || \
    warn "git pull failed (local changes?). Skipping pull."
else
  info "Cloning ${REPO_URL}..."
  if ! sudo -u "${DEPLOY_USER}" git clone "${REPO_URL}" "${INSTALL_DIR}" 2>/dev/null; then
    # Fallback: copy from current directory if running from the project
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${SCRIPT_DIR}/main.py" ]]; then
      info "Git clone failed — copying from ${SCRIPT_DIR}..."
      cp -r "${SCRIPT_DIR}" "${INSTALL_DIR}"
      chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "${INSTALL_DIR}"
    else
      die "Cannot clone repo and no local source found.\nSet --repo <git-url> or run deploy.sh from the project directory."
    fi
  fi
fi

chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "${INSTALL_DIR}"
ok "Repository ready at ${INSTALL_DIR}"

# =============================================================================
# STEP 3 — Python venv + dependencies
# =============================================================================
VENV_DIR="${INSTALL_DIR}/.venv"
info "Creating virtual environment..."

if [[ ! -d "${VENV_DIR}" ]]; then
  sudo -u "${DEPLOY_USER}" "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

info "Upgrading pip..."
sudo -u "${DEPLOY_USER}" "${PIP}" install --upgrade pip setuptools wheel -q

info "Installing dependencies (this can take 10–20 min for tensorflow/torch)..."
sudo -u "${DEPLOY_USER}" "${PIP}" install -r "${INSTALL_DIR}/requirements.txt" -q \
  --no-cache-dir 2>&1 | tail -5

ok "Python dependencies installed"

# =============================================================================
# STEP 4 — .env setup (interactive)
# =============================================================================
ENV_FILE="${INSTALL_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  info "No .env found — interactive setup:"
  echo ""

  prompt_secret() {
    local var="$1" label="$2" default="$3"
    read -rp "  ${label}: " val
    val="${val:-$default}"
    echo "${var}=\"${val}\""
  }

  {
    echo "# Generated by deploy.sh — $(date)"
    echo ""
    echo "# === Telegram (required) ==="
    prompt_secret "TELEGRAM_BOT_TOKEN" "Telegram Bot Token (from @BotFather)" ""
    prompt_secret "TELEGRAM_CHAT_ID"   "Telegram Chat ID (from @userinfobot)" ""
    echo ""
    echo "# === Market Data ==="
    echo "# Alpha Vantage: NOT used — yfinance is primary source. Leave blank."
    prompt_secret "ALPHA_VANTAGE_KEY"  "Alpha Vantage key [Enter to skip]" "unused"
    echo ""
    echo "# === Macro Data ==="
    prompt_secret "FRED_API_KEY"       "FRED API key (free: fred.stlouisfed.org)" ""
    echo ""
    echo "# === AI / NLP ==="
    prompt_secret "GEMINI_API_KEY"     "Gemini API key (free: aistudio.google.com)" ""
    echo ""
    echo "# === Optional: News Sentiment ==="
    echo "# NewsAPI: optional — if blank, news score = 0 (no crash, no impact)."
    prompt_secret "NEWS_API_KEY"       "NewsAPI key [Enter to skip]" ""
    echo ""
    echo "# === Mode ==="
    echo "TRADING_MODE=paper"
  } > "${ENV_FILE}"

  chmod 600 "${ENV_FILE}"
  chown "${DEPLOY_USER}:${DEPLOY_USER}" "${ENV_FILE}"
  ok ".env written to ${ENV_FILE}"
else
  ok ".env already exists — skipping (delete to reconfigure)"
fi

# Validate required keys
for key in TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID FRED_API_KEY; do
  val=$(grep "^${key}=" "${ENV_FILE}" | cut -d'"' -f2)
  if [[ -z "$val" || "$val" == "your_"* || "$val" == "optional" ]]; then
    warn "Missing required key: ${key} — edit ${ENV_FILE} before starting."
  fi
done

# =============================================================================
# STEP 5 — Create required directories
# =============================================================================
info "Creating runtime directories..."
for dir in logs cache; do
  mkdir -p "${INSTALL_DIR}/${dir}"
  chown "${DEPLOY_USER}:${DEPLOY_USER}" "${INSTALL_DIR}/${dir}"
done
ok "Directories ready"

# =============================================================================
# STEP 6 — ML model training (optional, recommended)
# =============================================================================
if [[ "${RUN_TRAIN}" == "true" ]]; then
  info "Training ML models (LSTM + XGB + LGB)..."
  info "This takes 15–30 min. Logs: ${INSTALL_DIR}/logs/"
  echo ""
  warn "Press Ctrl+C to skip training and continue with deploy."
  echo "  (Models can be trained later: python main.py --train)"
  echo ""

  TRAIN_LOG="${INSTALL_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"
  if sudo -u "${DEPLOY_USER}" \
      bash -c "cd '${INSTALL_DIR}' && '${PYTHON}' main.py --train" \
      2>&1 | tee "${TRAIN_LOG}"; then
    ok "ML models trained — saved to ${INSTALL_DIR}/cache/"
  else
    warn "Training failed or skipped. Check: ${TRAIN_LOG}"
    warn "Start service anyway — it will attempt retraining on first run."
  fi
fi

# =============================================================================
# STEP 7 — Systemd service
# =============================================================================
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
info "Installing systemd service: ${SERVICE_NAME}"

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=AlgoTrad paper trading bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${DEPLOY_USER}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${PYTHON} main.py ${PAPER_MODE}
Restart=on-failure
RestartSec=30
StartLimitIntervalSec=600
StartLimitBurst=5

# Load secrets
EnvironmentFile=${ENV_FILE}

# Logging → journald
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

# Time to send Telegram shutdown alert
TimeoutStopSec=30

# Watchdog: restart if process hangs for 10 min
WatchdogSec=600

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
ok "Service installed and enabled: ${SERVICE_NAME}"

# =============================================================================
# STEP 8 — Dashboard (Streamlit, optional)
# =============================================================================
if [[ "${RUN_DASHBOARD}" == "true" ]]; then
  info "Configuring Streamlit dashboard on port 8501..."

  DASHBOARD_SERVICE="/etc/systemd/system/algotrad-dashboard.service"
  cat > "${DASHBOARD_SERVICE}" <<EOF
[Unit]
Description=AlgoTrad Streamlit Dashboard
After=network-online.target

[Service]
Type=simple
User=${DEPLOY_USER}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/streamlit run dashboard/app.py --server.port 8501 --server.headless true
Restart=on-failure
RestartSec=10
EnvironmentFile=${ENV_FILE}
StandardOutput=journal
StandardError=journal
SyslogIdentifier=algotrad-dashboard

[Install]
WantedBy=multi-user.target
EOF

  # Nginx reverse proxy for dashboard
  NGINX_CONF="/etc/nginx/sites-available/algotrad"
  SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || echo "your-vps-ip")
  cat > "${NGINX_CONF}" <<EOF
server {
    listen 80;
    server_name ${SERVER_IP};

    location /dashboard/ {
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF
  ln -sf "${NGINX_CONF}" /etc/nginx/sites-enabled/algotrad
  nginx -t && systemctl reload nginx

  systemctl daemon-reload
  systemctl enable algotrad-dashboard
  ok "Dashboard service installed — accessible at http://${SERVER_IP}/dashboard/"
fi

# =============================================================================
# STEP 9 — Daily backup cron (paper_trades.csv + pnl_journal.csv)
# =============================================================================
info "Setting up daily backup cron for trading data..."

BACKUP_DIR="/home/${DEPLOY_USER}/backups/algotrad"
mkdir -p "${BACKUP_DIR}"
chown "${DEPLOY_USER}:${DEPLOY_USER}" "${BACKUP_DIR}"

CRON_FILE="/etc/cron.d/algotrad-backup"
cat > "${CRON_FILE}" <<EOF
# AlgoTrad daily backup — paper trading data
# Runs at 23:55 every day
55 23 * * * ${DEPLOY_USER} cp ${INSTALL_DIR}/logs/paper_trades.csv ${BACKUP_DIR}/paper_trades_\$(date +\%Y\%m\%d).csv 2>/dev/null; cp ${INSTALL_DIR}/logs/pnl_journal.csv ${BACKUP_DIR}/pnl_journal_\$(date +\%Y\%m\%d).csv 2>/dev/null; find ${BACKUP_DIR} -name "*.csv" -mtime +35 -delete
EOF
chmod 644 "${CRON_FILE}"
ok "Backup cron set: ${BACKUP_DIR} (daily at 23:55, 35-day retention)"

# =============================================================================
# STEP 10 — Log rotation
# =============================================================================
LOGROTATE_CONF="/etc/logrotate.d/algotrad"
cat > "${LOGROTATE_CONF}" <<EOF
${INSTALL_DIR}/logs/*.log {
    daily
    rotate 35
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ${DEPLOY_USER} ${DEPLOY_USER}
}
EOF
ok "Log rotation configured (35 days)"

# =============================================================================
# STEP 11 — UFW firewall
# =============================================================================
if command -v ufw &>/dev/null; then
  info "Configuring UFW firewall..."
  ufw --force enable
  ufw allow ssh
  ufw allow 80/tcp    # nginx (dashboard)
  ufw allow 443/tcp   # https if added later
  ok "UFW: SSH + HTTP/HTTPS allowed"
fi

# =============================================================================
# STEP 12 — Start service
# =============================================================================
info "Starting ${SERVICE_NAME}..."
systemctl start "${SERVICE_NAME}"
sleep 3

STATUS=$(systemctl is-active "${SERVICE_NAME}" 2>/dev/null || echo "unknown")
if [[ "${STATUS}" == "active" ]]; then
  ok "Service is RUNNING ✓"
else
  warn "Service status: ${STATUS}"
  warn "Check logs: journalctl -u ${SERVICE_NAME} -n 50"
fi

# =============================================================================
# SUMMARY
# =============================================================================
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || echo "your-vps-ip")

echo ""
echo -e "${CYN}══════════════════════════════════════════════════════${NC}"
echo -e "${GRN}  AlgoTrad deployed successfully!${NC}"
echo -e "${CYN}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Service:     systemctl status ${SERVICE_NAME}"
echo "  Logs live:   journalctl -u ${SERVICE_NAME} -f"
echo "  Logs file:   ${INSTALL_DIR}/logs/"
echo "  Backups:     ${BACKUP_DIR}"
echo "  .env:        ${ENV_FILE}"
echo ""
if [[ "${RUN_DASHBOARD}" == "true" ]]; then
  echo "  Dashboard:   http://${SERVER_IP}/dashboard/"
  echo "  (or SSH tunnel: ssh -L 8501:localhost:8501 ${DEPLOY_USER}@${SERVER_IP})"
else
  echo "  Dashboard:   not enabled (re-run with --dashboard)"
  echo "  SSH tunnel:  ssh -L 8501:localhost:8501 ${DEPLOY_USER}@${SERVER_IP}"
  echo "               then open: http://localhost:8501"
fi
echo ""
echo -e "${YLW}  Next steps:${NC}"
echo "  1. Verify Telegram alerts received"
echo "  2. Add UptimeRobot monitor → https://uptimerobot.com"
echo "     (ping: ${SERVER_IP}  |  alert: Telegram webhook)"
echo "  3. After 24h: check journalctl -u ${SERVICE_NAME} for errors"
echo ""
echo -e "${YLW}  Useful commands:${NC}"
echo "  systemctl restart ${SERVICE_NAME}     # restart bot"
echo "  systemctl stop    ${SERVICE_NAME}     # stop (sends Telegram alert)"
echo "  journalctl -u ${SERVICE_NAME} -f      # live logs"
echo "  journalctl -u ${SERVICE_NAME} --since today  # today only"
echo ""
