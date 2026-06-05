# AlgoTrad — Interactive Brokers Setup (Path A)

Goal: replace yfinance + PaperBroker simulator with a **real IB paper trading
account**. Same API as live → flip one port to go live. Gives tick-level data
(fixes scalping) and broker-grade fills/margin.

Target deployment: **IB Gateway headless on the Ubuntu VPS** (production).

---

## ⚠️ The 2 downloaded files are NOT what you need

`TWS API Install 1047.01` and `twsapi_macunix.1047.01` are only the **API client
source/samples**. With Python + `ib_async` you don't need either — `ib_async`
bundles its own client. You can delete them.

What you actually need = **IB Gateway** (the app that logs into IB and routes
orders) + an **IB account**.

---

## Step 1 — Open an IB account  ← START TODAY (longest lead time)

1. https://www.ibkr.com → **Open Account** → Individual.
2. Approval: typically 1–3 business days (identity + funding verification).
3. After approval, enable the **paper trading account**:
   Client Portal → Settings → Account Settings → **Paper Trading Account** →
   create. You get a separate paper username/password.
4. Paper account mirrors your live account's market-data subscriptions.

### Market data subscriptions (critical for scalping)

Real-time data is **not free** and is required for the microstructure edge
(CVD / absorption / tick). Subscribe in Client Portal → Settings → Market Data:

| Asset (current AlgoTrad) | Subscription needed | Approx cost |
|--------------------------|---------------------|-------------|
| US stocks (swing tickers)| US Securities Snapshot + Futures Value Bundle / NASDAQ TotalView | ~$1.50–$23/mo |
| ES futures (scalping)    | **CME Real-Time (NP,L1)** | ~$11–15/mo |
| Forex / Gold (day trade) | included (IDEALPRO spot) / COMEX for GC=F | varies |

Without these → you get **delayed data only** (useless for scalping). Budget
this in. Note: this is what makes Path A real vs the yfinance approximation.

---

## Step 2 — VPS prep (while waiting for account)

IB Gateway is a Java GUI app. Headless VPS needs a virtual display + an
auto-login supervisor (Gateway logs out daily and must be restarted).

```bash
# Java + virtual framebuffer + window manager helpers
sudo apt update
sudo apt install -y openjdk-17-jre xvfb x11vnc xterm libxtst6 libxrender1

# Optional: VNC to see the Gateway GUI once for first config
```

---

## Step 3 — Install IB Gateway (standalone, after account approved)

```bash
# Download the standalone Linux installer from IBKR
cd ~
wget https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh
chmod +x ibgateway-stable-standalone-linux-x64.sh
./ibgateway-stable-standalone-linux-x64.sh   # installs to ~/Jts or ~/ibgateway
```

First launch (needs a display — use Xvfb + VNC once) to set:
- **Configure → Settings → API → Settings**
  - ☑ Enable ActiveX and Socket Clients
  - Socket port: **4002** (paper)  ·  4001 = live
  - ☑ Trusted IPs: `127.0.0.1`
  - ☐ Read-Only API  (must be UNCHECKED to place orders)
- Login mode: **Paper Trading** with your paper username.

---

## Step 4 — IBC (headless auto-login + auto-restart)

Gateway forces a daily logout + must be relaunched. IBC automates it.

```bash
cd ~
wget https://github.com/IbcAlpha/IBC/releases/latest/download/IBCLinux-3.20.0.zip
mkdir -p ~/ibc && unzip IBCLinux-*.zip -d ~/ibc
# Edit ~/ibc/config.ini:
#   IbLoginId=<paper_username>
#   IbPassword=<paper_password>
#   TradingMode=paper
#   IbDir=~/Jts
#   OverrideTwsApiPort=4002
```

Run under Xvfb via systemd (see Step 6).

---

## Step 5 — Python lib (in AlgoTrad venv)

```bash
cd /home/antho/AlgoTrad
source .venv/bin/activate
pip install ib_async
echo "ib_async>=1.0.0" >> requirements.txt
```

---

## Step 6 — systemd service (keep Gateway alive headless)

`/etc/systemd/system/ibgateway.service`:

```ini
[Unit]
Description=IB Gateway via IBC (headless)
After=network-online.target

[Service]
User=antho
Environment=DISPLAY=:1
ExecStartPre=/usr/bin/Xvfb :1 -screen 0 1024x768x16 &
ExecStart=/home/antho/ibc/gatewaystart.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ibgateway
```

---

## Step 7 — AlgoTrad config

`config/settings.yaml` gets an `ib:` block (scaffolded separately). Connection
target on the VPS = `127.0.0.1:4002`, same machine as the bot.

---

## Order of operations summary

1. **Open IB account** (today — 1–3 days wait).
2. Subscribe market data (CME for ES = mandatory for scalping).
3. VPS: install Java + Xvfb.
4. Install IB Gateway.
5. Configure API (port 4002, trusted 127.0.0.1, read-only OFF).
6. IBC for headless auto-login.
7. `pip install ib_async`.
8. systemd service.
9. Build + wire the AlgoTrad IB connector (code — done in repo, opt-in via config).

Until the account exists, the connector stays **opt-in** (`ib.enabled: false`)
so the current yfinance + PaperBroker flow keeps running unchanged.
