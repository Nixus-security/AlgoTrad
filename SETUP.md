# AlgoTrad — Setup & Paper Trading Guide

## 1. Installation

```bash
# Python 3.10+ required
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configuration

```bash
# Create .env from scratch — minimum required keys:
cat > .env <<'EOF'
TELEGRAM_BOT_TOKEN="your_token"      # from @BotFather
TELEGRAM_CHAT_ID="your_chat_id"      # from @userinfobot
FRED_API_KEY="your_key"              # free: fred.stlouisfed.org/docs/api/api_key.html
GEMINI_API_KEY="your_key"            # free: aistudio.google.com/app/apikey
TRADING_MODE=paper
EOF
```

Optional (fallback to 0 if absent):
- `NEWS_API_KEY` → newsapi.org (free tier)
- `ALPHA_VANTAGE_KEY` → not used (yfinance is primary source)

Edit `config/settings.yaml` to:
- Add/remove assets from the universe
- Adjust `risk.confidence_threshold` (default 0.72)
- Set `risk.max_drawdown_pct` (default 5%)

## 3. Paper Trading (safe first run)

```bash
python main.py --paper
```

Signals print to console — **no Telegram, no real money**. Verify:
- Signal format is correct
- Confidence values are reasonable (0.5–0.9)
- Stop-loss distances make sense vs. current ATR

## 4. Backtesting

```bash
# Backtest single ticker
python main.py --backtest AAPL

# Results: metrics printed to console + chart saved in logs/
```

Check these thresholds before going live:
| Metric        | Minimum acceptable |
|---------------|--------------------|
| Win Rate      | > 50%              |
| Sharpe        | > 0.8              |
| Max Drawdown  | < 10%              |
| Profit Factor | > 1.3              |

## 5. Train / Retrain ML Model

```bash
python main.py --train
# Fetches historical data for all configured assets
# Trains LSTM with 3-fold time-series CV
# Saves model to cache/lstm_model.keras
```

Model retrains automatically every 24h in live mode.

## 6. Live Signal Mode

```bash
python main.py
```

> **Reminder:** This system only sends signals. All execution is manual.
> Never act on a signal without confirming the current market context.

## 7. VPS Deployment (1-month paper test)

**Requirements:** Ubuntu 22.04 LTS, ≥ 8 GB RAM, ≥ 10 GB disk.
Recommended: Hetzner CX32 (~8€/mois).

```bash
# ── On your LOCAL machine ─────────────────────────────────────────────────

# 1. Train models locally first (avoids 20-30 min on VPS)
python main.py --train

# 2. Transfer .env and models to VPS
scp .env antho@VPS_IP:/home/antho/AlgoTrad/.env
bash sync_models.sh VPS_IP

# ── On the VPS ───────────────────────────────────────────────────────────

# 3. Clone repo
git clone https://github.com/your-user/AlgoTrad.git /home/antho/AlgoTrad

# 4. Deploy (--no-train skips retraining since models were synced)
sudo bash /home/antho/AlgoTrad/deploy.sh --no-train

# ── Optional: Streamlit dashboard ────────────────────────────────────────
sudo bash /home/antho/AlgoTrad/deploy.sh --no-train --dashboard
# Access: ssh -L 8501:localhost:8501 antho@VPS_IP → http://localhost:8501
```

**Useful commands on VPS:**
```bash
sudo systemctl status algotrad          # service status
sudo journalctl -u algotrad -f          # live logs
sudo journalctl -u algotrad --since today
sudo systemctl restart algotrad         # restart
sudo systemctl stop algotrad            # stop (sends Telegram alert)
```

**Monitoring:** Add UptimeRobot (free) to get Telegram alert if VPS goes down.
https://uptimerobot.com

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     AlgoTrad                            │
│                                                         │
│  Market Data ──► Preprocessor ──────────────────────┐  │
│  (yfinance /      (features,                        │  │
│  Alpha Vantage)    sequences)                       │  │
│                                                     ▼  │
│  Macro/News ──────────────────────► FundamentalAnalyzer │
│  (FRED / NewsAPI)                                   │  │
│                                                     │  │
│  OHLCV ──► TechnicalAnalyzer (RSI/BB/MACD/MA)       │  │
│        └──► StatisticalAnalyzer (Z/Hurst/Season)    │  │
│        └──► MLPredictor (LSTM + Attention)          │  │
│                            │                        │  │
│  ┌─────────────────────────▼────────────────────────┘  │
│  │            SignalFusion                              │
│  │  ┌─────────────────────────────────────────┐        │
│  │  │  QuantumOptimizer (QAOA weight opt       │        │
│  │  │  + Quantum Random Walk probability)      │        │
│  │  └─────────────────────────────────────────┘        │
│  └──────────────────────────────────────────           │
│                    │                                    │
│              RiskManager                               │
│       (confidence / Sharpe / DD gate)                  │
│                    │                                    │
│            TelegramAlerter ──► 📱 Your Phone           │
└─────────────────────────────────────────────────────────┘
```

## 8. Signal Message Example

```
🟢 SIGNAL — AAPL
━━━━━━━━━━━━━━━━━━━━━━
📌 Direction:  BUY
📊 Strategy:   day_trading [1h]
💰 Price:      182.4500
🛑 Stop-Loss:  180.1200
🎯 Target:     187.0100
⚖️ R:R Ratio: 1 : 2.0
━━━━━━━━━━━━━━━━━━━━━━
🤖 AI Confidence: 78.3%
   [████████░░]
⚛️ Quantum P(up): 61.2%
📈 Est. Sharpe: 1.42
📉 Est. MaxDD: 2.2%
━━━━━━━━━━━━━━━━━━━━━━
💡 Reason: TA aligned · ML BUY · Z=-2.31
RSI 31 · Hurst 0.43
━━━━━━━━━━━━━━━━━━━━━━
⚠️ Manual execution only. Verify before acting.
```
