"""
Telegram alert bot.
Formats and sends trade signals. Rate-limited to avoid spam.
SIGNAL ONLY — no execution logic here.
"""
from __future__ import annotations
import os
import asyncio
import time
from collections import deque
from telegram import Bot
from telegram.constants import ParseMode
from models.signal_fusion import TradeSignal
from utils.logger import logger


class TelegramAlerter:
    def __init__(self, cfg: dict):
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.bot = Bot(token=token) if token else None
        self.max_per_hour = cfg["risk"]["max_signals_per_hour"]
        self._sent_times: deque = deque(maxlen=self.max_per_hour)

    # ── Rate gate ─────────────────────────────────────────────────────────────
    def _rate_ok(self) -> bool:
        now = time.time()
        self._sent_times = deque(
            (t for t in self._sent_times if now - t < 3600),
            maxlen=self.max_per_hour,
        )
        return len(self._sent_times) < self.max_per_hour

    # ── Format message ────────────────────────────────────────────────────────
    @staticmethod
    def _format(signal: TradeSignal, paper: bool = False, entry_price: float | None = None) -> str:
        emoji     = "🟢" if signal.direction == "BUY" else "🔴"
        conf_bar  = "█" * int(signal.confidence * 10) + "░" * (10 - int(signal.confidence * 10))
        header    = "🧪 *[PAPER] SIGNAL SIMULÉ*" if paper else f"{emoji} *SIGNAL LIVE*"
        disp_price = entry_price if entry_price else signal.price

        reasons = []
        bd = signal.breakdown
        if bd.get("ta_dir") == signal.direction:
            reasons.append("TA aligned")
        if bd.get("ml_dir") == signal.direction:
            reasons.append(f"ML {signal.direction}")
        if bd.get("stat_dir") == signal.direction:
            reasons.append(f"Z={bd.get('z_score', 0):+.2f}")
        if bd.get("fund_dir") == signal.direction:
            reasons.append(f"Sentiment={bd.get('sentiment', 0):+.2f}")
        reason_str = " · ".join(reasons) if reasons else "Multi-factor confluence"

        risk_reward = abs(signal.take_profit - signal.price) / max(
            abs(signal.price - signal.stop_loss), 1e-9
        )

        paper_note = "\n⚠️ _Mode PAPER — aucune exécution réelle_" if paper else \
                     "\n⚠️ _Manual execution only. Verify before acting._"

        msg = (
            f"{header} — *{signal.ticker}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *Direction:*  `{signal.direction}`\n"
            f"📊 *Strategy:*   `{signal.strategy} [{signal.timeframe}]`\n"
            f"💰 *Prix entrée:* `{disp_price:.4f}`\n"
            f"🛑 *Stop-Loss:*  `{signal.stop_loss:.4f}`\n"
            f"🎯 *Target:*     `{signal.take_profit:.4f}`\n"
            f"⚖️ *R:R Ratio:* `1 : {risk_reward:.1f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *AI Confidence:* `{signal.confidence:.1%}`\n"
            f"   `[{conf_bar}]`\n"
            f"⚛️ *Quantum P(up):* `{signal.quantum_up_prob:.1%}`\n"
            f"📈 *Est. Sharpe:* `{signal.sharpe_estimate}`\n"
            f"📉 *Est. MaxDD:* `{signal.drawdown_estimate:.1%}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 *Reason:* _{reason_str}_\n"
            f"RSI `{bd.get('rsi', 0):.0f}` · Hurst `{bd.get('hurst', 0):.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
            f"{paper_note}"
        )
        return msg

    # ── Send signal ───────────────────────────────────────────────────────────
    def send_signal(self, signal: TradeSignal, paper: bool = False,
                    entry_price: float | None = None) -> bool:
        if not self.bot:
            logger.warning("Telegram bot not configured — printing to console only")
            print(self._format(signal, paper=paper, entry_price=entry_price))
            return False
        if not self._rate_ok():
            logger.warning("Telegram rate limit reached — signal suppressed")
            return False

        text = self._format(signal, paper=paper, entry_price=entry_price)

        async def _send():
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
            )

        try:
            asyncio.run(_send())
            self._sent_times.append(time.time())
            mode_str = "PAPER" if paper else "LIVE"
            logger.info(f"Telegram [{mode_str}] signal sent: {signal.ticker} {signal.direction}")
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    # ── System status message ─────────────────────────────────────────────────
    def send_status(self, message: str) -> None:
        # Use _send_raw (sync requests) — asyncio.run() fails after tensorflow
        # closes the event loop during model training.
        self._send_raw(f"ℹ️ {message}")

    # ── Démarrage ─────────────────────────────────────────────────────────────
    def send_startup(self, mode: str) -> None:
        import datetime
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        paper_note = "\n🧪 _Mode PAPER — aucune exécution réelle_" if mode == "PAPER" else ""
        msg = (
            f"🟩 *AlgoTrad démarré*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 *Heure:* `{now}`\n"
            f"⚙️ *Mode:* `{mode}`\n"
            f"📡 *Scanner:* actif\n"
            f"🤖 *Gemini NLP:* actif\n"
            f"🛡️ *Kill switch:* actif"
            f"{paper_note}"
        )
        self._send_raw(msg)

    # ── Arrêt ─────────────────────────────────────────────────────────────────
    def send_shutdown(self, reason: str = "Manuel (Ctrl+C)",
                      signals_today: int = 0, cycles: int = 0) -> None:
        import datetime
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        msg = (
            f"🟥 *AlgoTrad arrêté*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 *Heure:* `{now}`\n"
            f"📋 *Raison:* `{reason}`\n"
            f"📊 *Signaux aujourd'hui:* `{signals_today}`\n"
            f"🔄 *Cycles effectués:* `{cycles}`"
        )
        self._send_raw(msg)

    # ── Erreur / alerte ───────────────────────────────────────────────────────
    def send_error(self, message: str, critical: bool = False) -> None:
        prefix = "🚨 *ALERTE CRITIQUE*" if critical else "⚠️ *Avertissement*"
        msg = (
            f"{prefix}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"`{message[:500]}`"
        )
        self._send_raw(msg)

    # ── Résumé quotidien + heartbeat ─────────────────────────────────────────
    def send_daily_summary(
        self,
        signals_today: int,
        cycles: int,
        pnl_stats: dict,
        mode: str = "PAPER",
    ) -> None:
        import datetime
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        stats = pnl_stats or {}
        total_closed = stats.get("total_closed", 0)
        wins         = stats.get("wins", 0)
        losses       = stats.get("losses", 0)
        win_rate     = stats.get("win_rate", 0.0)
        total_pnl    = stats.get("total_pnl_pct", 0.0)
        open_pos     = stats.get("open_positions", 0)
        pnl_sign     = "+" if total_pnl >= 0 else ""
        pnl_emoji    = "🟢" if total_pnl >= 0 else "🔴"

        msg = (
            f"📅 *Rapport quotidien AlgoTrad* [{mode}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 *Heure:* `{now}`\n"
            f"🔄 *Cycles:* `{cycles}`\n"
            f"📡 *Signaux envoyés:* `{signals_today}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pnl_emoji} *P&L cumulé:* `{pnl_sign}{total_pnl:.2%}`\n"
            f"✅ *Wins:* `{wins}` · ❌ *Losses:* `{losses}`\n"
            f"🎯 *Win rate:* `{win_rate:.1%}` ({total_closed} trades fermés)\n"
            f"📂 *Positions ouvertes:* `{open_pos}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💓 _Bot actif — prochain rapport demain_"
        )
        self._send_raw(msg)

    # ── Envoi brut via requests (pas d'async — fonctionne même en shutdown) ──
    def _send_raw(self, text: str) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not token or not self.chat_id:
            return
        try:
            import requests as _req
            _req.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            logger.error(f"Telegram _send_raw failed: {e}")
