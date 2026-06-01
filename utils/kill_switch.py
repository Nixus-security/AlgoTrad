"""
Kill switch — hard-stops the trading loop.
Triggers via: admin file, max daily loss, max daily trades,
              ou 3 pertes consécutives (pause 1h automatique).
Create/delete KILL_SWITCH file in project root to arm/disarm manually.
"""
from __future__ import annotations
import os
import time
from utils.logger import logger

KILL_FILE       = os.path.join(os.path.dirname(__file__), "..", "KILL_SWITCH")
KILL_DAILY_FILE = os.path.join(os.path.dirname(__file__), "..", "KILL_SWITCH_DAILY")

# Pause après N pertes consécutives
_MAX_CONSECUTIVE_LOSSES = 3
_CONSECUTIVE_PAUSE_SEC  = 3600   # 1 heure


class KillSwitch:

    def __init__(self, max_daily_loss_pct: float = 0.02, max_daily_trades: int = 20):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_trades = max_daily_trades
        self._daily_loss: float = 0.0
        self._daily_trades: int = 0
        self._triggered: bool = False

        # Séquences perdantes
        self._consecutive_losses: int = 0
        self._pause_until: float = 0.0

        # Restore daily kill state from disk (survives process restart)
        if os.path.exists(KILL_DAILY_FILE):
            try:
                reason = open(KILL_DAILY_FILE).read().strip()
                self._triggered = True
                logger.warning(f"Kill switch: daily limit already hit (restored from disk): {reason}")
            except Exception:
                self._triggered = True

    # ── Check before each trade ───────────────────────────────────────────────
    def check(self) -> tuple[bool, str]:
        """Returns (trading_allowed, reason). False = halt."""
        if self._triggered:
            return False, "Kill switch previously triggered"

        if os.path.exists(KILL_FILE):
            self._triggered = True
            logger.critical(f"KILL SWITCH: file {KILL_FILE} detected — halting")
            return False, f"Kill file present: {KILL_FILE}"

        if self._daily_loss >= self.max_daily_loss_pct:
            self._triggered = True
            reason = f"Daily loss limit: {self._daily_loss:.2%}"
            logger.critical(
                f"KILL SWITCH: daily loss {self._daily_loss:.2%} "
                f"≥ limit {self.max_daily_loss_pct:.2%}"
            )
            # Persist to disk so restart doesn't reset the daily kill
            try:
                open(KILL_DAILY_FILE, "w").write(reason)
            except Exception:
                pass
            return False, reason

        if self._daily_trades >= self.max_daily_trades:
            return False, f"Daily trade cap reached ({self._daily_trades})"

        # ── Pause après pertes consécutives ──────────────────────────────────
        if time.time() < self._pause_until:
            remaining_min = int((self._pause_until - time.time()) / 60)
            return False, (
                f"{_MAX_CONSECUTIVE_LOSSES} pertes consécutives — "
                f"pause encore {remaining_min} min"
            )

        return True, "OK"

    # ── Record outcome ────────────────────────────────────────────────────────
    def record_trade(self, pnl_pct: float = 0.0):
        if pnl_pct < 0:
            self._daily_loss += abs(pnl_pct)
            self._consecutive_losses += 1
            if self._consecutive_losses >= _MAX_CONSECUTIVE_LOSSES:
                self._pause_until = time.time() + _CONSECUTIVE_PAUSE_SEC
                logger.warning(
                    f"KillSwitch: {_MAX_CONSECUTIVE_LOSSES} pertes consécutives "
                    f"→ pause trading 1h (reprend à "
                    f"{time.strftime('%H:%M', time.localtime(self._pause_until))})"
                )
        else:
            self._consecutive_losses = 0   # reset dès un trade gagnant
        self._daily_trades += 1

    # ── Daily reset (call at start of each trading day) ───────────────────────
    def reset_daily(self):
        self._daily_loss = 0.0
        self._daily_trades = 0
        self._triggered = False
        self._consecutive_losses = 0
        self._pause_until = 0.0
        # Remove daily kill file so next day trading is allowed
        try:
            if os.path.exists(KILL_DAILY_FILE):
                os.remove(KILL_DAILY_FILE)
        except Exception:
            pass
        logger.info("Kill switch daily counters reset")

    # ── Manual arm / disarm ───────────────────────────────────────────────────
    def arm(self):
        with open(KILL_FILE, "w") as fh:
            fh.write("KILL\n")
        self._triggered = True
        logger.critical("Kill switch ARMED — trading halted")

    def disarm(self):
        if os.path.exists(KILL_FILE):
            os.remove(KILL_FILE)
        self._triggered = False
        logger.info("Kill switch disarmed — trading resumed")
