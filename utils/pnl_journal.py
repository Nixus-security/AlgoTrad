"""
P&L Journal — tracks paper/live trade outcomes.
Writes to logs/pnl_journal.csv (one row per closed trade).

Flow:
  1. record_signal()   → called when signal dispatched (creates OPEN row)
  2. update_positions() → called each cycle with current prices;
                          closes position if TP or SL hit
  3. get_stats()       → returns running performance metrics
"""
from __future__ import annotations
import csv
import os
import time
import datetime
from dataclasses import dataclass, field
from utils.logger import logger

JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "pnl_journal.csv")

COLUMNS = [
    "timestamp_open", "ticker", "direction", "entry_price",
    "stop_loss", "take_profit", "strategy", "confidence",
    "position_size_pct", "outcome", "exit_price", "pnl_pct",
    "duration_min", "exit_reason",
]


@dataclass
class _OpenPosition:
    ticker: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float
    position_size_pct: float
    opened_at: float = field(default_factory=time.time)
    row_ts: str = ""   # CSV key for update


class PnLJournal:
    def __init__(self):
        os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
        self._positions: dict[str, _OpenPosition] = {}
        self._ensure_header()

    # ── CSV init ──────────────────────────────────────────────────────────────
    def _ensure_header(self) -> None:
        if not os.path.exists(JOURNAL_PATH):
            with open(JOURNAL_PATH, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()

    # ── Record new signal ─────────────────────────────────────────────────────
    def record_signal(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        confidence: float,
        position_size_pct: float,
    ) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row: dict = {
            "timestamp_open":   ts,
            "ticker":           ticker,
            "direction":        direction,
            "entry_price":      round(entry_price, 6),
            "stop_loss":        round(stop_loss, 6),
            "take_profit":      round(take_profit, 6),
            "strategy":         strategy,
            "confidence":       round(confidence, 4),
            "position_size_pct": round(position_size_pct, 4),
            "outcome":          "OPEN",
            "exit_price":       "",
            "pnl_pct":          "",
            "duration_min":     "",
            "exit_reason":      "",
        }
        with open(JOURNAL_PATH, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)

        self._positions[ticker] = _OpenPosition(
            ticker=ticker, direction=direction,
            entry_price=entry_price, stop_loss=stop_loss,
            take_profit=take_profit, strategy=strategy,
            confidence=confidence, position_size_pct=position_size_pct,
            row_ts=ts,
        )
        logger.info(f"PnL Journal: OPEN {direction} {ticker} @ {entry_price:.4f}")

    # ── Check open positions against current prices ───────────────────────────
    def update_positions(self, prices: dict[str, float]) -> list[dict]:
        """
        Returns list of closed trades: [{ticker, direction, pnl_pct, exit_reason, duration_min}]
        Caller should pass pnl_pct to KillSwitch.record_trade().
        """
        closed: list[dict] = []
        for ticker, pos in list(self._positions.items()):
            price = prices.get(ticker)
            if price is None:
                continue

            hit_tp = hit_sl = False
            if pos.direction == "BUY":
                hit_tp = price >= pos.take_profit
                hit_sl = price <= pos.stop_loss
            else:  # SELL
                hit_tp = price <= pos.take_profit
                hit_sl = price >= pos.stop_loss

            if not (hit_tp or hit_sl):
                continue

            exit_reason = "TP" if hit_tp else "SL"
            exit_price  = pos.take_profit if hit_tp else pos.stop_loss
            duration    = int((time.time() - pos.opened_at) / 60)

            if pos.direction == "BUY":
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

            self._close_row(ticker, pos, exit_price, pnl_pct, duration, exit_reason)
            closed.append({
                "ticker":       ticker,
                "direction":    pos.direction,
                "pnl_pct":      pnl_pct,
                "exit_reason":  exit_reason,
                "duration_min": duration,
            })
        return closed

    def _close_row(
        self, ticker: str, pos: _OpenPosition,
        exit_price: float, pnl_pct: float,
        duration: int, exit_reason: str,
    ) -> None:
        outcome = "WIN" if pnl_pct > 0 else "LOSS"
        try:
            rows: list[dict] = []
            updated = False
            with open(JOURNAL_PATH, "r", newline="") as fh:
                for row in csv.DictReader(fh):
                    if (not updated
                            and row["ticker"] == ticker
                            and row["timestamp_open"] == pos.row_ts
                            and row["outcome"] == "OPEN"):
                        row["outcome"]      = outcome
                        row["exit_price"]   = round(exit_price, 6)
                        row["pnl_pct"]      = round(pnl_pct, 6)
                        row["duration_min"] = duration
                        row["exit_reason"]  = exit_reason
                        updated = True
                    rows.append(row)

            with open(JOURNAL_PATH, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            logger.error(f"PnL Journal: close row failed for {ticker}: {e}")

        del self._positions[ticker]
        sign = "+" if pnl_pct >= 0 else ""
        logger.info(
            f"PnL Journal: CLOSED {pos.direction} {ticker} @ {exit_price:.4f} "
            f"→ {sign}{pnl_pct:.2%} ({exit_reason}) {duration}min"
        )

    # ── Running stats ─────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        wins = losses = 0
        total_pnl = 0.0
        try:
            with open(JOURNAL_PATH, "r", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row["outcome"] in ("WIN", "LOSS"):
                        try:
                            total_pnl += float(row["pnl_pct"])
                        except (ValueError, TypeError):
                            pass
                        if row["outcome"] == "WIN":
                            wins += 1
                        else:
                            losses += 1
        except Exception as e:
            logger.error(f"PnL Journal: get_stats error: {e}")
            return {}

        total_closed = wins + losses
        return {
            "total_closed":    total_closed,
            "wins":            wins,
            "losses":          losses,
            "win_rate":        round(wins / total_closed, 4) if total_closed > 0 else 0.0,
            "total_pnl_pct":   round(total_pnl, 6),
            "open_positions":  len(self._positions),
        }

    # ── Force-close all positions at EOD (mark as EXPIRED) ───────────────────
    def eod_close_all(self, prices: dict[str, float]) -> None:
        """Mark remaining open positions as EXPIRED at last known price."""
        for ticker, pos in list(self._positions.items()):
            price = prices.get(ticker, pos.entry_price)
            duration = int((time.time() - pos.opened_at) / 60)
            if pos.direction == "BUY":
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - price) / pos.entry_price
            self._close_row(ticker, pos, price, pnl_pct, duration, "EXPIRED")
        if self._positions:
            logger.info(f"PnL Journal: EOD forced close {len(self._positions)} positions")
