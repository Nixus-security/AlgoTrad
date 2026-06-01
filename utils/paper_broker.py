"""
Paper Trading Execution Engine.
Simulates real broker behaviour for the three strategies.

Features:
  - Slippage simulation per strategy type
  - TP/SL checked against bar HIGH/LOW (not close) → realistic fill detection
  - Max hold-bar timeout (scalping: 30 bars, day trading: 8 bars, swing: unlimited)
  - EOD force-close for intraday strategies (day trading, scalping)
  - USD P&L in R-multiples (risk_amount = 88.71 per trade)
  - Per-strategy equity tracking (running capital)
  - CSV trade log at logs/paper_trades.csv
  - Telegram notification on every close (TP / SL / timeout / EOD)

Slippage:
  swing        : 0.08%  (market order on liquid small-cap)
  day_trading  : 0.03%  (Forex/Gold — tight spread, limit execution)
  scalping_hfq : 0.01%  (~1 tick on ES at current price)
"""
from __future__ import annotations
import csv
import os
import time
import datetime
import uuid
from dataclasses import dataclass, field
from utils.logger import logger

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_PER_TRADE = 88.71     # 1% of 8 871 — fixed per trade for all strategies
INITIAL_CAPITAL = 8_871.0

SLIPPAGE_PCT: dict[str, float] = {
    "swing":        0.0008,   # 0.08%
    "day_trading":  0.0003,   # 0.03%
    "scalping_hfq": 0.0001,   # 0.01% (~1 tick ES)
}

# Max bars before forced timeout exit (0 = unlimited)
MAX_HOLD_BARS: dict[str, int] = {
    "swing":        0,
    "day_trading":  8,     # 8 × 1H bars = 8 hours max intraday hold
    "scalping_hfq": 30,    # 30 × 1m bars = 30 min max scalp hold
}

# EOD force-close UTC hours for intraday strategies
EOD_CLOSE_HOUR_UTC: dict[str, int] = {
    "day_trading":  16,    # 16:00 UTC = NY close
    "scalping_hfq": 20,    # 20:00 UTC = CME equity close
}

PAPER_LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "paper_trades.csv")
COLUMNS = [
    "id", "strategy_type", "ticker", "direction",
    "timestamp_open", "timestamp_close",
    "raw_entry", "entry_price_with_slip",
    "stop_loss", "take_profit",
    "exit_price", "exit_reason",
    "hold_bars", "max_hold_bars",
    "r_multiple", "pnl_usd", "running_equity",
    "slippage_pct",
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PaperPosition:
    id: str
    ticker: str
    strategy_type: str        # "swing" | "day_trading" | "scalping_hfq"
    direction: str            # "BUY" | "SELL"
    raw_entry: float          # signal entry price (before slippage)
    entry_price: float        # actual fill = raw_entry ± slippage
    stop_loss: float
    take_profit: float
    risk_amount: float        # 88.71
    sl_distance: float        # |entry_price − stop_loss|
    slippage_pct: float
    max_hold_bars: int
    opened_at: float = field(default_factory=time.time)
    hold_bars: int = 0
    # Filled on close
    status: str = "open"      # "open" | "closed"
    exit_price: float = 0.0
    exit_reason: str = ""     # "TP" | "SL" | "TIMEOUT" | "EOD"
    r_multiple: float = 0.0   # +2.0 = TP, -1.0 = SL
    pnl_usd: float = 0.0
    closed_at: float = 0.0


@dataclass
class StrategyEquity:
    """Running equity for one strategy, starting at INITIAL_CAPITAL."""
    capital: float = INITIAL_CAPITAL
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    max_equity: float = INITIAL_CAPITAL
    min_equity: float = INITIAL_CAPITAL

    def record(self, pnl_usd: float) -> None:
        self.capital += pnl_usd
        self.total_trades += 1
        if pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.max_equity = max(self.max_equity, self.capital)
        self.min_equity = min(self.min_equity, self.capital)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def max_drawdown_usd(self) -> float:
        return self.min_equity - self.max_equity   # negative value

    @property
    def roi_pct(self) -> float:
        return (self.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL


# ── Broker ────────────────────────────────────────────────────────────────────

class PaperBroker:
    """
    Paper trading execution engine.
    One instance shared across all three strategies in paper mode.
    """

    def __init__(self, cfg: dict):
        self._positions: dict[str, PaperPosition] = {}   # id → position
        self._equity: dict[str, StrategyEquity] = {
            "swing":        StrategyEquity(),
            "day_trading":  StrategyEquity(),
            "scalping_hfq": StrategyEquity(),
        }
        os.makedirs(os.path.dirname(PAPER_LOG), exist_ok=True)
        self._ensure_header()

    # ── Execute ───────────────────────────────────────────────────────────────

    def execute(self, signal, strategy_type: str) -> PaperPosition:
        """
        Simulate order fill with slippage.

        signal must have: .ticker, .direction, .entry or .price,
                          .stop_loss, .take_profit, .risk_amount
        Returns PaperPosition (already registered internally).
        """
        raw_entry = float(getattr(signal, "entry", getattr(signal, "price", 0.0)))
        slip      = SLIPPAGE_PCT.get(strategy_type, 0.0005)
        if signal.direction == "BUY":
            fill = raw_entry * (1.0 + slip)    # pay more when buying
        else:
            fill = raw_entry * (1.0 - slip)    # receive less when selling

        sl_dist = abs(fill - signal.stop_loss)
        if sl_dist == 0:
            sl_dist = raw_entry * 0.005        # fallback: 0.5%

        pos = PaperPosition(
            id             = str(uuid.uuid4())[:8],
            ticker         = signal.ticker,
            strategy_type  = strategy_type,
            direction      = signal.direction,
            raw_entry      = raw_entry,
            entry_price    = round(fill, 6),
            stop_loss      = signal.stop_loss,
            take_profit    = signal.take_profit,
            risk_amount    = float(getattr(signal, "risk_amount", RISK_PER_TRADE)),
            sl_distance    = sl_dist,
            slippage_pct   = slip,
            max_hold_bars  = MAX_HOLD_BARS.get(strategy_type, 0),
        )
        self._positions[pos.id] = pos

        logger.info(
            f"[PAPER EXEC] {strategy_type.upper()} {pos.direction} {pos.ticker} "
            f"@ {pos.entry_price:.5f} (slip={slip:.3%}) "
            f"SL={pos.stop_loss:.5f}  TP={pos.take_profit:.5f}  id={pos.id}"
        )
        return pos

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        ohlcv_map: dict[str, dict],
        strategy_filter: str | None = None,
        telegram=None,
        paper: bool = True,
    ) -> list[PaperPosition]:
        """
        Check all open positions against latest OHLCV.

        ohlcv_map: {ticker: {"high": float, "low": float, "close": float}}
        strategy_filter: if set, only process that strategy type.
        Returns list of newly closed PaperPositions.

        Uses HIGH and LOW of the bar — not just close — for realistic TP/SL detection.
        When both TP and SL are within the bar range, assumes TP fills first if
        direction is favourable (optimistic but common simulation assumption).
        """
        closed: list[PaperPosition] = []
        now_utc_h = datetime.datetime.now(datetime.timezone.utc).hour

        for pos_id, pos in list(self._positions.items()):
            if pos.status != "open":
                continue
            if strategy_filter and pos.strategy_type != strategy_filter:
                continue

            bar = ohlcv_map.get(pos.ticker)
            if bar is None:
                pos.hold_bars += 1
                continue

            bar_high  = float(bar.get("high",  bar.get("close", pos.entry_price)))
            bar_low   = float(bar.get("low",   bar.get("close", pos.entry_price)))
            bar_close = float(bar.get("close", pos.entry_price))

            pos.hold_bars += 1

            # ── EOD check (intraday strategies) ───────────────────────────────
            eod_h = EOD_CLOSE_HOUR_UTC.get(pos.strategy_type)
            if eod_h and now_utc_h >= eod_h:
                self._close(pos, bar_close, "EOD", telegram, paper)
                closed.append(pos)
                continue

            # ── TP/SL detection using bar H/L ─────────────────────────────────
            if pos.direction == "BUY":
                hit_tp = bar_high  >= pos.take_profit
                hit_sl = bar_low   <= pos.stop_loss
            else:
                hit_tp = bar_low   <= pos.take_profit
                hit_sl = bar_high  >= pos.stop_loss

            # Both in same bar: TP first (optimistic fill)
            if hit_tp and hit_sl:
                hit_sl = False

            if hit_tp:
                self._close(pos, pos.take_profit, "TP", telegram, paper)
                closed.append(pos)
                continue

            if hit_sl:
                self._close(pos, pos.stop_loss, "SL", telegram, paper)
                closed.append(pos)
                continue

            # ── Timeout ───────────────────────────────────────────────────────
            if pos.max_hold_bars > 0 and pos.hold_bars >= pos.max_hold_bars:
                self._close(pos, bar_close, "TIMEOUT", telegram, paper)
                closed.append(pos)

        return closed

    # ── Force EOD close ───────────────────────────────────────────────────────

    def eod_close_all(
        self,
        ohlcv_map: dict[str, dict],
        strategy_filter: str | None = None,
        telegram=None,
        paper: bool = True,
    ) -> list[PaperPosition]:
        """Force-close all open positions at current price (EOD cleanup)."""
        closed: list[PaperPosition] = []
        for pos_id, pos in list(self._positions.items()):
            if pos.status != "open":
                continue
            if strategy_filter and pos.strategy_type != strategy_filter:
                continue
            bar = ohlcv_map.get(pos.ticker, {})
            price = float(bar.get("close", pos.entry_price))
            self._close(pos, price, "EOD", telegram, paper)
            closed.append(pos)
        return closed

    # ── Position queries ──────────────────────────────────────────────────────

    def open_positions(self, strategy_type: str | None = None) -> list[PaperPosition]:
        return [
            p for p in self._positions.values()
            if p.status == "open" and
               (strategy_type is None or p.strategy_type == strategy_type)
        ]

    def equity(self, strategy_type: str) -> StrategyEquity:
        return self._equity.get(strategy_type, StrategyEquity())

    def equity_summary(self) -> dict:
        out = {}
        for st, eq in self._equity.items():
            out[st] = {
                "capital": round(eq.capital, 2),
                "roi_pct": round(eq.roi_pct * 100, 2),
                "trades":  eq.total_trades,
                "wins":    eq.wins,
                "losses":  eq.losses,
                "win_rate": round(eq.win_rate * 100, 1),
                "max_dd_usd": round(eq.max_drawdown_usd, 2),
            }
        return out

    # ── Internal close ────────────────────────────────────────────────────────

    def _close(
        self,
        pos: PaperPosition,
        exit_price: float,
        reason: str,
        telegram=None,
        paper: bool = True,
    ) -> None:
        pos.status      = "closed"
        pos.exit_price  = round(exit_price, 6)
        pos.exit_reason = reason
        pos.closed_at   = time.time()

        # P&L in R-multiples
        if pos.direction == "BUY":
            signed_dist = exit_price - pos.entry_price
        else:
            signed_dist = pos.entry_price - exit_price

        r_mult        = signed_dist / pos.sl_distance
        pnl_usd       = r_mult * pos.risk_amount

        pos.r_multiple = round(r_mult, 3)
        pos.pnl_usd    = round(pnl_usd, 2)
        pos.pnl_pct    = round(signed_dist / pos.entry_price, 6)

        # Update equity
        eq = self._equity.get(pos.strategy_type)
        if eq:
            eq.record(pnl_usd)
            running_eq = eq.capital
        else:
            running_eq = INITIAL_CAPITAL

        # Log to CSV
        self._log_row(pos, running_eq)

        # Cleanup
        if pos.id in self._positions:
            del self._positions[pos.id]

        sign = "+" if pnl_usd >= 0 else ""
        logger.info(
            f"[PAPER CLOSE] {pos.strategy_type.upper()} {pos.direction} {pos.ticker} "
            f"@ {exit_price:.5f}  R={pos.r_multiple:+.2f}  "
            f"P&L={sign}{pnl_usd:.2f}$  reason={reason}  bars={pos.hold_bars}"
        )

        # Telegram
        if telegram:
            self._send_close_alert(pos, running_eq, telegram, paper)

    # ── Telegram close notification ───────────────────────────────────────────

    @staticmethod
    def _send_close_alert(
        pos: PaperPosition,
        running_equity: float,
        telegram,
        paper: bool,
    ) -> None:
        emoji_dir = "🟢" if pos.direction == "BUY" else "🔴"
        emoji_out = "✅" if pos.pnl_usd > 0 else "❌"
        mode_tag  = "📋 [PAPER]" if paper else "🔴 [LIVE]"
        sign      = "+" if pos.pnl_usd >= 0 else ""

        reason_map = {
            "TP":      "🎯 Take Profit atteint",
            "SL":      "🛑 Stop Loss touché",
            "TIMEOUT": "⏱️ Timeout — sortie forcée",
            "EOD":     "🌙 Clôture fin de session",
        }
        reason_str = reason_map.get(pos.exit_reason, pos.exit_reason)
        duration_m = int((pos.closed_at - pos.opened_at) / 60)

        msg = (
            f"{mode_tag} {emoji_out} *POSITION FERMÉE*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji_dir} *{pos.ticker}*  `{pos.direction}`  `[{pos.strategy_type}]`\n"
            f"📌 *Entrée :*  `{pos.entry_price:.5f}` _(slip={pos.slippage_pct:.3%})_\n"
            f"🚪 *Sortie :*  `{pos.exit_price:.5f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 *Raison :*  _{reason_str}_\n"
            f"📊 *R-multiple :*  `{pos.r_multiple:+.2f}R`\n"
            f"💰 *P&L :*  `{sign}{pos.pnl_usd:.2f}$`\n"
            f"⏳ *Durée :*  `{duration_m} min`  ·  `{pos.hold_bars} bars`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏦 *Capital stratégie :*  `{running_equity:.2f}$`"
        )
        try:
            telegram._send_raw(msg)
        except Exception as e:
            logger.warning(f"PaperBroker: Telegram close alert failed: {e}")

    # ── CSV logging ───────────────────────────────────────────────────────────

    def _ensure_header(self) -> None:
        if not os.path.exists(PAPER_LOG):
            with open(PAPER_LOG, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()

    def _log_row(self, pos: PaperPosition, running_equity: float) -> None:
        row = {
            "id":                       pos.id,
            "strategy_type":            pos.strategy_type,
            "ticker":                   pos.ticker,
            "direction":                pos.direction,
            "timestamp_open":           datetime.datetime.fromtimestamp(pos.opened_at)
                                            .strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_close":          datetime.datetime.fromtimestamp(pos.closed_at)
                                            .strftime("%Y-%m-%d %H:%M:%S"),
            "raw_entry":                pos.raw_entry,
            "entry_price_with_slip":    pos.entry_price,
            "stop_loss":                pos.stop_loss,
            "take_profit":              pos.take_profit,
            "exit_price":               pos.exit_price,
            "exit_reason":              pos.exit_reason,
            "hold_bars":                pos.hold_bars,
            "max_hold_bars":            pos.max_hold_bars,
            "r_multiple":               pos.r_multiple,
            "pnl_usd":                  pos.pnl_usd,
            "running_equity":           round(running_equity, 2),
            "slippage_pct":             pos.slippage_pct,
        }
        try:
            with open(PAPER_LOG, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)
        except Exception as e:
            logger.error(f"PaperBroker: CSV write failed: {e}")
