"""
STRATÉGIE 3 — SCALPING HAUTE FRÉQUENCE
Market   : S&P 500 (ES=F futures or ^GSPC)
Timeframe: 1m (yfinance minimum; swap to 1s feed in production)
Logic    : CVD absorption + POC/VWAP proximity at tick level

Risk model:
  Capital  : 8 871
  Risk     : 1% per trade = 88.71
  R:R      : 1:2 strict
  SL       : 3 ticks (3 × 0.25pt = 0.75pt) → $37.50 per contract (ES)
  TP       : 6 ticks (6 × 0.25pt = 1.50pt) → $75.00 per contract
  Contracts: floor(88.71 / (3 ticks × $12.50/tick)) = 2 contracts

Circuit breakers:
  - 3 consecutive losses   → 10-bar mandatory pause
  - Session DD > 3% ($266) → stop the day
  - Session P&L > 5% ($443)→ stop the day (protect gains)
  - First 5 bars at open   → no trading (auction volatility)
  - Bar range > 3× avg     → skip (flash-crash / news spike filter)

Signal types:
  bull_absorption  — volume spike + neg delta + price holds/rises at key level
  bear_absorption  — volume spike + pos delta + price holds/falls at key level
  cvd_divergence   — price/CVD divergence + volume confirmation at key level

NOTE: Production requires tick-level data feed (IB TWS, Rithmic, CQG).
      Replace MarketDataConnector.get_ohlcv(..., "1m") with 1s feed.
      Strategy logic is feed-agnostic.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from utils.logger import logger
from strategies.volume_profile import (
    compute_volume_profile,
    compute_vwap,
    compute_volume_delta,
    compute_cvd,
    detect_absorption,
)

CAPITAL          = 8_871.0
RISK_PER_TRADE   = CAPITAL * 0.01      # 88.71

# ES futures constants (used when ticker = ES=F)
ES_TICK_SIZE     = 0.25                # 1 tick = 0.25 points
ES_TICK_USD      = 12.50               # $12.50 per tick per contract

# SPY / equity constants (used when ticker = SPY or ^GSPC proxy)
SPY_TICK_SIZE    = 0.01                # 1 tick = $0.01
SPY_SL_POINTS    = 0.75               # SL distance in $ (≈0.1% of $756)
SPY_TP_POINTS    = 1.50               # TP = 2 × SL

RR_RATIO         = 2.0
SL_TICKS_DEFAULT = 3
TP_TICKS_DEFAULT = 6                   # SL × R:R = 3 × 2 = 6

_ES_TICKERS  = {"ES=F", "MES=F"}
_SPY_TICKERS = {"SPY", "QQQ", "^GSPC"}

MAX_DD_SESSION      = CAPITAL * 0.03   # $266.13
MAX_PROFIT_SESSION  = CAPITAL * 0.05   # $443.55

# NYSE / CME Equity session (UTC)
MARKET_OPEN_H,  MARKET_OPEN_M  = 13, 30   # 09:30 ET
MARKET_CLOSE_H, MARKET_CLOSE_M = 20,  0   # 16:00 ET

AVOID_OPEN_BARS = 5    # skip first 5 bars (open auction noise)

_DEFAULT_VP_LOOKBACK = 30
_DEFAULT_SPIKE_MULT  = 2.5
_DEFAULT_VWAP_TOL    = 2.0    # ±2 points proximity to VWAP/POC


@dataclass
class ScalpSignal:
    ticker: str
    direction: str          # "BUY" | "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float       # 0–1
    risk_amount: float      # 88.71
    contracts: int
    sl_ticks: int
    tp_ticks: int
    setup_type: str         # "bull_absorption" | "bear_absorption" | "cvd_divergence"
    poc: float
    vah: float
    val: float
    vwap: float
    cvd_at_signal: float
    absorption_bull: bool
    absorption_bear: bool
    volume_spike_ratio: float
    atr: float
    breakdown: dict = field(default_factory=dict)


class ScalpingHFQStrategy:
    """
    High-frequency scalping engine for SPX / ES futures.
    .analyse(ticker, df_1m) must be called every bar (every 1 min in live mode).
    Call .reset_session() at start of each trading day.
    """

    def __init__(self, cfg: dict):
        s = cfg.get("strategies", {}).get("scalping_hfq", {})
        self.vp_lookback   = int(s.get("vp_lookback",    _DEFAULT_VP_LOOKBACK))
        self.spike_mult    = float(s.get("spike_mult",   _DEFAULT_SPIKE_MULT))
        self.min_vol_ratio = float(s.get("min_vol_ratio", 1.5))
        self.vwap_tol_pts  = float(s.get("vwap_tol_pts", _DEFAULT_VWAP_TOL))
        self.sl_ticks      = int(s.get("sl_ticks",        SL_TICKS_DEFAULT))
        self.tp_ticks      = int(s.get("tp_ticks",        TP_TICKS_DEFAULT))

        # Adaptive params (patched externally)
        self._blocked_setups:      list[str] = []
        self._blacklisted_tickers: list[str] = []

        # Per-session state
        self._consecutive_losses = 0
        self._session_pnl_usd    = 0.0
        self._bars_since_open    = 0
        self._pause_until_bar    = 0

    # ── Session management ────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """Call once at the start of each trading day (09:30 ET)."""
        self._consecutive_losses = 0
        self._session_pnl_usd    = 0.0
        self._bars_since_open    = 0
        self._pause_until_bar    = 0
        logger.info("Scalp HFQ: session reset")

    def record_result(self, pnl_usd: float) -> None:
        """Call when a position is closed (TP or SL hit)."""
        self._session_pnl_usd += pnl_usd
        if pnl_usd < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3:
                self._pause_until_bar = self._bars_since_open + 10
                logger.warning(
                    f"Scalp HFQ: 3 consecutive losses → pause until bar "
                    f"{self._pause_until_bar}"
                )
        else:
            self._consecutive_losses = 0

    # ── Circuit breakers ──────────────────────────────────────────────────────

    def circuit_breaker(self) -> tuple[bool, str]:
        """Returns (block_trading, reason). True = do NOT trade."""
        if self._session_pnl_usd <= -MAX_DD_SESSION:
            return True, f"Max daily DD hit ({self._session_pnl_usd:.2f}$)"
        if self._session_pnl_usd >= MAX_PROFIT_SESSION:
            return True, f"Daily profit target reached ({self._session_pnl_usd:.2f}$)"
        if self._bars_since_open < AVOID_OPEN_BARS:
            return True, f"Open auction period (bar {self._bars_since_open})"
        if self._bars_since_open < self._pause_until_bar:
            remaining = self._pause_until_bar - self._bars_since_open
            return True, f"Post-loss pause ({remaining} bars left)"
        return False, ""

    # ── Main analysis ──────────────────────────────────────────────────────────

    @property
    def bars_since_open(self) -> int:
        """
        Real elapsed 1m bars since market open (09:30 ET = 13:30 UTC).
        Time-based — independent of loop frequency.
        """
        now = datetime.now(timezone.utc)
        open_time = now.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
                                second=0, microsecond=0)
        if now < open_time:
            return 0
        return int((now - open_time).total_seconds() / 60)

    def analyse(self, ticker: str, df: pd.DataFrame) -> ScalpSignal | None:
        """
        df: 1m OHLCV for current session (rolling window, min 15 bars).
        Returns ScalpSignal or None.
        """
        self._bars_since_open = self.bars_since_open   # sync with real time

        if ticker in self._blacklisted_tickers:
            return None

        blocked, reason = self.circuit_breaker()
        if blocked:
            logger.debug(f"Scalp [{ticker}]: CB — {reason}")
            return None

        if len(df) < 15:
            return None

        if not _market_open():
            return None

        # Extreme volatility filter (flash crash / news spike)
        if _regime_extreme(df):
            logger.debug(f"Scalp [{ticker}]: extreme volatility — skip")
            return None

        # ── Volume Profile (session) ──────────────────────────────────────────
        n_vp          = min(self.vp_lookback, len(df))
        poc, vah, val = compute_volume_profile(df.tail(n_vp))
        vwap          = float(compute_vwap(df).iloc[-1])

        # ── CVD ───────────────────────────────────────────────────────────────
        cvd          = compute_cvd(df)
        cvd_now      = float(cvd.iloc[-1])
        n_back       = min(11, len(cvd))
        cvd_slope    = cvd_now - float(cvd.iloc[-n_back])

        # ── Volume ────────────────────────────────────────────────────────────
        avg_vol   = float(df["volume"].tail(20).mean())
        last_vol  = float(df["volume"].iloc[-1])
        vol_ratio = last_vol / max(avg_vol, 1.0)
        vol_spike = vol_ratio >= self.min_vol_ratio

        # ── ATR ───────────────────────────────────────────────────────────────
        atr = _atr(df)

        # ── Absorption ────────────────────────────────────────────────────────
        bull_abs, bear_abs = detect_absorption(df, n_bars=3, spike_mult=self.spike_mult)

        price = float(df["close"].iloc[-1])

        # ── Key level proximity ───────────────────────────────────────────────
        at_poc  = abs(price - poc)  <= self.vwap_tol_pts
        at_vwap = abs(price - vwap) <= self.vwap_tol_pts
        at_val  = abs(price - val)  <= self.vwap_tol_pts * 1.5
        at_vah  = abs(price - vah)  <= self.vwap_tol_pts * 1.5
        at_key  = at_poc or at_vwap or at_val or at_vah

        if not at_key:
            return None     # Only take setups at structurally important levels

        # ── Position sizing — ES futures vs SPY/equity ────────────────────────
        if ticker in _ES_TICKERS:
            sl_dist   = self.sl_ticks * ES_TICK_SIZE
            tp_dist   = self.tp_ticks * ES_TICK_SIZE
            contracts = max(1, int(RISK_PER_TRADE / (self.sl_ticks * ES_TICK_USD)))
        else:
            # SPY / equity: SL = fixed $ distance, size in shares
            sl_dist   = SPY_SL_POINTS
            tp_dist   = SPY_TP_POINTS
            contracts = max(1, int(RISK_PER_TRADE / SPY_SL_POINTS))  # shares

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL 1: Bullish Absorption
        # Volume spike + net selling (delta ≤ 0) + price holds / rises at support
        # ──────────────────────────────────────────────────────────────────────
        if "bull_absorption" not in self._blocked_setups and bull_abs and vol_spike and (at_poc or at_val or at_vwap):
            if cvd_slope <= 0:   # CVD not rising → sellers present but absorbed
                sl = price - sl_dist
                tp = price + tp_dist
                return _make(
                    ticker, "BUY", price, sl, tp, "bull_absorption",
                    poc, vah, val, vwap, cvd_now,
                    bull_abs, bear_abs, vol_ratio, atr, contracts,
                    self.sl_ticks, self.tp_ticks,
                    confidence=0.72,
                    extra={"cvd_slope": cvd_slope, "at_poc": at_poc, "at_vwap": at_vwap},
                )

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL 2: Bearish Absorption
        # Volume spike + net buying (delta ≥ 0) + price holds / falls at resistance
        # ──────────────────────────────────────────────────────────────────────
        if "bear_absorption" not in self._blocked_setups and bear_abs and vol_spike and (at_poc or at_vah or at_vwap):
            if cvd_slope >= 0:   # CVD not falling → buyers present but absorbed
                sl = price + sl_dist
                tp = price - tp_dist
                return _make(
                    ticker, "SELL", price, sl, tp, "bear_absorption",
                    poc, vah, val, vwap, cvd_now,
                    bull_abs, bear_abs, vol_ratio, atr, contracts,
                    self.sl_ticks, self.tp_ticks,
                    confidence=0.72,
                    extra={"cvd_slope": cvd_slope, "at_poc": at_poc, "at_vwap": at_vwap},
                )

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL 3: CVD Divergence (secondary setup)
        # Price/CVD divergence + VD confirmation at key level
        # ──────────────────────────────────────────────────────────────────────
        if vol_spike and at_key:
            prc5    = df["close"].tail(5)
            cvd5    = cvd.tail(5)
            vd      = compute_volume_delta(df)
            vd_last = float(vd.iloc[-1])

            bull_div = (
                float(prc5.iloc[-1]) < float(prc5.iloc[0]) and
                float(cvd5.iloc[-1]) > float(cvd5.iloc[0])
            )
            bear_div = (
                float(prc5.iloc[-1]) > float(prc5.iloc[0]) and
                float(cvd5.iloc[-1]) < float(cvd5.iloc[0])
            )

            if "cvd_divergence" not in self._blocked_setups and bull_div and vd_last > 0 and (at_poc or at_val or at_vwap):
                sl = price - sl_dist
                tp = price + tp_dist
                return _make(
                    ticker, "BUY", price, sl, tp, "cvd_divergence",
                    poc, vah, val, vwap, cvd_now,
                    False, False, vol_ratio, atr, contracts,
                    self.sl_ticks, self.tp_ticks,
                    confidence=0.65,
                    extra={"bull_div": True, "vd_last": vd_last},
                )

            if "cvd_divergence" not in self._blocked_setups and bear_div and vd_last < 0 and (at_poc or at_vah or at_vwap):
                sl = price + sl_dist
                tp = price - tp_dist
                return _make(
                    ticker, "SELL", price, sl, tp, "cvd_divergence",
                    poc, vah, val, vwap, cvd_now,
                    False, False, vol_ratio, atr, contracts,
                    self.sl_ticks, self.tp_ticks,
                    confidence=0.65,
                    extra={"bear_div": True, "vd_last": vd_last},
                )

        return None


# ── Module helpers ─────────────────────────────────────────────────────────────

def _market_open() -> bool:
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    after_open  = (h > MARKET_OPEN_H)  or (h == MARKET_OPEN_H  and m >= MARKET_OPEN_M)
    before_close= (h < MARKET_CLOSE_H) or (h == MARKET_CLOSE_H and m < MARKET_CLOSE_M)
    return after_open and before_close


def _regime_extreme(df: pd.DataFrame, window: int = 20) -> bool:
    """True if current bar's range > 3× average range (news spike / flash crash)."""
    if len(df) < window + 1:
        return False
    avg_range  = float((df["high"] - df["low"]).tail(window).mean())
    last_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
    return last_range > 3.0 * avg_range if avg_range > 0 else False


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not np.isnan(val) else float(tr.mean())


def _make(
    ticker, direction, entry, sl, tp, setup_type,
    poc, vah, val, vwap, cvd_now,
    bull_abs, bear_abs, vol_ratio, atr, contracts,
    sl_ticks, tp_ticks, confidence, extra: dict,
) -> ScalpSignal:
    return ScalpSignal(
        ticker             = ticker,
        direction          = direction,
        entry              = round(entry, 2),
        stop_loss          = round(sl, 2),
        take_profit        = round(tp, 2),
        confidence         = confidence,
        risk_amount        = RISK_PER_TRADE,
        contracts          = contracts,
        sl_ticks           = sl_ticks,
        tp_ticks           = tp_ticks,
        setup_type         = setup_type,
        poc                = round(poc, 2),
        vah                = round(vah, 2),
        val                = round(val, 2),
        vwap               = round(vwap, 2),
        cvd_at_signal      = round(cvd_now, 0),
        absorption_bull    = bull_abs,
        absorption_bear    = bear_abs,
        volume_spike_ratio = round(vol_ratio, 2),
        atr                = round(atr, 2),
        breakdown          = {
            **extra,
            "poc": poc, "vah": vah, "val": val, "vwap": vwap,
            "contracts": contracts,
            "sl_ticks": sl_ticks, "tp_ticks": tp_ticks,
        },
    )
