"""
STRATÉGIE 1 — SWING TRADING
Markets  : Small / Mid Cap equities
Timeframe: Daily (1D)
Max freq : 5 trades / week

Logic (Auction Market Theory):
  A. VAL Rejection  (LONG)  — price ≤ VAL, CVD divergent +, VD positive
  B. VAH Rejection  (SHORT) — price ≥ VAH, CVD divergent -, VD negative
  C. POC Breakout   (LONG)  — close > POC+0.5%, vol ≥ 1.5×avg, CVD rising
  D. POC Breakdown  (SHORT) — close < POC−0.5%, vol ≥ 1.5×avg, CVD falling

Risk model (per trade):
  Capital  : 8 871
  Risk     : 1% = 88.71
  R:R      : 1:2 strict  → TP = Entry ± 2 × (Entry − SL)
  SL Long  : VAL − 1 ATR (setup A) / POC − 0.5 ATR (setup C)
  SL Short : VAH + 1 ATR (setup B) / POC + 0.5 ATR (setup D)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from utils.logger import logger
from strategies.volume_profile import (
    compute_volume_profile,
    compute_vwap,
    compute_volume_delta,
    compute_cvd,
)

_DEFAULT_CAPITAL  = 8_871.0           # overridden by settings.yaml strategies.swing.capital
CAPITAL           = _DEFAULT_CAPITAL  # kept for backward compat — use self.risk_per_trade
RISK_PER_TRADE    = CAPITAL * 0.01    # 88.71 — overridden in __init__ from cfg
RR_RATIO          = 2.0
MAX_TRADES_WEEK   = 5

_DEFAULT_VP_LOOKBACK      = 20
_DEFAULT_MIN_BALANCE_DAYS = 10
_DEFAULT_BREAKOUT_VOL_MULT= 1.5
_DEFAULT_BREAKOUT_PCT     = 0.005
_DEFAULT_MIN_CONFLUENCE   = 2


@dataclass
class SwingSignal:
    ticker: str
    direction: str               # "BUY" | "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float            # 0–1
    risk_amount: float           # always RISK_PER_TRADE
    position_size_shares: float  # RISK / (entry − SL)
    setup_type: str              # "val_rejection" | "vah_rejection" | "poc_breakout" | "poc_breakdown"
    confluence_score: int        # 0–4
    poc: float
    vah: float
    val: float
    vwap: float
    cvd_divergent: bool
    volume_delta_confirming: bool
    vol_ratio: float
    atr: float
    is_balancing: bool
    breakdown: dict = field(default_factory=dict)


class SwingTradingStrategy:
    """
    Swing trading engine for small/mid cap equities.
    Call .analyse(ticker, df_daily) once per day per candidate ticker.
    Call .can_trade_this_week() before calling analyse to respect 5-trade limit.
    """

    def __init__(self, cfg: dict):
        s = cfg.get("strategies", {}).get("swing", {})
        self.vp_lookback        = int(s.get("vp_lookback",        _DEFAULT_VP_LOOKBACK))
        self.min_balance_days   = int(s.get("min_balance_days",   _DEFAULT_MIN_BALANCE_DAYS))
        self.breakout_vol_mult  = float(s.get("breakout_vol_mult",_DEFAULT_BREAKOUT_VOL_MULT))
        self.breakout_pct       = float(s.get("breakout_pct",     _DEFAULT_BREAKOUT_PCT))
        self.min_confluence     = int(s.get("min_confluence",     _DEFAULT_MIN_CONFLUENCE))
        # Capital from settings.yaml (not hardcoded) — avoids desync when capital changes
        capital           = float(s.get("capital", _DEFAULT_CAPITAL))
        risk_pct          = float(s.get("risk_pct", 0.01))
        self.risk_per_trade = capital * risk_pct
        # Weekly trade counter
        self._week_key: str    = ""
        self._trades_this_week = 0
        # Adaptive params injected externally (set by main.py via AdaptiveParams.patch_strategy)
        self._blocked_setups:   list[str] = []
        self._blacklisted_tickers: list[str] = []

    # ── Weekly counter ─────────────────────────────────────────────────────────
    def can_trade_this_week(self) -> bool:
        import datetime
        today = datetime.date.today()
        # ISO week string e.g. "2025-W22"
        wk = f"{today.isocalendar().year}-W{today.isocalendar().week:02d}"
        if wk != self._week_key:
            self._week_key        = wk
            self._trades_this_week = 0
        return self._trades_this_week < MAX_TRADES_WEEK

    def record_trade(self) -> None:
        self._trades_this_week += 1

    # ── Main analysis ──────────────────────────────────────────────────────────
    def analyse(self, ticker: str, df: pd.DataFrame) -> SwingSignal | None:
        """
        df: Daily OHLCV, minimum 30 bars.
        Returns SwingSignal or None if no qualifying setup found.
        """
        # Adaptive: skip blacklisted tickers
        if ticker in self._blacklisted_tickers:
            logger.debug(f"Swing [{ticker}]: blacklisted by AdaptiveParams — skip")
            return None

        min_bars = self.vp_lookback + 5
        if len(df) < min_bars:
            logger.debug(f"Swing [{ticker}]: {len(df)} bars < required {min_bars}")
            return None

        # ── Volume Profile ────────────────────────────────────────────────────
        df_vp        = df.tail(self.vp_lookback)
        poc, vah, val = compute_volume_profile(df_vp)

        # ── VWAP (anchored to VP window) ──────────────────────────────────────
        vwap = float(compute_vwap(df_vp).iloc[-1])

        # ── Volume Delta & CVD ────────────────────────────────────────────────
        vd  = compute_volume_delta(df)
        cvd = compute_cvd(df)

        price      = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else price

        # ── ATR(14) ───────────────────────────────────────────────────────────
        atr = _atr(df)
        if atr <= 0:
            logger.debug(f"Swing [{ticker}]: ATR=0, insufficient data")
            return None

        # ── AMT Balancing phase ───────────────────────────────────────────────
        # Balancing = price range contained in < 15% of POC over last N days
        recent       = df.tail(self.min_balance_days)
        range_pct    = (recent["high"].max() - recent["low"].min()) / max(poc, 1e-9)
        is_balancing = bool(range_pct < 0.15)

        # ── Volume ratio ──────────────────────────────────────────────────────
        avg_vol   = float(df["volume"].tail(20).mean())
        last_vol  = float(df["volume"].iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        # ── CVD divergence ────────────────────────────────────────────────────
        cvd_5 = cvd.tail(5)
        prc_5 = df["close"].tail(5)

        bull_div = (
            float(prc_5.iloc[-1]) <= float(prc_5.iloc[0]) and
            float(cvd_5.iloc[-1]) >= float(cvd_5.iloc[0]) * 0.97
        )
        bear_div = (
            float(prc_5.iloc[-1]) >= float(prc_5.iloc[0]) and
            float(cvd_5.iloc[-1]) <= float(cvd_5.iloc[0]) * 1.03
        )

        vd_last = float(vd.iloc[-1])

        # ──────────────────────────────────────────────────────────────────────
        # SETUP A — VAL Rejection (LONG)
        # ──────────────────────────────────────────────────────────────────────
        if "val_rejection" not in self._blocked_setups and price <= val * 1.005:
            if bull_div and vd_last > 0:
                conf = _score_long(poc, price, is_balancing, bull_div, vd_last > 0)
                if conf >= self.min_confluence:
                    sl = _clamp_sl_long(val - atr, price, atr)
                    tp = price + (price - sl) * RR_RATIO
                    ps = self.risk_per_trade / max(price - sl, 0.01)
                    return _make(
                        ticker, "BUY", price, sl, tp, conf, "val_rejection",
                        poc, vah, val, vwap, bull_div, True, vol_ratio, atr,
                        is_balancing, ps,
                        extra={"price_vs_val": round(price - val, 4),
                               "cvd_chg_5d": round(float(cvd_5.iloc[-1] - cvd_5.iloc[0]), 0)},
                        risk_per_trade=self.risk_per_trade,
                    )

        # ──────────────────────────────────────────────────────────────────────
        # SETUP B — VAH Rejection (SHORT)
        # ──────────────────────────────────────────────────────────────────────
        if "vah_rejection" not in self._blocked_setups and price >= vah * 0.995:
            if bear_div and vd_last < 0:
                conf = _score_short(poc, price, is_balancing, bear_div, vd_last < 0)
                if conf >= self.min_confluence:
                    sl = _clamp_sl_short(vah + atr, price, atr)
                    tp = price - (sl - price) * RR_RATIO
                    ps = self.risk_per_trade / max(sl - price, 0.01)
                    return _make(
                        ticker, "SELL", price, sl, tp, conf, "vah_rejection",
                        poc, vah, val, vwap, bear_div, True, vol_ratio, atr,
                        is_balancing, ps,
                        extra={"price_vs_vah": round(price - vah, 4),
                               "cvd_chg_5d": round(float(cvd_5.iloc[-1] - cvd_5.iloc[0]), 0)},
                        risk_per_trade=self.risk_per_trade,
                    )

        # ──────────────────────────────────────────────────────────────────────
        # SETUP C — POC Breakout (LONG)
        # ──────────────────────────────────────────────────────────────────────
        thresh_up = poc * (1.0 + self.breakout_pct)
        if "poc_breakout" not in self._blocked_setups and price > thresh_up and prev_close <= thresh_up:
            if vol_ratio >= self.breakout_vol_mult and float(cvd.iloc[-1]) > float(cvd.iloc[-2]):
                sl = _clamp_sl_long(poc - 0.5 * atr, price, atr)
                tp = price + (price - sl) * RR_RATIO
                ps = self.risk_per_trade / max(price - sl, 0.01)
                return _make(
                    ticker, "BUY", price, sl, tp, 3, "poc_breakout",
                    poc, vah, val, vwap, False, vd_last > 0, vol_ratio, atr,
                    is_balancing, ps,
                    extra={"vol_ratio": round(vol_ratio, 2),
                           "breakout_pct": round((price - poc) / poc, 4)},
                    risk_per_trade=self.risk_per_trade,
                )

        # ──────────────────────────────────────────────────────────────────────
        # SETUP D — POC Breakdown (SHORT)
        # ──────────────────────────────────────────────────────────────────────
        thresh_dn = poc * (1.0 - self.breakout_pct)
        if "poc_breakdown" not in self._blocked_setups and price < thresh_dn and prev_close >= thresh_dn:
            if vol_ratio >= self.breakout_vol_mult and float(cvd.iloc[-1]) < float(cvd.iloc[-2]):
                sl = _clamp_sl_short(poc + 0.5 * atr, price, atr)
                tp = price - (sl - price) * RR_RATIO
                ps = self.risk_per_trade / max(sl - price, 0.01)
                return _make(
                    ticker, "SELL", price, sl, tp, 3, "poc_breakdown",
                    poc, vah, val, vwap, False, vd_last < 0, vol_ratio, atr,
                    is_balancing, ps,
                    extra={"vol_ratio": round(vol_ratio, 2),
                           "breakdown_pct": round((poc - price) / poc, 4)},
                    risk_per_trade=self.risk_per_trade,
                )

        return None


# ── Module-level helpers ───────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not np.isnan(val) else float(tr.mean())


def _score_long(poc, price, balancing, cvd_div, vd_pos) -> int:
    return int(cvd_div) + int(vd_pos) + int(poc > price) + int(balancing)


def _score_short(poc, price, balancing, cvd_div, vd_neg) -> int:
    return int(cvd_div) + int(vd_neg) + int(poc < price) + int(balancing)


def _clamp_sl_long(sl_raw, price, atr) -> float:
    """Ensure SL is strictly below entry; minimum 0.5 ATR distance."""
    return float(min(sl_raw, price - 0.5 * atr))


def _clamp_sl_short(sl_raw, price, atr) -> float:
    return float(max(sl_raw, price + 0.5 * atr))


def _make(
    ticker, direction, entry, sl, tp, conf, setup_type,
    poc, vah, val, vwap, cvd_div, vd_conf, vol_ratio, atr, balancing,
    pos_size, extra: dict, risk_per_trade: float = RISK_PER_TRADE,
) -> SwingSignal:
    # conf ∈ [0,4] → confidence ∈ [0.55, 0.95] with mult=0.10 (was 0.08→max 0.87)
    confidence = float(np.clip(0.55 + conf * 0.10, 0.55, 0.95))
    return SwingSignal(
        ticker                 = ticker,
        direction              = direction,
        entry                  = round(entry, 4),
        stop_loss              = round(sl, 4),
        take_profit            = round(tp, 4),
        confidence             = confidence,
        risk_amount            = risk_per_trade,
        position_size_shares   = round(pos_size, 2),
        setup_type             = setup_type,
        confluence_score       = conf,
        poc                    = round(poc, 4),
        vah                    = round(vah, 4),
        val                    = round(val, 4),
        vwap                   = round(vwap, 4),
        cvd_divergent          = cvd_div,
        volume_delta_confirming= vd_conf,
        vol_ratio              = round(vol_ratio, 2),
        atr                    = round(atr, 4),
        is_balancing           = balancing,
        breakdown              = {**extra, "poc": poc, "vah": vah, "val": val, "atr": atr},
    )
