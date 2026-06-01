"""
STRATÉGIE 2 — DAY TRADING
Markets  : Forex (EUR/USD, GBP/USD, USD/JPY) + Gold (XAU/USD via GC=F)
Timeframe: 4H structure → 1H execution (multi-timeframe)
Max freq : 3 trades / day (all pairs combined)
Sessions : London 07–10 UTC  |  NY 13–16 UTC

Logic:
  4H layer  : Establish directional bias via VWAP position + CVD slope + VA
  1H layer  : Execution at pullback to VWAP or Value Area boundary with
              Orderflow / Volume Delta / CVD micro-divergence confirmation

Risk model:
  Capital  : 8 871
  Risk     : 1% per trade = 88.71
  R:R      : 1:2 strict
  SL       : 1.5 × ATR(14, 1H) beyond entry
  Lot size : RISK_PER_TRADE / (pip_risk × pip_usd_per_lot)

Rules:
  - 2 consecutive losses today → stop trading remainder of day
  - Never hold through session close (17:30 CET max)
  - Gold (GC=F) NY session only
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
    compute_orderflow_score,
)

CAPITAL         = 8_871.0
RISK_PER_TRADE  = CAPITAL * 0.01    # 88.71
RR_RATIO        = 2.0
MAX_TRADES_DAY  = 3

# UTC session windows
LONDON_START, LONDON_END = 7, 10
NY_START,     NY_END     = 13, 16

# Pair config: pip size + USD value per pip per 1 standard lot (100 000 units)
PAIR_META: dict[str, dict] = {
    "EURUSD=X": {"pip": 0.0001, "pip_usd": 10.0, "name": "EUR/USD"},
    "GBPUSD=X": {"pip": 0.0001, "pip_usd": 10.0, "name": "GBP/USD"},
    "USDJPY=X": {"pip": 0.01,   "pip_usd":  9.1, "name": "USD/JPY"},   # ~9.1$ at 110 USDJPY
    "GC=F":     {"pip": 0.10,   "pip_usd": 10.0, "name": "Gold (XAU)"},
}

_DEFAULT_VP_BARS    = 20
_DEFAULT_MIN_CONF   = 3
_DEFAULT_ATR_MULT   = 1.5
_DEFAULT_VWAP_TOL   = 0.001    # 0.1%


@dataclass
class DayTradingSignal:
    ticker: str
    direction: str          # "BUY" | "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float       # 0–1
    risk_amount: float      # 88.71
    lot_size: float
    setup_type: str         # "vwap_pullback" | "val_rejection" | "vah_rejection"
    confluence_score: int   # 0–5
    session: str            # "london" | "ny"
    poc_4h: float
    vah_4h: float
    val_4h: float
    vwap_4h: float
    vwap_1h: float
    cvd_bias_4h: str        # "bullish" | "bearish"
    orderflow_score: float  # -1 to +1
    atr_1h: float
    pip_risk: float
    dxy_bias: str | None = None    # "bullish"|"bearish"|None (Gold only)
    dxy_slope_pct: float | None = None  # % change DXY last 6×1H bars
    breakdown: dict = field(default_factory=dict)


class DayTradingForexStrategy:
    """
    Multi-timeframe Forex + Gold day trading strategy.
    Requires 4H OHLCV and 1H OHLCV for each pair.
    """

    def __init__(self, cfg: dict):
        dt  = cfg.get("strategies", {}).get("day_trading", {})
        self.vp_bars_4h    = int(dt.get("vp_bars_4h",    _DEFAULT_VP_BARS))
        self.min_confluence= int(dt.get("min_confluence", _DEFAULT_MIN_CONF))
        self.atr_sl_mult   = float(dt.get("atr_sl_mult", _DEFAULT_ATR_MULT))
        self.vwap_tol      = float(dt.get("vwap_tol",    _DEFAULT_VWAP_TOL))
        # Daily trade counter
        self._day_key: str    = ""
        self._trades_today    = 0
        self._losses_today    = 0

        # Adaptive params (patched externally by AdaptiveParams.patch_strategy)
        self._blocked_setups:      list[str] = []
        self._blocked_sessions:    list[str] = []
        self._blacklisted_tickers: list[str] = []

    # ── Daily counters ─────────────────────────────────────────────────────────
    def _refresh_day(self) -> None:
        import datetime
        today = datetime.date.today().isoformat()
        if today != self._day_key:
            self._day_key      = today
            self._trades_today = 0
            self._losses_today = 0

    def can_trade_today(self) -> bool:
        self._refresh_day()
        if self._trades_today >= MAX_TRADES_DAY:
            return False
        if self._losses_today >= 2:       # 2 consecutive losses → stop day
            return False
        return True

    def record_trade(self, was_loss: bool = False) -> None:
        self._refresh_day()
        self._trades_today += 1
        if was_loss:
            self._losses_today += 1
        else:
            self._losses_today = 0        # reset on win

    # ── Session check ──────────────────────────────────────────────────────────
    def active_session(self) -> tuple[bool, str]:
        h = datetime.now(timezone.utc).hour
        if LONDON_START <= h < LONDON_END:
            return True, "london"
        if NY_START <= h < NY_END:
            return True, "ny"
        return False, "closed"

    # ── Main analysis ──────────────────────────────────────────────────────────
    def analyse(
        self,
        ticker: str,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_dxy_1h: pd.DataFrame | None = None,
    ) -> DayTradingSignal | None:
        """
        df_4h    : 4H OHLCV (min 25 bars)
        df_1h    : 1H OHLCV (min 20 bars)
        df_dxy_1h: 1H OHLCV for DXY (DX-Y.NYB) — used only for GC=F as
                   inverse-correlation filter. Optional; skipped if None.
        """
        active, session = self.active_session()
        if not active:
            logger.debug(f"DayTrading [{ticker}]: session closed")
            return None

        if session in self._blocked_sessions:
            logger.debug(f"DayTrading [{ticker}]: session {session} blocked — skip")
            return None

        if ticker in self._blacklisted_tickers:
            logger.debug(f"DayTrading [{ticker}]: blacklisted — skip")
            return None

        # Gold: NY only
        if ticker == "GC=F" and session != "ny":
            return None

        if len(df_4h) < self.vp_bars_4h + 5 or len(df_1h) < 20:
            logger.debug(f"DayTrading [{ticker}]: insufficient bars")
            return None

        meta = PAIR_META.get(ticker, {"pip": 0.0001, "pip_usd": 10.0, "name": ticker})

        # ── 4H Structure ─────────────────────────────────────────────────────
        vp_4h             = df_4h.tail(self.vp_bars_4h)
        poc_4h, vah_4h, val_4h = compute_volume_profile(vp_4h)
        vwap_4h           = float(compute_vwap(vp_4h).iloc[-1])
        cvd_4h            = compute_cvd(df_4h)
        n_back            = min(6, len(cvd_4h))
        cvd_4h_slope      = float(cvd_4h.iloc[-1]) - float(cvd_4h.iloc[-n_back])

        price = float(df_1h["close"].iloc[-1])

        # 4H directional bias: both VWAP position AND CVD slope must align
        if cvd_4h_slope > 0 and price > vwap_4h:
            bias = "bullish"
        elif cvd_4h_slope < 0 and price < vwap_4h:
            bias = "bearish"
        else:
            logger.debug(f"DayTrading [{ticker}]: 4H bias neutral — skip")
            return None

        # ── 1H Execution ─────────────────────────────────────────────────────
        vwap_1h  = float(compute_vwap(df_1h).iloc[-1])
        vd_1h    = compute_volume_delta(df_1h)
        cvd_1h   = compute_cvd(df_1h)
        vd_last  = float(vd_1h.iloc[-1])
        atr_1h   = _atr(df_1h)
        of_score = compute_orderflow_score(df_1h, window=5)

        # CVD micro-divergence (last 3 × 1H bars)
        prc3 = df_1h["close"].tail(3)
        cvd3 = cvd_1h.tail(3)
        bull_micro = (
            float(prc3.iloc[-1]) <= float(prc3.iloc[0]) and
            float(cvd3.iloc[-1]) >= float(cvd3.iloc[0])
        )
        bear_micro = (
            float(prc3.iloc[-1]) >= float(prc3.iloc[0]) and
            float(cvd3.iloc[-1]) <= float(cvd3.iloc[0])
        )

        # S/R levels from swing highs/lows of last 30 × 4H bars
        sr     = _swing_sr(df_4h.tail(30))
        near_sr= _is_near(price, sr, tol=atr_1h * 0.5)

        # ── DXY filter (Gold only) ────────────────────────────────────────────
        # DXY and Gold are inversely correlated:
        #   BUY  Gold → need DXY bearish (declining dollar)
        #   SELL Gold → need DXY bullish (rising dollar)
        # Adds +1 confluence when DXY slope aligns; acts as HARD BLOCK when
        # DXY strongly contradicts the direction (|slope| > 0.15%).
        dxy_bias: str | None      = None
        dxy_slope_pct: float | None = None
        dxy_confirms_long  = False
        dxy_confirms_short = False

        if ticker == "GC=F" and df_dxy_1h is not None and len(df_dxy_1h) >= 8:
            n_dxy         = min(6, len(df_dxy_1h))
            dxy_close     = df_dxy_1h["close"]
            dxy_now       = float(dxy_close.iloc[-1])
            dxy_prev      = float(dxy_close.iloc[-n_dxy])
            dxy_slope_pct = (dxy_now - dxy_prev) / max(dxy_prev, 1e-9) * 100
            dxy_vwap      = float(compute_vwap(df_dxy_1h).iloc[-1])
            dxy_above_vwap= dxy_now > dxy_vwap

            if dxy_slope_pct < 0 and not dxy_above_vwap:
                dxy_bias = "bearish"          # dollar weakening → Gold-bullish
            elif dxy_slope_pct > 0 and dxy_above_vwap:
                dxy_bias = "bullish"          # dollar strengthening → Gold-bearish
            else:
                dxy_bias = "neutral"

            # Hard block: DXY strongly contradicts Gold direction
            STRONG_SLOPE = 0.15               # 0.15% per 6 bars = meaningful move
            if bias == "bullish" and dxy_bias == "bullish" and abs(dxy_slope_pct) > STRONG_SLOPE:
                logger.debug(f"DayTrading [GC=F]: DXY strongly bullish ({dxy_slope_pct:+.3f}%) — BUY blocked")
                return None
            if bias == "bearish" and dxy_bias == "bearish" and abs(dxy_slope_pct) > STRONG_SLOPE:
                logger.debug(f"DayTrading [GC=F]: DXY strongly bearish ({dxy_slope_pct:+.3f}%) — SELL blocked")
                return None

            dxy_confirms_long  = (dxy_bias == "bearish")
            dxy_confirms_short = (dxy_bias == "bullish")
            logger.debug(
                f"DayTrading [GC=F]: DXY slope={dxy_slope_pct:+.3f}% "
                f"vwap_pos={'above' if dxy_above_vwap else 'below'} bias={dxy_bias}"
            )

        # ── LONG setup ────────────────────────────────────────────────────────
        if bias == "bullish":
            at_vwap = abs(price - vwap_1h) / max(vwap_1h, 1e-9) < self.vwap_tol
            at_val  = price <= val_4h * 1.002

            conf = (
                int(at_vwap or at_val)     +
                int(vd_last > 0)           +
                int(bull_micro)            +
                int(of_score > 0.10)       +
                int(near_sr)               +
                int(dxy_confirms_long)     # +1 if DXY bearish (inverse correlation)
            )

            setup_name = "vwap_pullback" if at_vwap else "val_rejection"
            if setup_name in self._blocked_setups:
                logger.debug(f"DayTrading [{ticker}]: setup {setup_name} blocked")
                return None

            if conf >= self.min_confluence:
                sl  = price - atr_1h * self.atr_sl_mult
                tp  = price + (price - sl) * RR_RATIO
                return _make_signal(
                    ticker, "BUY", price, sl, tp, conf,
                    setup_name,
                    session, poc_4h, vah_4h, val_4h,
                    vwap_4h, vwap_1h, bias, of_score, atr_1h, meta,
                    dxy_bias=dxy_bias, dxy_slope_pct=dxy_slope_pct,
                    extra={"at_vwap": at_vwap, "at_val": at_val,
                           "vd_last": vd_last, "bull_micro_div": bull_micro,
                           "near_sr": near_sr, "cvd_4h_slope": cvd_4h_slope,
                           "dxy_confirms": dxy_confirms_long},
                )

        # ── SHORT setup ───────────────────────────────────────────────────────
        if bias == "bearish":
            at_vwap = abs(price - vwap_1h) / max(vwap_1h, 1e-9) < self.vwap_tol
            at_vah  = price >= vah_4h * 0.998

            conf = (
                int(at_vwap or at_vah)     +
                int(vd_last < 0)           +
                int(bear_micro)            +
                int(of_score < -0.10)      +
                int(near_sr)               +
                int(dxy_confirms_short)    # +1 if DXY bullish (inverse correlation)
            )

            setup_name = "vwap_pullback" if at_vwap else "vah_rejection"
            if setup_name in self._blocked_setups:
                logger.debug(f"DayTrading [{ticker}]: setup {setup_name} blocked")
                return None

            if conf >= self.min_confluence:
                sl  = price + atr_1h * self.atr_sl_mult
                tp  = price - (sl - price) * RR_RATIO
                return _make_signal(
                    ticker, "SELL", price, sl, tp, conf,
                    setup_name,
                    session, poc_4h, vah_4h, val_4h,
                    vwap_4h, vwap_1h, bias, of_score, atr_1h, meta,
                    dxy_bias=dxy_bias, dxy_slope_pct=dxy_slope_pct,
                    extra={"at_vwap": at_vwap, "at_vah": at_vah,
                           "vd_last": vd_last, "bear_micro_div": bear_micro,
                           "near_sr": near_sr, "cvd_4h_slope": cvd_4h_slope,
                           "dxy_confirms": dxy_confirms_short},
                )

        return None


# ── Module helpers ─────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not np.isnan(val) else float(tr.mean())


def _swing_sr(df: pd.DataFrame) -> list[float]:
    """Detect swing highs / lows as S/R levels."""
    highs = df["high"].values
    lows  = df["low"].values
    levels: list[float] = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2]:
            levels.append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2]:
            levels.append(float(lows[i]))
    return sorted(levels)


def _is_near(price: float, levels: list[float], tol: float) -> bool:
    return any(abs(price - lvl) <= tol for lvl in levels)


def _make_signal(
    ticker, direction, entry, sl, tp, conf, setup_type,
    session, poc_4h, vah_4h, val_4h, vwap_4h, vwap_1h,
    bias, of_score, atr_1h, meta, extra: dict,
    dxy_bias: str | None = None,
    dxy_slope_pct: float | None = None,
) -> DayTradingSignal:
    pip_risk = max((abs(entry - sl)) / meta["pip"], 1.0)
    lot_size = RISK_PER_TRADE / (pip_risk * meta["pip_usd"])
    return DayTradingSignal(
        ticker          = ticker,
        direction       = direction,
        entry           = round(entry, 5),
        stop_loss       = round(sl, 5),
        take_profit     = round(tp, 5),
        confidence      = float(np.clip(0.55 + conf * 0.07, 0.55, 0.92)),
        risk_amount     = RISK_PER_TRADE,
        lot_size        = round(max(lot_size, 0.01), 2),
        setup_type      = setup_type,
        confluence_score= conf,
        session         = session,
        poc_4h          = round(poc_4h, 5),
        vah_4h          = round(vah_4h, 5),
        val_4h          = round(val_4h, 5),
        vwap_4h         = round(vwap_4h, 5),
        vwap_1h         = round(vwap_1h, 5),
        cvd_bias_4h     = bias,
        orderflow_score = round(of_score, 3),
        atr_1h          = round(atr_1h, 5),
        pip_risk        = round(pip_risk, 1),
        dxy_bias        = dxy_bias,
        dxy_slope_pct   = round(dxy_slope_pct, 4) if dxy_slope_pct is not None else None,
        breakdown       = {**extra, "pair": meta["name"],
                           "poc_4h": poc_4h, "atr_1h": atr_1h,
                           "dxy_bias": dxy_bias, "dxy_slope_pct": dxy_slope_pct},
    )
