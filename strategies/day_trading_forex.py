"""
DAY TRADING STRATEGY
Markets  : Gold (GC=F) · Nasdaq-100 ETF (QQQ)
Timeframe: 4H structure → 1H execution
Max freq : 3 trades / day per strategy (independent counters)
Sessions :
  GC=F  — London 07–10 UTC  |  NY 13–16 UTC
  QQQ   — NYSE  14–20 UTC

Logic:
  4H : Directional bias via VWAP position + CVD slope + Value Area
  1H : Entry at pullback to VWAP or VA boundary with
       Orderflow / Volume Delta / CVD micro-divergence confirmation

Risk:
  Capital : 8 871 per strategy
  Risk    : 1% per trade = 88.71
  R:R     : 1:2 strict
  SL      : 1.5 × ATR(14, 1H)
  2 consecutive losses today → stop trading for the day
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
    compute_session_vwap,
    compute_volume_delta,
    compute_cvd,
    compute_orderflow_score,
)

CAPITAL        = 8_871.0
RISK_PER_TRADE = CAPITAL * 0.01    # 88.71
RR_RATIO       = 2.0
MAX_TRADES_DAY = 3

# UTC session windows
LONDON_START,    LONDON_END    = 7,  10   # Gold London open
NY_START,        NY_END        = 13, 16   # Gold NY session
NY_NASDAQ_START, NY_NASDAQ_END = 14, 20   # QQQ NYSE hours (~13:30–20:00 UTC EDT)

PAIR_META: dict[str, dict] = {
    "GC=F": {"pip": 0.10, "pip_usd": 10.0,  "name": "Gold (XAU/USD)"},
    "QQQ":  {"pip": 0.01, "pip_usd": 0.01,  "name": "Nasdaq-100 (QQQ)"},
    "SPY":  {"pip": 0.01, "pip_usd": 0.01,  "name": "S&P 500 ETF (SPY)"},
    "NQ=F": {"pip": 1.00, "pip_usd": 20.0,  "name": "Nasdaq Futures (NQ=F)"},
}

_DEFAULT_VP_BARS  = 20
_DEFAULT_MIN_CONF = 3
_DEFAULT_ATR_MULT = 1.5
_DEFAULT_VWAP_TOL = 0.001    # 0.1%
_DEFAULT_POC_TOL  = 0.002    # 0.2% proximity band around POC (wider than VWAP)


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
    poc_1h: float           # POC from 1H session volume profile
    vah_4h: float
    val_4h: float
    vwap_4h: float
    vwap_1h: float
    cvd_bias_4h: str        # "bullish" | "bearish"
    orderflow_score: float  # -1 to +1
    atr_1h: float
    pip_risk: float
    dxy_bias: str | None = None
    dxy_slope_pct: float | None = None
    breakdown: dict = field(default_factory=dict)


class DayTradingStrategy:
    """
    Multi-timeframe day trading strategy.
    Instantiate once per asset (Gold or Nasdaq) for independent daily counters.
    """

    def __init__(self, cfg: dict, strategy_key: str = "gold"):
        dt = cfg.get("strategies", {}).get(strategy_key, {})
        self.vp_bars_4h    = int(dt.get("vp_bars_4h",    _DEFAULT_VP_BARS))
        self.min_confluence= int(dt.get("min_confluence", _DEFAULT_MIN_CONF))
        self.atr_sl_mult   = float(dt.get("atr_sl_mult",  _DEFAULT_ATR_MULT))
        self.vwap_tol      = float(dt.get("vwap_tol",     _DEFAULT_VWAP_TOL))
        self.poc_tol       = float(dt.get("poc_tol",      _DEFAULT_POC_TOL))
        self.rr_ratio      = float(dt.get("rr_ratio",     RR_RATIO))
        self.require_dxy   = bool(dt.get("require_dxy",   False))
        self._max_trades   = int(dt.get("max_trades_day", MAX_TRADES_DAY))
        self._day_key: str = ""
        self._trades_today = 0
        self._losses_today = 0

        # Sessions loaded from config: {name: [start_h, end_h]} — fully configurable
        raw_sessions = dt.get("sessions_utc", {})
        self._sessions: dict[str, tuple[int, int]] = {
            name: (int(v[0]), int(v[1])) for name, v in raw_sessions.items()
        }

        # Pre-load blocked setups/sessions/directions from config
        self._blocked_setups:      list[str] = list(dt.get("blocked_setups", []))
        self._blocked_sessions:    list[str] = list(dt.get("blocked_sessions", []))
        self._blocked_directions:  list[str] = [d.upper() for d in dt.get("blocked_directions", [])]
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
        if self._trades_today >= self._max_trades:
            return False
        if self._losses_today >= 2:
            return False
        return True

    def record_trade(self, was_loss: bool = False) -> None:
        self._refresh_day()
        self._trades_today += 1
        if was_loss:
            self._losses_today += 1
        else:
            self._losses_today = 0

    # ── Session check (config-driven) ────────────────────────────────────────
    def active_session(self, ticker: str) -> tuple[bool, str]:
        h = datetime.now(timezone.utc).hour
        for name, (start, end) in self._sessions.items():
            if start <= h < end:
                return True, name
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
        df_dxy_1h: 1H DXY data — used only for GC=F inverse-correlation filter
        """
        active, session = self.active_session(ticker)
        if not active:
            logger.debug(f"DayTrading [{ticker}]: session closed")
            return None

        if session in self._blocked_sessions:
            logger.debug(f"DayTrading [{ticker}]: session {session} blocked")
            return None

        if ticker in self._blacklisted_tickers:
            logger.debug(f"DayTrading [{ticker}]: blacklisted")
            return None

        if len(df_4h) < self.vp_bars_4h + 5 or len(df_1h) < 20:
            logger.debug(f"DayTrading [{ticker}]: insufficient bars")
            return None

        meta = PAIR_META.get(ticker, {"pip": 0.01, "pip_usd": 0.01, "name": ticker})

        # ── 4H Structure ─────────────────────────────────────────────────────
        vp_4h             = df_4h.tail(self.vp_bars_4h)
        poc_4h, vah_4h, val_4h = compute_volume_profile(vp_4h)
        vwap_4h           = float(compute_vwap(vp_4h).iloc[-1])
        cvd_4h            = compute_cvd(df_4h)
        n_back            = min(6, len(cvd_4h))
        cvd_4h_slope      = float(cvd_4h.iloc[-1]) - float(cvd_4h.iloc[-n_back])

        price = float(df_1h["close"].iloc[-1])

        cvd_total     = float(cvd_4h.iloc[-1])
        slope_min_pct = 0.01
        slope_meaningful = (
            abs(cvd_total) < 1 or
            abs(cvd_4h_slope) >= abs(cvd_total) * slope_min_pct
        )

        if cvd_4h_slope > 0 and price > vwap_4h and slope_meaningful:
            bias = "bullish"
        elif cvd_4h_slope < 0 and price < vwap_4h and slope_meaningful:
            bias = "bearish"
        else:
            logger.debug(
                f"DayTrading [{ticker}]: 4H bias neutral — "
                f"slope={cvd_4h_slope:+.0f} vs total={cvd_total:+.0f} "
                f"meaningful={slope_meaningful}"
            )
            return None

        # ── 1H Execution ─────────────────────────────────────────────────────
        vwap_1h  = float(compute_session_vwap(df_1h).iloc[-1])
        vd_1h    = compute_volume_delta(df_1h)
        cvd_1h   = compute_cvd(df_1h)
        vd_last  = float(vd_1h.iloc[-1])
        atr_1h   = _atr(df_1h)
        of_score = compute_orderflow_score(df_1h, window=5)

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

        sr      = _swing_sr(df_4h.tail(30))
        near_sr = _is_near(price, sr, tol=atr_1h * 0.5)

        # ── POC proximity (4H + 1H session) ──────────────────────────────────
        # POC = price level with highest traded volume = strongest S/R magnet.
        # at_poc_4h: price within poc_tol of 4H POC (multi-day level)
        # at_poc_1h: price within poc_tol of current session POC (intraday level)
        #
        # Session VP anchored to today's session open (UTC):
        #   Gold London: 07 UTC  |  Gold NY: 13 UTC  |  Nasdaq: 14 UTC
        _session_open_h = {"london": 7, "ny": 13}.get(session, 14)
        if isinstance(df_1h.index, pd.DatetimeIndex):
            import datetime as _dt
            _today_utc = _dt.datetime.now(_dt.timezone.utc).date()
            _session_start = pd.Timestamp(
                _today_utc.year, _today_utc.month, _today_utc.day,
                _session_open_h, tzinfo=_dt.timezone.utc
            )
            _tz = df_1h.index.tz
            if _tz is not None:
                _session_start = _session_start.astimezone(_tz)
            else:
                _session_start = _session_start.replace(tzinfo=None)
            _sess_bars = df_1h[df_1h.index >= _session_start]
            _sess_df   = _sess_bars if len(_sess_bars) >= 3 else df_1h.tail(12)
        else:
            _sess_df = df_1h.tail(12)
        poc_1h, _, _ = compute_volume_profile(_sess_df)
        at_poc_4h = abs(price - poc_4h) / max(poc_4h, 1e-9) < self.poc_tol
        at_poc_1h = abs(price - poc_1h) / max(poc_1h, 1e-9) < self.poc_tol
        at_poc    = at_poc_4h or at_poc_1h
        logger.debug(
            f"DayTrading [{ticker}]: POC_4H={poc_4h:.4f} POC_1H={poc_1h:.4f} "
            f"at_poc_4h={at_poc_4h} at_poc_1h={at_poc_1h} price={price:.4f}"
        )

        # ── DXY filter (Gold only) ────────────────────────────────────────────
        # Gold and DXY inversely correlated:
        #   BUY  Gold → need DXY bearish (declining dollar)
        #   SELL Gold → need DXY bullish (rising dollar)
        dxy_bias: str | None        = None
        dxy_slope_pct: float | None = None
        dxy_confirms_long           = False
        dxy_confirms_short          = False

        if ticker == "GC=F" and df_dxy_1h is not None and len(df_dxy_1h) >= 8:
            n_dxy          = min(6, len(df_dxy_1h))
            dxy_close      = df_dxy_1h["close"]
            dxy_now        = float(dxy_close.iloc[-1])
            dxy_prev       = float(dxy_close.iloc[-n_dxy])
            dxy_slope_pct  = (dxy_now - dxy_prev) / max(dxy_prev, 1e-9) * 100
            dxy_vwap       = float(compute_vwap(df_dxy_1h).iloc[-1])
            dxy_above_vwap = dxy_now > dxy_vwap

            if dxy_slope_pct < 0 and not dxy_above_vwap:
                dxy_bias = "bearish"
            elif dxy_slope_pct > 0 and dxy_above_vwap:
                dxy_bias = "bullish"
            else:
                dxy_bias = "neutral"

            # require_dxy=True: DXY must confirm direction (neutral or wrong = block).
            # require_dxy=False: soft filter only (strong contra still blocks).
            if self.require_dxy:
                if bias == "bullish" and dxy_bias != "bearish":
                    logger.debug(f"DayTrading [GC=F]: DXY={dxy_bias} not bearish — BUY blocked (require_dxy)")
                    return None
                if bias == "bearish" and dxy_bias != "bullish":
                    logger.debug(f"DayTrading [GC=F]: DXY={dxy_bias} not bullish — SELL blocked (require_dxy)")
                    return None
            else:
                STRONG_SLOPE = 0.15
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
        if bias == "bullish" and "BUY" in self._blocked_directions:
            logger.debug(f"DayTrading [{ticker}]: BUY direction blocked")
            bias = None   # skip LONG

        if bias == "bullish":
            at_vwap = abs(price - vwap_1h) / max(vwap_1h, 1e-9) < self.vwap_tol
            at_val  = price <= val_4h * 1.002

            conf = (
                int(at_vwap or at_val)   +
                int(vd_last > 0)         +
                int(bull_micro)          +
                int(of_score > 0.10)     +
                int(near_sr)             +
                int(at_poc)              +   # +1 if price at 4H or 1H POC
                int(dxy_confirms_long)
            )

            if at_vwap:
                setup_name = "vwap_pullback"
            elif at_val:
                setup_name = "val_rejection"
            elif at_poc:
                setup_name = "poc_rejection"
            else:
                # No positional anchor (not at VWAP/VAL/POC) → skip
                logger.debug(f"DayTrading [{ticker}]: no positional setup (continuation) — skip")
                return None

            if setup_name in self._blocked_setups:
                logger.debug(f"DayTrading [{ticker}]: setup {setup_name} blocked")
                return None

            if conf >= self.min_confluence:
                sl = price - atr_1h * self.atr_sl_mult
                tp = price + (price - sl) * self.rr_ratio
                return _make_signal(
                    ticker, "BUY", price, sl, tp, conf, setup_name,
                    session, poc_4h, poc_1h, vah_4h, val_4h,
                    vwap_4h, vwap_1h, bias, of_score, atr_1h, meta,
                    dxy_bias=dxy_bias, dxy_slope_pct=dxy_slope_pct,
                    extra={"at_vwap": at_vwap, "at_val": at_val,
                           "at_poc_4h": at_poc_4h, "at_poc_1h": at_poc_1h,
                           "vd_last": vd_last, "bull_micro_div": bull_micro,
                           "near_sr": near_sr, "cvd_4h_slope": cvd_4h_slope,
                           "dxy_confirms": dxy_confirms_long},
                )

        # ── SHORT setup ───────────────────────────────────────────────────────
        if bias == "bearish" and "SELL" in self._blocked_directions:
            logger.debug(f"DayTrading [{ticker}]: SELL direction blocked")
            return None

        if bias == "bearish":
            at_vwap = abs(price - vwap_1h) / max(vwap_1h, 1e-9) < self.vwap_tol
            at_vah  = price >= vah_4h * 0.998

            conf = (
                int(at_vwap or at_vah)   +
                int(vd_last < 0)         +
                int(bear_micro)          +
                int(of_score < -0.10)    +
                int(near_sr)             +
                int(at_poc)              +   # +1 if price at 4H or 1H POC
                int(dxy_confirms_short)
            )

            if at_vwap:
                setup_name = "vwap_pullback"
            elif at_vah:
                setup_name = "vah_rejection"
            elif at_poc:
                setup_name = "poc_rejection"
            else:
                logger.debug(f"DayTrading [{ticker}]: no positional setup (continuation) — skip")
                return None

            if setup_name in self._blocked_setups:
                logger.debug(f"DayTrading [{ticker}]: setup {setup_name} blocked")
                return None

            if conf >= self.min_confluence:
                sl = price + atr_1h * self.atr_sl_mult
                tp = price - (sl - price) * self.rr_ratio
                return _make_signal(
                    ticker, "SELL", price, sl, tp, conf, setup_name,
                    session, poc_4h, poc_1h, vah_4h, val_4h,
                    vwap_4h, vwap_1h, bias, of_score, atr_1h, meta,
                    dxy_bias=dxy_bias, dxy_slope_pct=dxy_slope_pct,
                    extra={"at_vwap": at_vwap, "at_vah": at_vah,
                           "at_poc_4h": at_poc_4h, "at_poc_1h": at_poc_1h,
                           "vd_last": vd_last, "bear_micro_div": bear_micro,
                           "near_sr": near_sr, "cvd_4h_slope": cvd_4h_slope,
                           "dxy_confirms": dxy_confirms_short},
                )

        return None


# Backward compatibility alias
DayTradingForexStrategy = DayTradingStrategy


# ── Module helpers ─────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not np.isnan(val) else float(tr.mean())


def _swing_sr(df: pd.DataFrame) -> list[float]:
    highs  = df["high"].values
    lows   = df["low"].values
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
    session, poc_4h, poc_1h, vah_4h, val_4h, vwap_4h, vwap_1h,
    bias, of_score, atr_1h, meta, extra: dict,
    dxy_bias: str | None = None,
    dxy_slope_pct: float | None = None,
) -> DayTradingSignal:
    pip_risk = max((abs(entry - sl)) / meta["pip"], 1.0)
    lot_size = RISK_PER_TRADE / (pip_risk * meta["pip_usd"])
    return DayTradingSignal(
        ticker           = ticker,
        direction        = direction,
        entry            = round(entry, 5),
        stop_loss        = round(sl, 5),
        take_profit      = round(tp, 5),
        confidence       = float(np.clip(0.55 + conf * 0.07, 0.55, 0.92)),
        risk_amount      = RISK_PER_TRADE,
        lot_size         = round(max(lot_size, 0.01), 2),
        setup_type       = setup_type,
        confluence_score = conf,
        session          = session,
        poc_4h           = round(poc_4h, 5),
        poc_1h           = round(poc_1h, 5),
        vah_4h           = round(vah_4h, 5),
        val_4h           = round(val_4h, 5),
        vwap_4h          = round(vwap_4h, 5),
        vwap_1h          = round(vwap_1h, 5),
        cvd_bias_4h      = bias,
        orderflow_score  = round(of_score, 3),
        atr_1h           = round(atr_1h, 5),
        pip_risk         = round(pip_risk, 1),
        dxy_bias         = dxy_bias,
        dxy_slope_pct    = round(dxy_slope_pct, 4) if dxy_slope_pct is not None else None,
        breakdown        = {**extra, "pair": meta["name"],
                            "poc_4h": poc_4h, "poc_1h": poc_1h, "atr_1h": atr_1h,
                            "dxy_bias": dxy_bias, "dxy_slope_pct": dxy_slope_pct},
    )
