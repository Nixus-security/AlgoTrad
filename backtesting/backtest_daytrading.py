"""
Backtest engine for DayTradingStrategy (VP / VWAP / CVD logic).
Replays analyse() bar-by-bar on historical data — NOT the ML pipeline.

Two modes:
  - PRECISION  : 4H→1H data (~2 years max via yfinance)
  - DAILY PROXY: 1D data (10+ years) — same VP/VWAP/CVD logic on daily bars

After each run, a LOSS ANALYSIS report breaks down P&L by:
  setup type / session / confluence / exit reason / direction / DXY
  + auto-recommendations (block bad setups, raise confluence, block bad sessions)

Usage:
  python main.py --backtest GC=F
  python main.py --backtest QQQ
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from utils.logger import logger
from strategies.day_trading_forex import DayTradingStrategy

SLIPPAGE_PCT  = 0.0008
COMMISSION    = 3.50    # $ per round trip
MAX_HOLD_BARS = 12      # 12 × 1H = 12 hours max hold (was 8 — eod exits were profitable, let positions run)
EOD_HOUR_UTC  = 20
MIN_HISTORY_YEARS = 10  # target; warn if precision data < 2 years

# Daily proxy settings (replaces 4H/1H with 1D bars)
DAILY_VP_BARS    = 20   # ~1 month of daily bars as "4H VP"
DAILY_1H_BARS    = 5    # ~1 week of daily bars as "1H execution"
DAILY_HOLD_BARS  = 5    # max 5 daily bars before timeout


# ── Strategy subclass for backtest ────────────────────────────────────────────
class _BtStrategy(DayTradingStrategy):
    """Overrides active_session() to use historical bar timestamp."""

    def __init__(self, cfg: dict, key: str):
        super().__init__(cfg, key)
        self._bt_hour: int = 14
        self._bt_force_session: str | None = None  # set for daily proxy

    def active_session(self, ticker: str) -> tuple[bool, str]:  # type: ignore[override]
        if self._bt_force_session is not None:
            return True, self._bt_force_session
        h = self._bt_hour
        # Use sessions loaded from config (respects settings.yaml changes)
        for name, (start, end) in self._sessions.items():
            if start <= h < end:
                return True, name
        return False, "closed"


# ── Main backtest class ────────────────────────────────────────────────────────
class BacktestDayTrading:

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(
        self,
        ticker: str,
        strategy_key: str | None = None,
        plot: bool = True,
    ) -> dict:
        if strategy_key is None:
            strategy_key = "gold" if ticker == "GC=F" else "nasdaq"

        logger.info(f"Backtest DayTrading: {ticker} [strategy={strategy_key}]")

        from data.connectors.market_data import MarketDataConnector
        market = MarketDataConnector()

        # ── Precision backtest: 4H + execution TF ────────────────────────────
        # yfinance limits: 4H → 730 days, 1H → 730 days, 15m → 60 days
        st_cfg_exec = self.cfg.get("strategies", {}).get(strategy_key, {})
        exec_tf     = st_cfg_exec.get("timeframes", {}).get("execution", "1h")
        exec_period = "60d" if exec_tf == "15m" else "2y"
        # max_hold_bars: 12h = 12×1H bars or 48×15m bars
        max_hold = 48 if exec_tf == "15m" else MAX_HOLD_BARS

        df_4h = market.get_ohlcv(ticker, "4h", "2y")
        df_1h = market.get_ohlcv(ticker, exec_tf, exec_period)
        logger.info(f"Execution TF: {exec_tf}  period: {exec_period}  max_hold: {max_hold} bars")

        if df_4h is None or df_1h is None:
            raise ValueError(f"No data for {ticker}")
        if len(df_4h) < 30 or len(df_1h) < 30:
            raise ValueError(f"Insufficient data: 4H={len(df_4h)} 1H={len(df_1h)}")

        df_4h = _utc(df_4h)
        df_1h = _utc(df_1h)

        days_precision = (df_1h.index[-1] - df_1h.index[0]).days
        years_precision = days_precision / 365.25
        if years_precision < 0.1:
            raise ValueError(f"Precision data too short: {years_precision:.2f} years")
        if exec_tf == "15m" and years_precision < 0.15:
            logger.warning(f"15m data: only {days_precision} days (yfinance 15m limit = 60 days)")
        elif exec_tf != "15m" and years_precision < 1.5:
            logger.warning(
                f"Precision data only {years_precision:.1f} years "
                f"(yfinance 1H limit ~2 years). "
                f"Running 10-year DAILY PROXY for extended view."
            )

        # DXY for Gold — only if use_dxy: true in settings.yaml
        _use_dxy = self.cfg.get("strategies", {}).get(strategy_key, {}).get("use_dxy", False)
        df_dxy = None
        if ticker == "GC=F" and _use_dxy:
            try:
                raw = market.get_ohlcv("DX-Y.NYB", "1h", "2y")
                df_dxy = _utc(raw) if raw is not None and len(raw) >= 8 else None
            except Exception as e:
                logger.warning(f"DXY fetch failed ({e}) — filter disabled")

        st_cfg  = self.cfg.get("strategies", {}).get(strategy_key, {})
        capital = float(st_cfg.get("capital", 8871.0))
        risk    = capital * float(st_cfg.get("risk_pct", 0.01))

        # ── Run precision backtest ────────────────────────────────────────────
        label_tf = f"{exec_tf} bars"
        logger.info(f"Precision backtest: {len(df_1h)} {label_tf} ({years_precision:.1f} years)")
        trades_precision, equity_curve_precision = _run_precision(
            ticker, df_4h, df_1h, df_dxy, self.cfg, strategy_key, capital, risk,
            max_hold_bars=max_hold,
        )

        metrics = _metrics(trades_precision, equity_curve_precision, capital)
        metrics["ticker"]          = ticker
        metrics["data_years"]      = round(years_precision, 1)
        metrics["mode"]            = "precision_4H1H"

        _log_results(ticker, metrics, mode="PRECISION (4H→1H)")

        # ── Loss analysis on precision trades ─────────────────────────────────
        loss_report = _loss_analysis(trades_precision, capital)
        _log_loss_report(loss_report)
        _save_recommendations(ticker, loss_report)
        metrics["loss_analysis"] = loss_report

        if plot:
            _plot(ticker, equity_curve_precision, trades_precision)

        return metrics


# ── Precision walk (4H→1H) ───────────────────────────────────────────────────

def _run_precision(
    ticker, df_4h, df_1h, df_dxy, cfg, strategy_key, capital, risk,
    max_hold_bars: int = MAX_HOLD_BARS,
) -> tuple[list[dict], list[float]]:
    strategy = _BtStrategy(cfg, strategy_key)
    vp_bars  = strategy.vp_bars_4h

    trades: list[dict] = []
    equity = capital
    equity_curve = [equity]
    open_trade: dict | None = None
    MIN_START = max(30, vp_bars + 5)

    for i in range(MIN_START, len(df_1h) - 1):
        bar = df_1h.iloc[i]
        ts  = df_1h.index[i]
        if ts.weekday() > 4:
            continue

        if open_trade is not None:
            result = _check_exit(open_trade, bar, i, max_hold_bars, EOD_HOUR_UTC)
            if result is not None:
                pnl, reason = result
                equity += pnl
                open_trade.update({"pnl_usd": pnl, "exit_reason": reason, "ts_close": ts})
                trades.append(open_trade)
                open_trade = None
                equity_curve.append(equity)
            continue

        strategy._bt_hour = ts.hour
        active, session = strategy.active_session(ticker)
        if not active:
            continue

        df_4h_sl = df_4h[df_4h.index <= ts].tail(vp_bars + 10)
        df_1h_sl = df_1h.iloc[:i + 1].tail(60)
        if len(df_4h_sl) < vp_bars + 5 or len(df_1h_sl) < 20:
            continue

        dxy_sl = None
        if df_dxy is not None:
            dxy_sl = df_dxy[df_dxy.index <= ts].tail(20)
            if len(dxy_sl) < 8:
                dxy_sl = None

        try:
            sig = strategy.analyse(ticker, df_4h_sl, df_1h_sl, df_dxy_1h=dxy_sl)
        except Exception as e:
            logger.debug(f"analyse() error at {ts}: {e}")
            continue

        if sig is None:
            continue

        slip  = (1 + SLIPPAGE_PCT) if sig.direction == "BUY" else (1 - SLIPPAGE_PCT)
        entry = float(bar["close"]) * slip
        dpp   = risk / max(abs(entry - sig.stop_loss), 1e-9)

        open_trade = {
            "ticker":           ticker,
            "direction":        sig.direction,
            "entry":            entry,
            "stop_loss":        sig.stop_loss,
            "take_profit":      sig.take_profit,
            "setup_type":       sig.setup_type,
            "confluence":       sig.confluence_score,
            "session":          session,
            "ts_open":          ts,
            "bar_open":         i,
            "risk_usd":         risk,
            "dollar_per_point": dpp,
            "dxy_bias":         sig.dxy_bias,
            "at_poc":           sig.breakdown.get("at_poc_4h", False) or sig.breakdown.get("at_poc_1h", False),
            "of_score":         sig.orderflow_score,
            "cvd_bias":         sig.cvd_bias_4h,
        }

    if open_trade is not None:
        last = df_1h.iloc[-1]
        pnl  = _pnl_at_price(open_trade, float(last["close"]))
        open_trade.update({"pnl_usd": pnl, "exit_reason": "end_of_data",
                           "ts_close": df_1h.index[-1]})
        trades.append(open_trade)
        equity += pnl
        equity_curve.append(equity)

    return trades, equity_curve


# ── Daily proxy walk (1D bars, 10+ years) ────────────────────────────────────

def _run_daily_proxy(
    ticker, df_daily, df_dxy_daily, cfg, strategy_key, capital, risk
) -> tuple[list[dict], list[float]]:
    """
    Approximates strategy on daily bars for 10-year coverage.
    4H equivalent = last DAILY_VP_BARS daily bars
    1H equivalent = last DAILY_1H_BARS daily bars
    Session: always "ny" (daily bars have no intraday hours)
    Timeout: DAILY_HOLD_BARS days
    """
    strategy = _BtStrategy(cfg, strategy_key)
    strategy._bt_force_session = "ny"   # bypass session check

    trades: list[dict] = []
    equity = capital
    equity_curve = [equity]
    open_trade: dict | None = None
    MIN_START = DAILY_VP_BARS + 5

    for i in range(MIN_START, len(df_daily) - DAILY_HOLD_BARS):
        bar = df_daily.iloc[i]
        ts  = df_daily.index[i]
        if ts.weekday() > 4:
            continue

        # Exit check on daily bars
        if open_trade is not None:
            result = _check_exit_daily(open_trade, bar, i)
            if result is not None:
                pnl, reason = result
                equity += pnl
                open_trade.update({"pnl_usd": pnl, "exit_reason": reason, "ts_close": ts})
                trades.append(open_trade)
                open_trade = None
                equity_curve.append(equity)
            continue

        # Build slices
        df_4h_sl = df_daily.iloc[:i + 1].tail(DAILY_VP_BARS + 10)  # ~1 month
        df_1h_sl = df_daily.iloc[:i + 1].tail(DAILY_1H_BARS + 15)  # ~3 weeks

        if len(df_4h_sl) < DAILY_VP_BARS + 5 or len(df_1h_sl) < 10:
            continue

        # DXY daily slice
        dxy_sl = None
        if df_dxy_daily is not None:
            dxy_sl = df_dxy_daily[df_dxy_daily.index <= ts].tail(20)
            if len(dxy_sl) < 8:
                dxy_sl = None

        try:
            sig = strategy.analyse(ticker, df_4h_sl, df_1h_sl, df_dxy_1h=dxy_sl)
        except Exception as e:
            logger.debug(f"Daily proxy analyse() error at {ts}: {e}")
            continue

        if sig is None:
            continue

        slip  = (1 + SLIPPAGE_PCT) if sig.direction == "BUY" else (1 - SLIPPAGE_PCT)
        entry = float(bar["close"]) * slip
        dpp   = risk / max(abs(entry - sig.stop_loss), 1e-9)

        open_trade = {
            "ticker":           ticker,
            "direction":        sig.direction,
            "entry":            entry,
            "stop_loss":        sig.stop_loss,
            "take_profit":      sig.take_profit,
            "setup_type":       sig.setup_type,
            "confluence":       sig.confluence_score,
            "session":          "ny",
            "ts_open":          ts,
            "bar_open":         i,
            "risk_usd":         risk,
            "dollar_per_point": dpp,
            "dxy_bias":         sig.dxy_bias,
            "at_poc":           sig.breakdown.get("at_poc_4h", False) or sig.breakdown.get("at_poc_1h", False),
            "of_score":         sig.orderflow_score,
            "cvd_bias":         sig.cvd_bias_4h,
        }

    if open_trade is not None:
        last = df_daily.iloc[-1]
        pnl  = _pnl_at_price(open_trade, float(last["close"]))
        open_trade.update({"pnl_usd": pnl, "exit_reason": "end_of_data",
                           "ts_close": df_daily.index[-1]})
        trades.append(open_trade)
        equity += pnl
        equity_curve.append(equity)

    return trades, equity_curve


# ── Loss Analysis ─────────────────────────────────────────────────────────────

def _loss_analysis(trades: list[dict], capital: float) -> dict:
    """
    Break down P&L by setup / session / confluence / exit reason / direction / DXY.
    Auto-generate recommendations to block underperforming segments.
    """
    if len(trades) < 5:
        return {"note": "Too few trades for meaningful analysis", "recommendations": []}

    df = pd.DataFrame(trades)
    df["won"]     = df["pnl_usd"] > 0
    df["pnl_pct"] = df["pnl_usd"] / capital * 100

    def _breakdown(col: str) -> dict:
        if col not in df.columns:
            return {}
        out = {}
        for val in sorted(df[col].dropna().unique(), key=str):
            sub = df[df[col] == val]
            wins_pnl   = sub[sub["pnl_usd"] > 0]["pnl_usd"].sum()
            losses_pnl = abs(sub[sub["pnl_usd"] <= 0]["pnl_usd"].sum())
            out[str(val)] = {
                "n":             int(len(sub)),
                "win_rate":      round(float(sub["won"].mean()), 3),
                "avg_pnl":       round(float(sub["pnl_usd"].mean()), 2),
                "total_pnl":     round(float(sub["pnl_usd"].sum()), 2),
                "profit_factor": round(wins_pnl / max(losses_pnl, 1e-9), 3),
            }
        return out

    by_setup    = _breakdown("setup_type")
    by_session  = _breakdown("session")
    by_exit     = _breakdown("exit_reason")
    by_direction= _breakdown("direction")
    by_dxy      = _breakdown("dxy_bias") if "dxy_bias" in df.columns else {}

    # Confluence breakdown (numeric → group low ≤3 vs high ≥4)
    by_confluence = _breakdown("confluence")
    if by_confluence:
        low_keys  = [k for k in by_confluence if int(k) <= 3]
        high_keys = [k for k in by_confluence if int(k) >= 4]
        low_sub   = df[df["confluence"].astype(str).isin(low_keys)]
        high_sub  = df[df["confluence"].astype(str).isin(high_keys)]
        by_confluence["_group_low≤3"] = {
            "n":       int(len(low_sub)),
            "win_rate": round(float(low_sub["won"].mean()), 3) if len(low_sub) else 0,
            "avg_pnl":  round(float(low_sub["pnl_usd"].mean()), 2) if len(low_sub) else 0,
        }
        by_confluence["_group_high≥4"] = {
            "n":       int(len(high_sub)),
            "win_rate": round(float(high_sub["won"].mean()), 3) if len(high_sub) else 0,
            "avg_pnl":  round(float(high_sub["pnl_usd"].mean()), 2) if len(high_sub) else 0,
        }

    # POC trades vs non-POC
    by_poc = {}
    if "at_poc" in df.columns:
        by_poc = _breakdown("at_poc")

    # ── Auto-recommendations ──────────────────────────────────────────────────
    recommendations: list[str] = []
    config_adjustments: dict = {}

    MIN_TRADES = 5  # require at least 5 trades before recommending

    # Block setups with WR < 35% or negative profit factor
    for setup, s in by_setup.items():
        if s["n"] >= MIN_TRADES and (s["win_rate"] < 0.35 or s["profit_factor"] < 0.80):
            recommendations.append(
                f"BLOCK setup '{setup}' — WR={s['win_rate']:.0%} PF={s['profit_factor']:.2f} ({s['n']} trades)"
            )
            config_adjustments.setdefault("blocked_setups", []).append(setup)

    # Block sessions with negative avg PnL
    for sess, s in by_session.items():
        if s["n"] >= MIN_TRADES and s["avg_pnl"] < 0:
            recommendations.append(
                f"BLOCK session '{sess}' — avg_pnl={s['avg_pnl']:.2f}$ ({s['n']} trades)"
            )
            config_adjustments.setdefault("blocked_sessions", []).append(sess)

    # Raise min_confluence if high confluence significantly outperforms
    g_low  = by_confluence.get("_group_low≤3", {})
    g_high = by_confluence.get("_group_high≥4", {})
    if g_low.get("n", 0) >= MIN_TRADES and g_high.get("n", 0) >= MIN_TRADES:
        wr_diff = g_high.get("win_rate", 0) - g_low.get("win_rate", 0)
        if wr_diff > 0.10:
            recommendations.append(
                f"RAISE min_confluence to 4 — "
                f"high={g_high['win_rate']:.0%} vs low={g_low['win_rate']:.0%} "
                f"(+{wr_diff:.0%} WR gain)"
            )
            config_adjustments["min_confluence"] = 4

    # Directional bias
    buy_s  = by_direction.get("BUY", {})
    sell_s = by_direction.get("SELL", {})
    if buy_s.get("n", 0) >= MIN_TRADES and sell_s.get("n", 0) >= MIN_TRADES:
        if buy_s["win_rate"] < 0.35 and sell_s["win_rate"] >= 0.45:
            recommendations.append(
                f"CONSIDER SELL-ONLY — BUY WR={buy_s['win_rate']:.0%} vs SELL WR={sell_s['win_rate']:.0%}"
            )
        elif sell_s["win_rate"] < 0.35 and buy_s["win_rate"] >= 0.45:
            recommendations.append(
                f"CONSIDER BUY-ONLY — SELL WR={sell_s['win_rate']:.0%} vs BUY WR={buy_s['win_rate']:.0%}"
            )

    # POC trades outperform?
    poc_true  = by_poc.get("True", {})
    poc_false = by_poc.get("False", {})
    if poc_true.get("n", 0) >= MIN_TRADES and poc_false.get("n", 0) >= MIN_TRADES:
        if poc_true["win_rate"] - poc_false["win_rate"] > 0.10:
            recommendations.append(
                f"POC TRADES OUTPERFORM — at_poc WR={poc_true['win_rate']:.0%} "
                f"vs no_poc WR={poc_false['win_rate']:.0%}. "
                f"Consider requiring POC for entry."
            )

    # DXY filter effectiveness (Gold only)
    dxy_bear = by_dxy.get("bearish", {})  # DXY bearish = Gold bullish confirmation
    dxy_bull = by_dxy.get("bullish", {})
    if dxy_bear.get("n", 0) >= MIN_TRADES:
        recommendations.append(
            f"DXY bearish (Gold-bullish) trades — WR={dxy_bear['win_rate']:.0%} "
            f"avg={dxy_bear['avg_pnl']:.2f}$"
        )

    # Timeout/EOD as % of all trades (too many → hold time too short or too long)
    total = len(df)
    timeouts = df[df["exit_reason"].isin(["timeout", "eod_close"])].shape[0] if "exit_reason" in df.columns else 0
    if total > 0 and timeouts / total > 0.40:
        recommendations.append(
            f"HIGH TIMEOUT RATE {timeouts/total:.0%} — "
            f"many trades not hitting TP/SL within {MAX_HOLD_BARS}h. "
            f"Consider widening TP or tightening SL multiplier."
        )

    return {
        "by_setup":        by_setup,
        "by_session":      by_session,
        "by_exit":         by_exit,
        "by_direction":    by_direction,
        "by_confluence":   by_confluence,
        "by_poc":          by_poc,
        "by_dxy":          by_dxy,
        "recommendations": recommendations,
        "config_adjustments": config_adjustments,
    }


def _log_loss_report(report: dict, label: str = "Precision") -> None:
    if "note" in report:
        logger.info(f"Loss analysis [{label}]: {report['note']}")
        return

    logger.info(f"\n{'═'*55}")
    logger.info(f"  LOSS ANALYSIS — {label}")
    logger.info(f"{'═'*55}")

    def _fmt(d: dict, title: str) -> None:
        if not d:
            return
        logger.info(f"\n  [{title}]")
        for k, v in d.items():
            if k.startswith("_group"):
                logger.info(
                    f"    {k:20s}  n={v.get('n',0):3d}  "
                    f"WR={v.get('win_rate',0):.0%}  "
                    f"avg={v.get('avg_pnl',0):+7.2f}$"
                )
            elif isinstance(v, dict) and "n" in v:
                logger.info(
                    f"    {str(k):20s}  n={v['n']:3d}  "
                    f"WR={v['win_rate']:.0%}  "
                    f"avg={v['avg_pnl']:+7.2f}$  "
                    f"PF={v.get('profit_factor',0):.2f}"
                )

    _fmt(report.get("by_setup", {}),       "By Setup")
    _fmt(report.get("by_session", {}),     "By Session")
    _fmt(report.get("by_direction", {}),   "By Direction")
    _fmt(report.get("by_exit", {}),        "By Exit Reason")
    _fmt(report.get("by_confluence", {}),  "By Confluence")
    _fmt(report.get("by_poc", {}),         "POC at entry")
    _fmt(report.get("by_dxy", {}),         "By DXY Bias (Gold)")

    recs = report.get("recommendations", [])
    if recs:
        logger.info(f"\n  ⚡ RECOMMENDATIONS ({len(recs)})")
        for r in recs:
            logger.info(f"    → {r}")

    logger.info(f"{'═'*55}\n")


def _save_recommendations(ticker: str, report: dict) -> None:
    """Save adaptive adjustments to config/adaptive_backtest.json."""
    adj = report.get("config_adjustments", {})
    recs = report.get("recommendations", [])
    if not adj and not recs:
        return
    safe = ticker.replace("=", "_").replace("^", "").replace("-", "_")
    path = os.path.join(
        os.path.dirname(__file__), "..", "config", f"adaptive_backtest_{safe}.json"
    )
    payload = {
        "ticker":            ticker,
        "adjustments":       adj,
        "recommendations":   recs,
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Recommendations saved → {path}")
    except Exception as e:
        logger.warning(f"Could not save recommendations: {e}")


# ── Exit logic ────────────────────────────────────────────────────────────────

def _pnl_at_price(trade: dict, exit_price: float) -> float:
    dpp = trade["dollar_per_point"]
    if trade["direction"] == "BUY":
        return (exit_price - trade["entry"]) * dpp - COMMISSION
    else:
        return (trade["entry"] - exit_price) * dpp - COMMISSION


def _check_exit(
    trade: dict, bar: pd.Series, bar_idx: int,
    max_hold: int = MAX_HOLD_BARS, eod_hour: int = EOD_HOUR_UTC
) -> tuple[float, str] | None:
    hi = float(bar["high"])
    lo = float(bar["low"])
    ts = bar.name
    sl = trade["stop_loss"]
    tp = trade["take_profit"]
    d  = trade["direction"]

    if ts.hour >= eod_hour:
        return _pnl_at_price(trade, float(bar["close"])), "eod_close"
    if bar_idx - trade["bar_open"] >= max_hold:
        return _pnl_at_price(trade, float(bar["close"])), "timeout"

    if d == "BUY":
        if lo <= sl and hi >= tp:
            return _pnl_at_price(trade, sl), "stop_loss"
        if lo <= sl:
            return _pnl_at_price(trade, sl), "stop_loss"
        if hi >= tp:
            return _pnl_at_price(trade, tp), "take_profit"
    else:
        if hi >= sl and lo <= tp:
            return _pnl_at_price(trade, sl), "stop_loss"
        if hi >= sl:
            return _pnl_at_price(trade, sl), "stop_loss"
        if lo <= tp:
            return _pnl_at_price(trade, tp), "take_profit"
    return None


def _check_exit_daily(trade: dict, bar: pd.Series, bar_idx: int) -> tuple[float, str] | None:
    """Daily proxy exit: no EOD, timeout = DAILY_HOLD_BARS days."""
    hi = float(bar["high"])
    lo = float(bar["low"])
    sl = trade["stop_loss"]
    tp = trade["take_profit"]
    d  = trade["direction"]

    if bar_idx - trade["bar_open"] >= DAILY_HOLD_BARS:
        return _pnl_at_price(trade, float(bar["close"])), "timeout"

    if d == "BUY":
        if lo <= sl and hi >= tp:
            return _pnl_at_price(trade, sl), "stop_loss"
        if lo <= sl:
            return _pnl_at_price(trade, sl), "stop_loss"
        if hi >= tp:
            return _pnl_at_price(trade, tp), "take_profit"
    else:
        if hi >= sl and lo <= tp:
            return _pnl_at_price(trade, sl), "stop_loss"
        if hi >= sl:
            return _pnl_at_price(trade, sl), "stop_loss"
        if lo <= tp:
            return _pnl_at_price(trade, tp), "take_profit"
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def _utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _metrics(trades: list[dict], equity_curve: list[float], capital: float) -> dict:
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0,
            "annualised_return": 0.0, "avg_trade_pnl": 0.0, "calmar": 0.0,
        }

    pnls   = [t["pnl_usd"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(trades)

    win_rate      = len(wins) / n
    profit_factor = sum(wins) / max(abs(sum(losses)), 1e-9) if losses else float("inf")
    avg_pnl       = sum(pnls) / n

    pnl_pcts = [p / capital for p in pnls]
    sharpe   = 0.0
    if len(pnl_pcts) > 2 and np.std(pnl_pcts) > 0:
        sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252)

    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / np.maximum(peak, 1.0)
    max_dd = float(dd.max())

    ann_ret = 0.0
    try:
        t0   = pd.Timestamp(trades[0]["ts_open"])
        t1   = pd.Timestamp(trades[-1].get("ts_close", trades[-1]["ts_open"]))
        days = max((t1 - t0).days, 1)
        ann_ret = (equity_curve[-1] / capital) ** (365.0 / days) - 1
    except Exception:
        pass

    calmar = ann_ret / max_dd if max_dd > 0 else 0.0

    return {
        "n_trades":          n,
        "win_rate":          round(win_rate, 4),
        "sharpe":            round(sharpe, 3),
        "max_drawdown":      round(max_dd, 4),
        "profit_factor":     round(profit_factor, 3),
        "annualised_return": round(ann_ret, 4),
        "avg_trade_pnl":     round(avg_pnl, 2),
        "calmar":            round(calmar, 3),
    }


def _log_results(ticker: str, m: dict, mode: str = "") -> None:
    logger.info("=" * 55)
    logger.info(f"BACKTEST {mode} — {ticker}  ({m.get('data_years', '?')} years)")
    logger.info(f"  Trades:        {m['n_trades']}")
    logger.info(f"  Win Rate:      {m['win_rate']:.1%}")
    logger.info(f"  Sharpe:        {m['sharpe']}")
    logger.info(f"  Max Drawdown:  {m['max_drawdown']:.1%}")
    logger.info(f"  Profit Factor: {m['profit_factor']}")
    logger.info(f"  Ann. Return:   {m['annualised_return']:.1%}")
    logger.info("=" * 55)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(
    ticker: str,
    eq_precision: list[float],
    trades_precision: list[dict],
    eq_daily: list[float] | None = None,
    trades_daily: list[dict] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt

        n_rows = 3 if eq_daily else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))

        # Precision equity curve
        ax1 = axes[0]
        ax1.plot(eq_precision, color="steelblue", linewidth=1.5, label="Precision 4H→1H")
        ax1.axhline(eq_precision[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax1.set_title(f"Day Trading Backtest — {ticker}  (VP / VWAP / CVD / POC)")
        ax1.set_ylabel("Equity ($)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Precision trade P&L bars
        ax2 = axes[1]
        pnls   = [t.get("pnl_usd", 0) for t in trades_precision]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.75)
        ax2.axhline(0, color="black", linewidth=0.8)
        wins   = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        ax2.set_title(f"Precision Trade P&L  ({wins}W / {losses}L)")
        ax2.set_ylabel("P&L ($)")
        ax2.grid(True, alpha=0.3)

        # Daily proxy equity curve
        if eq_daily and trades_daily and n_rows == 3:
            ax3 = axes[2]
            ax3.plot(eq_daily, color="darkorange", linewidth=1.2, label="Daily proxy 1D (10y)")
            ax3.axhline(eq_daily[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            wins_d   = sum(1 for t in trades_daily if t.get("pnl_usd", 0) > 0)
            losses_d = sum(1 for t in trades_daily if t.get("pnl_usd", 0) <= 0)
            ax3.set_title(f"Daily Proxy Equity  ({wins_d}W / {losses_d}L  over 10+ years)")
            ax3.set_ylabel("Equity ($)")
            ax3.grid(True, alpha=0.3)
            ax3.legend()

        plt.tight_layout()
        safe = ticker.replace("=", "_").replace("^", "").replace("-", "_")
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{safe}_dt_backtest.png")
        plt.savefig(path, dpi=120)
        plt.close()
        logger.info(f"Backtest plot saved: {path}")

    except ImportError:
        logger.warning("matplotlib not available — plot skipped")
    except Exception as e:
        logger.warning(f"Plot error: {e}")
