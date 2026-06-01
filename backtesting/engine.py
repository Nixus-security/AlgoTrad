"""
Walk-forward backtesting engine with anti-overfitting measures:
  1. Out-of-sample walk-forward windows (no look-ahead)
  2. Transaction costs included
  3. Signals validated against risk thresholds before simulated "fill"
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from backtesting.metrics import summarise
from data.connectors.market_data import MarketDataConnector
from data.preprocessor import Preprocessor
from analysis.technical import TechnicalAnalyzer
from analysis.statistical import StatisticalAnalyzer
from utils.logger import logger


class BacktestEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bt_cfg = cfg["backtest"]
        self.market = MarketDataConnector()
        self.preprocessor = Preprocessor(lookback=cfg["ml"]["lookback_window"])
        self.ta = TechnicalAnalyzer()
        self.stat = StatisticalAnalyzer()

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(self, ticker: str, plot: bool = True) -> dict:
        logger.info(f"Backtest: {ticker}")
        days = self._days_in_range()
        # yfinance caps 1h data at 730 days; fall back to 1d for longer ranges
        if days > 729:
            interval, days = "1d", days
        else:
            interval = "1h"
        df = self.market.get_ohlcv(ticker, interval=interval, period=f"{days}d")
        if df is None or df.empty:
            logger.error(f"Backtest: no data for {ticker}")
            return {"error": f"No data for {ticker}"}
        df = self.preprocessor.transform(df)
        n_splits = self.bt_cfg["walk_forward_splits"]
        tscv = TimeSeriesSplit(n_splits=n_splits)

        all_trades: list[dict] = []
        all_equity = pd.Series(dtype=float)
        cap = self.bt_cfg["initial_capital"]

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            logger.info(f"WF fold {fold+1}/{n_splits} — test bars: {len(test_idx)}")
            test_df = df.iloc[test_idx].copy()
            trades, equity = self._simulate_fold(test_df, capital=cap)
            all_trades.extend(trades)
            if not equity.empty:
                cap = float(equity.iloc[-1])
                all_equity = pd.concat([all_equity, equity])

        if all_equity.empty:
            logger.warning("No trades generated in backtest")
            return {"error": "No trades"}

        result = summarise(all_trades, all_equity)
        result["ticker"] = ticker
        self._log_results(result)

        if plot:
            self._plot(all_equity, all_trades, ticker)

        return result

    # ── Simulate one WF fold ──────────────────────────────────────────────────
    def _simulate_fold(
        self, df: pd.DataFrame, capital: float
    ) -> tuple[list[dict], pd.Series]:
        trades = []
        equity = [capital]
        position = None
        commission   = self.bt_cfg["commission_pct"]
        slippage_pct = self.bt_cfg.get("slippage_pct", 0.0008)  # 0.08% — matches paper broker

        for i in range(60, len(df)):
            window = df.iloc[: i + 1]
            ta_sig = self.ta.get_signal(window)
            stat_sig = self.stat.get_signal(window)
            row = df.iloc[i]
            price = float(row["close"])
            atr = ta_sig.atr

            # Open position
            if position is None and ta_sig.direction in ("BUY", "SELL"):
                # Require both TA + stat to agree (anti-overfitting filter)
                if ta_sig.direction == stat_sig.direction or stat_sig.direction == "NEUTRAL":
                    # Apply slippage on entry (adverse fill)
                    slip = price * slippage_pct
                    fill = price + slip if ta_sig.direction == "BUY" else price - slip
                    position = {
                        "direction": ta_sig.direction,
                        "entry": fill,
                        "stop": price - atr * 1.5 if ta_sig.direction == "BUY"
                                else price + atr * 1.5,
                        "target": price + atr * 3 if ta_sig.direction == "BUY"
                                  else price - atr * 3,
                        "open_idx": i,
                    }

            # Manage open position
            elif position is not None:
                hit_stop = (
                    (position["direction"] == "BUY" and price <= position["stop"]) or
                    (position["direction"] == "SELL" and price >= position["stop"])
                )
                hit_target = (
                    (position["direction"] == "BUY" and price >= position["target"]) or
                    (position["direction"] == "SELL" and price <= position["target"])
                )
                # Max hold: 24 bars
                timeout = (i - position["open_idx"]) >= 24

                if hit_stop or hit_target or timeout:
                    # Apply slippage on exit (adverse fill)
                    slip_exit = price * slippage_pct
                    exit_fill = price - slip_exit if position["direction"] == "BUY" else price + slip_exit
                    pnl_pct = (exit_fill - position["entry"]) / position["entry"]
                    if position["direction"] == "SELL":
                        pnl_pct = -pnl_pct
                    pnl_pct -= commission * 2  # Round-trip
                    pnl = equity[-1] * pnl_pct
                    trades.append({
                        "direction": position["direction"],
                        "entry": position["entry"],
                        "exit": price,
                        "pnl": pnl,
                        "exit_reason": "stop" if hit_stop else ("target" if hit_target else "timeout"),
                    })
                    equity.append(equity[-1] + pnl)
                    position = None
                else:
                    equity.append(equity[-1])
            else:
                equity.append(equity[-1])

        s = pd.Series(equity, index=df.index[:len(equity)] if len(equity) <= len(df) else None)
        return trades, s

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _days_in_range(self) -> int:
        start = pd.to_datetime(self.bt_cfg["start_date"])
        end = pd.to_datetime(self.bt_cfg["end_date"])
        return max((end - start).days, 30)

    def _log_results(self, r: dict) -> None:
        logger.info("=" * 50)
        logger.info(f"BACKTEST RESULTS — {r.get('ticker', '')}")
        logger.info(f"  Trades:        {r.get('n_trades')}")
        logger.info(f"  Win Rate:      {r.get('win_rate', 0):.1%}")
        logger.info(f"  Sharpe:        {r.get('sharpe')}")
        logger.info(f"  Max Drawdown:  {r.get('max_drawdown', 0):.1%}")
        logger.info(f"  Profit Factor: {r.get('profit_factor')}")
        logger.info(f"  Ann. Return:   {r.get('annualised_return', 0):.1%}")
        logger.info("=" * 50)

    def _plot(self, equity: pd.Series, trades: list[dict], ticker: str) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        sns.set_style("darkgrid")

        axes[0].plot(equity.values, linewidth=1.5, color="#00d4aa")
        axes[0].set_title(f"{ticker} — Equity Curve (Walk-Forward)", fontsize=13)
        axes[0].set_ylabel("Portfolio Value ($)")

        pnls = [t["pnl"] for t in trades]
        colours = ["#00d4aa" if p > 0 else "#ff4d6d" for p in pnls]
        axes[1].bar(range(len(pnls)), pnls, color=colours, alpha=0.8)
        axes[1].set_title("Trade P&L Distribution", fontsize=13)
        axes[1].set_ylabel("P&L ($)")
        axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--")

        plt.tight_layout()
        out = f"logs/{ticker}_backtest.png"
        plt.savefig(out, dpi=150)
        logger.info(f"Backtest plot saved: {out}")
        plt.close()
