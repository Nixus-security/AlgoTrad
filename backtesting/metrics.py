"""
Backtesting performance metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return wins / len(trades)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02, periods: int = 252) -> float:
    """Annualised Sharpe."""
    excess = returns - risk_free / periods
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / returns.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    roll_max = equity_curve.cummax()
    dd = (equity_curve - roll_max) / roll_max
    return float(abs(dd.min()))


def profit_factor(trades: list[dict]) -> float:
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0) or 1e-9
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)) or 1e-9
    return gross_profit / gross_loss


def calmar_ratio(annualised_return: float, mdd: float) -> float:
    return annualised_return / mdd if mdd != 0 else 0.0


def summarise(
    trades: list[dict], equity_curve: pd.Series, periods: int = 252
) -> dict:
    if not trades:
        return {"error": "No trades executed"}
    returns = equity_curve.pct_change().dropna()
    ann_ret = (1 + returns.mean()) ** periods - 1
    mdd = max_drawdown(equity_curve)
    return {
        "n_trades": len(trades),
        "win_rate": round(win_rate(trades), 4),
        "sharpe": round(sharpe_ratio(returns, periods=periods), 3),
        "max_drawdown": round(mdd, 4),
        "profit_factor": round(profit_factor(trades), 3),
        "calmar": round(calmar_ratio(ann_ret, mdd), 3),
        "annualised_return": round(ann_ret, 4),
        "avg_trade_pnl": round(np.mean([t["pnl"] for t in trades]), 4),
    }
