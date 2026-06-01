"""
Statistical analysis: seasonality, mean-reversion z-score, correlation regime.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


@dataclass
class StatisticalSignal:
    direction: str      # "BUY" | "SELL" | "NEUTRAL"
    strength: float
    z_score: float      # Mean-reversion z-score on 20-period window
    is_mean_reverting: bool
    hurst: float        # Hurst exponent (< 0.5 = mean-reverting)
    seasonality_bias: float  # +1 bullish, -1 bearish seasonal pattern


class StatisticalAnalyzer:

    # ── Hurst exponent via R/S analysis ───────────────────────────────────────
    @staticmethod
    def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    # ── Z-score mean reversion signal ─────────────────────────────────────────
    @staticmethod
    def zscore_signal(series: pd.Series, window: int = 20) -> float:
        mu = series.rolling(window).mean().iloc[-1]
        sigma = series.rolling(window).std().iloc[-1]
        return (series.iloc[-1] - mu) / max(sigma, 1e-9)

    # ── ADF stationarity test (mean-reversion proxy) ─────────────────────────
    @staticmethod
    def is_mean_reverting(series: pd.Series) -> bool:
        result = adfuller(series.dropna(), maxlag=12)
        return result[1] < 0.05  # p-value < 5% → reject unit root → stationary

    # ── Seasonal day-of-week bias (based on historical returns) ───────────────
    @staticmethod
    def seasonality_bias(df: pd.DataFrame) -> float:
        df = df.copy()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["dow"] = df.index.dayofweek
        today_dow = pd.Timestamp.today().dayofweek
        dow_mean = df.groupby("dow")["log_ret"].mean()
        bias = dow_mean.get(today_dow, 0.0)
        # Normalise to [-1, +1] using cross-day range
        rng = dow_mean.max() - dow_mean.min()
        return float(bias / rng) if rng != 0 else 0.0

    # ── Full signal ────────────────────────────────────────────────────────────
    def get_signal(self, df: pd.DataFrame) -> StatisticalSignal:
        close = df["close"]
        z = self.zscore_signal(close)
        mr = self.is_mean_reverting(close.tail(100))
        hurst = self.hurst_exponent(close.values[-50:])
        seas = self.seasonality_bias(df)

        # Mean-reversion play: extreme z → fade it
        if z < -2.0 and mr:
            direction, strength = "BUY", min(abs(z) / 3.0, 1.0)
        elif z > 2.0 and mr:
            direction, strength = "SELL", min(abs(z) / 3.0, 1.0)
        else:
            # Trend-following mode
            if seas > 0.2:
                direction, strength = "BUY", abs(seas)
            elif seas < -0.2:
                direction, strength = "SELL", abs(seas)
            else:
                direction, strength = "NEUTRAL", 0.5

        return StatisticalSignal(
            direction=direction,
            strength=strength,
            z_score=z,
            is_mean_reverting=mr,
            hurst=hurst,
            seasonality_bias=seas,
        )
