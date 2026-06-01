"""
Adaptive strategy selector.
Chooses scalping vs day-trading based on current volatility regime.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class StrategySelector:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def select(self, df: pd.DataFrame) -> tuple[str, str]:
        """
        Returns (strategy_name, timeframe).
        strategy_name: 'scalping' | 'day_trading'
        timeframe:     '5m' | '1h' | '1d'
        """
        volatility = self._current_volatility(df)
        adx = self._adx(df)

        # High volatility + low trend strength → scalping (range play)
        if volatility > 0.015 and adx < 25:
            logger.info(f"Strategy: scalping (vol={volatility:.4f}, ADX={adx:.1f})")
            return "scalping", self.cfg["timeframes"]["scalping"]

        # Strong trend → day trading on 1h
        if adx >= 25:
            logger.info(f"Strategy: day_trading (vol={volatility:.4f}, ADX={adx:.1f})")
            return "day_trading", self.cfg["timeframes"]["day_trading"]

        # Default: day trading
        return "day_trading", self.cfg["timeframes"]["day_trading"]

    @staticmethod
    def _current_volatility(df: pd.DataFrame, window: int = 20) -> float:
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        return float(log_ret.tail(window).std())

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index — simplified DMI calculation."""
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return float(dx.rolling(period).mean().iloc[-1])
