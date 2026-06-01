"""
Technical analysis engine.
Returns a TechnicalSignal dataclass with direction, strength [0-1], and indicator values.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class TechnicalSignal:
    direction: str        # "BUY" | "SELL" | "NEUTRAL"
    strength: float       # 0.0 – 1.0
    rsi: float
    bb_position: float    # 0=lower band, 1=upper band
    ma_alignment: float   # positive = bullish stack, negative = bearish
    macd_hist: float
    atr: float
    details: dict


class TechnicalAnalyzer:

    # ── Compute all indicators ─────────────────────────────────────────────────
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.obv(append=True)
        df.dropna(inplace=True)
        return df

    # ── Generate signal from latest bar ───────────────────────────────────────
    def get_signal(self, df: pd.DataFrame) -> TechnicalSignal:
        df = self.compute(df)
        if df.empty:
            return TechnicalSignal(
                direction="NEUTRAL", strength=0.5, rsi=50.0, bb_position=0.5,
                ma_alignment=0.0, macd_hist=0.0, atr=0.0, details={},
            )
        row = df.iloc[-1]

        rsi = row.get("RSI_14", 50.0)
        macd_hist = row.get("MACDh_12_26_9", 0.0)
        bb_upper = row.get("BBU_20_2.0", row["close"] * 1.02)
        bb_lower = row.get("BBL_20_2.0", row["close"] * 0.98)
        bb_pos = (row["close"] - bb_lower) / max(bb_upper - bb_lower, 1e-9)
        ema9 = row.get("EMA_9", row["close"])
        ema21 = row.get("EMA_21", row["close"])
        ema50 = row.get("EMA_50", row["close"])
        atr = row.get("ATRr_14", row["close"] * 0.01)

        # MA alignment score: fully bullish = +1, fully bearish = -1
        ma_align = 0.0
        if ema9 > ema21:
            ma_align += 0.5
        if ema21 > ema50:
            ma_align += 0.5
        if ema9 < ema21:
            ma_align -= 0.5
        if ema21 < ema50:
            ma_align -= 0.5

        # Score components (each in [0,1])
        rsi_score = self._rsi_score(rsi)
        bb_score = self._bb_score(bb_pos)
        ma_score = (ma_align + 1) / 2          # re-range [-1,1] → [0,1]
        macd_score = 1.0 if macd_hist > 0 else 0.0

        bull_score = np.mean([rsi_score, bb_score, ma_score, macd_score])

        if bull_score >= 0.65:
            direction = "BUY"
            strength = bull_score
        elif bull_score <= 0.35:
            direction = "SELL"
            strength = 1.0 - bull_score
        else:
            direction = "NEUTRAL"
            strength = 0.5

        return TechnicalSignal(
            direction=direction,
            strength=strength,
            rsi=rsi,
            bb_position=bb_pos,
            ma_alignment=ma_align,
            macd_hist=macd_hist,
            atr=atr,
            details={
                "rsi_score": rsi_score,
                "bb_score": bb_score,
                "ma_score": ma_score,
                "macd_score": macd_score,
                "ema9": ema9, "ema21": ema21, "ema50": ema50,
                "bb_upper": bb_upper, "bb_lower": bb_lower,
            },
        )

    # ── Indicator scoring helpers ──────────────────────────────────────────────
    @staticmethod
    def _rsi_score(rsi: float) -> float:
        """RSI: oversold (<30) → bullish score 1; overbought (>70) → 0."""
        if rsi < 30:
            return 1.0
        if rsi > 70:
            return 0.0
        # Linear in [30,70]
        return 1.0 - (rsi - 30) / 40.0

    @staticmethod
    def _bb_score(bb_pos: float) -> float:
        """Price near lower band = oversold = bullish."""
        return 1.0 - min(max(bb_pos, 0.0), 1.0)
