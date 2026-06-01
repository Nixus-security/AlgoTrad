"""
Advanced feature engineering.
Inputs: intraday OHLCV, daily OHLCV, microstructure, catalyst, sentiment.
Outputs: FeatureSet with all computed features + flat array for ML.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class FeatureSet:
    # Microstructure
    rvol: float                  # Relative volume vs 30d avg
    spread_pct: float            # Bid-ask spread %
    liquidity_score: float       # Composite 0–1

    # Price action
    gap_pct: float               # Today's gap % from prev close
    gap_strength: float          # |gap| / ATR
    vwap_position: float         # 0=below VWAP, 1=above, 0.9=reclaim
    momentum_1h: float           # 1-bar return
    momentum_strength: float     # momentum / volatility (z-score like)
    wick_rejection_upper: float  # upper wick / candle range
    wick_rejection_lower: float  # lower wick / candle range

    # Volume
    volume_anomaly_z: float      # z-score of current volume
    volume_trend: float          # normalized slope of last 5 bars

    # Breakout
    breakout_strength: float     # 0–1
    fake_breakout_prob: float    # 0–1

    # Regime
    volatility_percentile: float # current vol / historical vol (capped at 3)

    # External signals
    catalyst_score: float        # 0–1
    sentiment_score: float       # -1 to +1 (bearish→bullish)
    fomo_score: float            # 0–1
    squeeze_prob: float          # 0–1

    # Flat array (auto-built)
    array: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.array = [
            self.rvol, self.spread_pct, self.liquidity_score,
            self.gap_pct, self.gap_strength, self.vwap_position,
            self.momentum_1h, self.momentum_strength,
            self.wick_rejection_upper, self.wick_rejection_lower,
            self.volume_anomaly_z, self.volume_trend,
            self.breakout_strength, self.fake_breakout_prob,
            self.volatility_percentile,
            self.catalyst_score, self.sentiment_score,
            self.fomo_score, self.squeeze_prob,
        ]


class FeatureEngine:

    def compute(
        self,
        df: pd.DataFrame,        # intraday OHLCV (1h)
        df_daily: pd.DataFrame,  # daily OHLCV (30d)
        micro=None,              # MicrostructureData | None
        catalyst=None,           # CatalystData | None
        sentiment=None,          # SentimentData | None
    ) -> FeatureSet:

        rvol = self._rvol(df)
        gap_pct, gap_str = self._gap(df_daily)
        vwap_pos = self._vwap_position(df)
        mom_1h, mom_str = self._momentum(df)
        wick_up, wick_dn = self._wick_rejection(df)
        vol_z, vol_trend = self._volume_anomaly(df)
        bk_str = self._breakout_strength(df)
        fake_prob = self._fake_breakout_prob(df)
        vol_pct = self._volatility_percentile(df)

        spread_pct = getattr(micro, "spread_pct", 0.0) or 0.0
        liquidity = getattr(micro, "liquidity_score", 0.5) or 0.5
        cat_score = getattr(catalyst, "catalyst_score", 0.0) or 0.0
        sent_score = float((getattr(sentiment, "bullish_pct", 0.5) or 0.5) * 2 - 1)
        fomo = getattr(sentiment, "fomo_score", 0.0) or 0.0
        squeeze = getattr(sentiment, "squeeze_probability", 0.0) or 0.0

        return FeatureSet(
            rvol=rvol, spread_pct=spread_pct, liquidity_score=liquidity,
            gap_pct=gap_pct, gap_strength=gap_str, vwap_position=vwap_pos,
            momentum_1h=mom_1h, momentum_strength=mom_str,
            wick_rejection_upper=wick_up, wick_rejection_lower=wick_dn,
            volume_anomaly_z=vol_z, volume_trend=vol_trend,
            breakout_strength=bk_str, fake_breakout_prob=fake_prob,
            volatility_percentile=vol_pct,
            catalyst_score=cat_score, sentiment_score=sent_score,
            fomo_score=fomo, squeeze_prob=squeeze,
        )

    # ── RVOL ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _rvol(df: pd.DataFrame, window: int = 20) -> float:
        if "volume" not in df.columns or df["volume"].isna().all():
            return 1.0
        vol = df["volume"].replace(0, np.nan)
        avg = vol.rolling(window).mean().iloc[-1]
        curr = vol.iloc[-1]
        if not avg or pd.isna(avg) or avg == 0:
            return 1.0
        return float(np.clip(curr / avg, 0, 20))

    # ── Gap % ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _gap(df_daily: pd.DataFrame) -> tuple[float, float]:
        if len(df_daily) < 2:
            return 0.0, 0.0
        prev_close = float(df_daily["close"].iloc[-2])
        today_open = float(df_daily["open"].iloc[-1])
        if prev_close == 0:
            return 0.0, 0.0
        gap_pct = (today_open - prev_close) / prev_close
        atr = (df_daily["high"] - df_daily["low"]).rolling(14).mean().iloc[-1]
        gap_strength = float(np.clip(abs(today_open - prev_close) / max(atr, 1e-9), 0, 5))
        return float(gap_pct), gap_strength

    # ── VWAP position ─────────────────────────────────────────────────────────
    @staticmethod
    def _vwap_position(df: pd.DataFrame) -> float:
        if len(df) < 3:
            return 0.5
        has_vol = "volume" in df.columns and not df["volume"].isna().all()
        if has_vol:
            vol = df["volume"].replace(0, np.nan)
            typ = (df["high"] + df["low"] + df["close"]) / 3
            cum_vol = vol.cumsum().replace(0, np.nan)
            vwap = (typ * vol).cumsum() / cum_vol
        else:
            vwap = (df["high"] + df["low"] + df["close"]) / 3

        price = df["close"].iloc[-1]
        vwap_last = vwap.iloc[-1]
        if pd.isna(vwap_last) or vwap_last == 0:
            return 0.5

        was_below = df["close"].iloc[-3] < vwap.iloc[-3]
        now_above = price > vwap_last
        if was_below and now_above:
            return 0.9  # VWAP reclaim — strong signal
        return 1.0 if price > vwap_last else 0.0

    # ── Momentum ──────────────────────────────────────────────────────────────
    @staticmethod
    def _momentum(df: pd.DataFrame) -> tuple[float, float]:
        if len(df) < 2:
            return 0.0, 0.0
        p_now = float(df["close"].iloc[-1])
        p_prev = float(df["close"].iloc[-2])
        ret_1h = (p_now - p_prev) / p_prev if p_prev != 0 else 0.0
        vol_20 = df["close"].pct_change().rolling(20).std().iloc[-1]
        strength = ret_1h / max(vol_20, 1e-9) if not pd.isna(vol_20) else 0.0
        return float(ret_1h), float(np.clip(abs(strength), 0, 5))

    # ── Wick rejection ────────────────────────────────────────────────────────
    @staticmethod
    def _wick_rejection(df: pd.DataFrame) -> tuple[float, float]:
        row = df.iloc[-1]
        h, l = float(row["high"]), float(row["low"])
        o, c = float(row["open"]), float(row["close"])
        rng = h - l
        if rng < 1e-9:
            return 0.0, 0.0
        upper = (h - max(o, c)) / rng
        lower = (min(o, c) - l) / rng
        return float(upper), float(lower)

    # ── Volume anomaly ────────────────────────────────────────────────────────
    @staticmethod
    def _volume_anomaly(df: pd.DataFrame, window: int = 20) -> tuple[float, float]:
        if "volume" not in df.columns or len(df) < window:
            return 0.0, 0.0
        vol = df["volume"].replace(0, np.nan)
        mu = vol.rolling(window).mean().iloc[-1]
        sigma = vol.rolling(window).std().iloc[-1]
        curr = vol.iloc[-1]
        z = float((curr - mu) / max(sigma, 1e-9)) if (mu and not pd.isna(mu)) else 0.0
        slope = float(np.polyfit(range(5), vol.iloc[-5:].fillna(0).values, 1)[0])
        trend = float(np.clip(slope / max(mu, 1e-9), -1, 1)) if (mu and mu > 0) else 0.0
        return float(np.clip(z, -5, 5)), trend

    # ── Breakout strength ─────────────────────────────────────────────────────
    @staticmethod
    def _breakout_strength(df: pd.DataFrame, lookback: int = 20) -> float:
        if len(df) < lookback:
            return 0.0
        resistance = float(df["close"].iloc[-lookback:-1].max())
        price = float(df["close"].iloc[-1])
        if price <= resistance:
            return 0.0
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        bk_pct = (price - resistance) / max(atr, 1e-9)
        rvol_factor = 1.0
        if "volume" in df.columns:
            avg = df["volume"].rolling(lookback).mean().iloc[-1]
            curr = df["volume"].iloc[-1]
            rvol_factor = float(curr / max(avg, 1e-9)) if avg and avg > 0 else 1.0
        return float(np.clip(bk_pct * 0.5 + (rvol_factor - 1) * 0.5, 0, 1))

    # ── Fake breakout probability ─────────────────────────────────────────────
    @staticmethod
    def _fake_breakout_prob(df: pd.DataFrame, lookback: int = 20) -> float:
        if len(df) < lookback + 3:
            return 0.3
        closes = df["close"].values
        resistance = float(np.max(closes[-lookback - 3:-3]))
        # Check last 3 bars for a breakout that reversed
        broke_idx = next(
            (i for i in range(1, 4) if closes[-i - 1] > resistance), None
        )
        if broke_idx is None:
            return 0.2
        if closes[-1] < resistance:
            return 0.85  # Price fell back below → likely fake
        row = df.iloc[-broke_idx - 1]
        rng = float(row["high"] - row["low"])
        upper_wick = float((row["high"] - max(row["open"], row["close"])) / max(rng, 1e-9))
        return float(np.clip(0.2 + upper_wick * 0.6, 0, 1))

    # ── Volatility percentile ─────────────────────────────────────────────────
    @staticmethod
    def _volatility_percentile(df: pd.DataFrame, window: int = 20) -> float:
        if len(df) < window + 5:
            return 0.5
        rets = df["close"].pct_change().dropna()
        curr_vol = rets.tail(window).std()
        hist_vol = rets.std()
        return float(np.clip(curr_vol / max(hist_vol, 1e-9), 0, 3) / 3)
