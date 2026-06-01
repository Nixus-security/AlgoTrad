"""
Feature engineering pipeline — runs after raw OHLCV fetch.
Outputs normalised feature matrix ready for ML + TA modules.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.scaler = RobustScaler()

    # ── Log-returns + lag features ────────────────────────────────────────────
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["ret_5"]   = df["log_ret"].rolling(5).mean()
        df["ret_20"]  = df["log_ret"].rolling(20).mean()
        df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan)
        return df

    # ── Volatility ────────────────────────────────────────────────────────────
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["vol_20"] = df["log_ret"].rolling(20).std()
        df["atr"]    = self._atr(df, 14)
        return df

    # ── Technical indicators ──────────────────────────────────────────────────
    def add_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]

        # RSI-14
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta).clip(lower=0).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - 100 / (1 + rs)) / 100  # normalise to [0,1]

        # MACD histogram (normalised by price)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (macd - signal) / close.replace(0, np.nan)

        # Bollinger %B
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        upper  = sma20 + 2 * std20
        lower  = sma20 - 2 * std20
        band_w = (upper - lower).replace(0, np.nan)
        df["bb_pct"] = ((close - lower) / band_w).clip(0, 1)

        return df

    # ── VWAP deviation ────────────────────────────────────────────────────────
    def add_vwap_dev(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "vwap" in df.columns:
            df["vwap_dev"] = (df["close"] / df["vwap"].replace(0, np.nan)) - 1
        else:
            df["vwap_dev"] = 0.0
        return df

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_cp = (df["high"] - df["close"].shift()).abs()
        low_cp = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ── Volume features ───────────────────────────────────────────────────────
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        has_volume = (
            "volume" in df.columns
            and not df["volume"].isna().all()
            and (df["volume"] != 0).any()
        )
        if has_volume:
            rolling_mean = df["volume"].rolling(20).mean().replace(0, np.nan)
            df["vol_ratio"] = df["volume"] / rolling_mean
            cum_vol = df["volume"].cumsum().replace(0, np.nan)
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / cum_vol
        else:
            df["vol_ratio"] = 1.0
            df["vwap"] = df["close"]
        return df

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_returns(df)
        df = self.add_volatility(df)
        df = self.add_volume_features(df)
        df = self.add_technical(df)
        df = self.add_vwap_dev(df)
        df.dropna(inplace=True)
        return df

    # ── Build 3-D sequence tensor for LSTM: (samples, timesteps, features) ───
    def build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        threshold: float = 0.003,   # 0.3% min move to label — skip ambiguous zone
        horizon: int = 3,           # bars ahead to predict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        threshold: forward-return cutoff. Samples where |fwd_ret| < threshold are dropped.
        horizon:   number of bars ahead to compute forward return.
        Scaler is fit here on the full passed data (caller controls train/val split).
        """
        available = [c for c in feature_cols if c in df.columns]
        data   = df[available].values.astype(np.float32)
        closes = df["close"].values

        data_scaled = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.lookback, len(data_scaled) - horizon):
            c0 = closes[i]
            ch = closes[i + horizon]
            if c0 <= 0:
                continue
            fwd_ret = (ch - c0) / c0

            if fwd_ret > threshold:
                label = 1
            elif fwd_ret < -threshold:
                label = 0
            else:
                continue  # ambiguous — skip

            X.append(data_scaled[i - self.lookback : i])
            y.append(label)

        if not X:
            return np.array([]).reshape(0, self.lookback, len(available)), np.array([])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
