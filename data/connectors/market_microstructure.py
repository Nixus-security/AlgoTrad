"""
Market microstructure: spread, liquidity, volume, RVOL — all via yfinance.
Works for both equities and crypto (BTC-USD, SOL-USD).
Float/short interest/beta/premarket fields are 0 for crypto — not applicable.
"""
from __future__ import annotations
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from utils.logger import logger


@dataclass
class MicrostructureData:
    ticker: str
    bid: float
    ask: float
    spread_pct: float           # (ask - bid) / mid
    mid: float
    liquidity_score: float      # 0–1 composite (volume + tight spread)
    float_shares_m: float       # float in millions
    short_interest_pct: float   # % of float sold short
    short_ratio: float          # days-to-cover
    beta: float
    premarket_volume: float
    premarket_change_pct: float # % vs prev close
    avg_volume_30d: float
    rvol: float                 # current vol / 30d avg


class MicrostructureConnector:

    def get(self, ticker: str) -> MicrostructureData:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            fast = tk.fast_info
        except Exception as e:
            logger.error(f"Microstructure {ticker}: {e}")
            return self._empty(ticker)

        bid = float(info.get("bid") or 0)
        ask = float(info.get("ask") or 0)
        mid = (bid + ask) / 2 if bid and ask else float(getattr(fast, "last_price", 0) or 0)
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0

        avg_vol = float(info.get("averageVolume") or info.get("averageVolume10days") or 0)
        curr_vol = float(info.get("volume") or info.get("regularMarketVolume") or 0)
        rvol = curr_vol / avg_vol if avg_vol > 0 else 1.0

        # Liquidity: tight spread is main signal for crypto (vol scale varies by asset)
        # For crypto, vol_score uses a lower denominator since volume is in coin units
        is_crypto = ticker.endswith("-USD") or ticker.endswith("-USDT")
        vol_denom = 50_000.0 if is_crypto else 5_000_000.0
        vol_score = min(avg_vol / vol_denom, 1.0) if avg_vol > 0 else 0.5
        spread_score = max(0.0, 1.0 - spread_pct * 200)
        liquidity_score = (vol_score + spread_score) / 2

        # Float/short/beta: not applicable for crypto — default to neutral values
        float_raw = float(info.get("floatShares") or 0)
        shares_short = float(info.get("sharesShort") or 0)
        short_pct = shares_short / float_raw if float_raw > 0 else 0.0
        short_ratio = float(info.get("shortRatio") or 0)
        beta = float(info.get("beta") or 1.0)

        # Premarket: not applicable for crypto (24/7) — default to 0
        prev_close = float(info.get("previousClose") or mid or 1)
        premarket_price = float(info.get("preMarketPrice") or prev_close)
        premarket_vol = float(info.get("preMarketVolume") or 0)
        premarket_chg = (premarket_price - prev_close) / prev_close if prev_close else 0.0

        return MicrostructureData(
            ticker=ticker,
            bid=bid, ask=ask, spread_pct=spread_pct, mid=mid,
            liquidity_score=liquidity_score,
            float_shares_m=float_raw / 1e6,
            short_interest_pct=short_pct,
            short_ratio=short_ratio,
            beta=beta,
            premarket_volume=premarket_vol,
            premarket_change_pct=premarket_chg,
            avg_volume_30d=avg_vol,
            rvol=rvol,
        )

    @staticmethod
    def _empty(ticker: str) -> MicrostructureData:
        return MicrostructureData(
            ticker=ticker, bid=0, ask=0, spread_pct=0, mid=0,
            liquidity_score=0.5, float_shares_m=0,
            short_interest_pct=0, short_ratio=0, beta=1.0,
            premarket_volume=0, premarket_change_pct=0,
            avg_volume_30d=0, rvol=1.0,
        )
