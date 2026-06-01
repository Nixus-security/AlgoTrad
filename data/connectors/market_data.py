"""
Market data connector — primary: yfinance (free).
Falls back to Alpha Vantage for intraday when needed.
"""
from __future__ import annotations
import os
import random
import time
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from utils.logger import logger

_YF_MAX_RETRIES = 3
_YF_RETRY_BASE  = 2.0   # seconds — doubles each attempt + jitter


class MarketDataConnector:
    def __init__(self):
        av_key = os.getenv("ALPHA_VANTAGE_KEY", "demo")
        self._av = TimeSeries(key=av_key, output_format="pandas")

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        interval: str = "1h",
        period: str = "60d",
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """
        interval: '5m' | '15m' | '1h' | '1d'
        period:   yfinance period string ('5d', '60d', '1y', '2y', 'max')
        """
        if source == "yfinance":
            return self._yf_ohlcv(ticker, interval, period)
        return self._av_ohlcv(ticker, interval)

    def _yf_ohlcv(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        for attempt in range(_YF_MAX_RETRIES):
            try:
                logger.info(f"Fetching {ticker} [{interval}] via yfinance")
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True)
                # yfinance ≥ 0.2 returns MultiIndex; level ordering varies by version
                if isinstance(df.columns, pd.MultiIndex):
                    price_names = {"open", "high", "low", "close", "volume"}
                    for lvl in range(df.columns.nlevels):
                        vals = df.columns.get_level_values(lvl)
                        if any(str(v).lower() in price_names for v in vals):
                            df.columns = vals
                            break
                    else:
                        df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).lower() for c in df.columns]
                df.index = pd.to_datetime(df.index, utc=True)
                df.dropna(inplace=True)
                if not df.empty:
                    return df
                logger.warning(
                    f"{ticker}: yfinance empty response "
                    f"(attempt {attempt + 1}/{_YF_MAX_RETRIES})"
                )
            except Exception as e:
                logger.warning(
                    f"{ticker}: yfinance error attempt {attempt + 1}/{_YF_MAX_RETRIES}: {e}"
                )

            if attempt < _YF_MAX_RETRIES - 1:
                wait = _YF_RETRY_BASE * (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.info(f"{ticker}: retrying in {wait:.1f}s")
                time.sleep(wait)

        # All retries exhausted — Alpha Vantage fallback (intraday only)
        logger.warning(f"{ticker}: yfinance failed after {_YF_MAX_RETRIES} attempts — trying Alpha Vantage")
        try:
            return self._av_ohlcv(ticker, interval)
        except Exception as e:
            logger.error(f"{ticker}: Alpha Vantage fallback also failed: {e}")
            return pd.DataFrame()

    def _av_ohlcv(self, ticker: str, interval: str) -> pd.DataFrame:
        iv_map = {"5m": "5min", "15m": "15min", "1h": "60min"}
        av_iv = iv_map.get(interval, "60min")
        logger.info(f"Fetching {ticker} [{av_iv}] via Alpha Vantage")
        df, _ = self._av.get_intraday(ticker, interval=av_iv, outputsize="full")
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        return df

    # ── Real-time quote ────────────────────────────────────────────────────────
    def get_quote(self, ticker: str) -> dict:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        return {
            "ticker": ticker,
            "price": info.last_price,
            "bid": getattr(info, "bid", None),
            "ask": getattr(info, "ask", None),
            "volume": info.three_month_average_volume,
        }

    # ── Order book snapshot (yfinance best-effort) ────────────────────────────
    def get_order_book(self, ticker: str) -> dict:
        tk = yf.Ticker(ticker)
        return {"bids": tk.bids, "asks": tk.asks}

    # ── Batch fetch for multiple assets ───────────────────────────────────────
    def batch_ohlcv(
        self, tickers: list[str], interval: str = "1h", period: str = "60d"
    ) -> dict[str, pd.DataFrame]:
        result = {}
        for t in tickers:
            try:
                result[t] = self.get_ohlcv(t, interval, period)
                time.sleep(0.3)  # Rate-limit courtesy
            except Exception as e:
                logger.error(f"Failed {t}: {e}")
        return result
