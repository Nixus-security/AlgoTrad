"""
Market data connector — primary: Alpaca (free, stable).
Falls back to yfinance if Alpaca keys not configured.
"""
from __future__ import annotations
import os
import time
import datetime
import random
import pandas as pd
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()  # ensure .env loaded even when imported directly

_ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
_ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
_USE_ALPACA    = bool(_ALPACA_KEY and _ALPACA_SECRET)

# Alpaca timeframe mapping
_TF_MAP = {
    "1m":  ("1",  "Minute"),
    "5m":  ("5",  "Minute"),
    "15m": ("15", "Minute"),
    "1h":  ("1",  "Hour"),
    "4h":  ("4",  "Hour"),
    "1d":  ("1",  "Day"),
}

# Period string → timedelta
def _period_to_start(period: str) -> datetime.datetime:
    now = datetime.datetime.now(datetime.timezone.utc)
    p = period.lower()
    if p.endswith("d"):   return now - datetime.timedelta(days=int(p[:-1]))
    if p.endswith("mo"):  return now - datetime.timedelta(days=int(p[:-2]) * 30)
    if p.endswith("y"):   return now - datetime.timedelta(days=int(p[:-1]) * 365)
    if p == "max":        return now - datetime.timedelta(days=365 * 6)
    if p == "2y":         return now - datetime.timedelta(days=730)
    if p == "10y":        return now - datetime.timedelta(days=3650)
    return now - datetime.timedelta(days=60)


class MarketDataConnector:
    def __init__(self):
        self._alpaca_client = None
        if _USE_ALPACA:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._alpaca_client = StockHistoricalDataClient(
                    api_key=_ALPACA_KEY,
                    secret_key=_ALPACA_SECRET,
                )
                logger.info("MarketDataConnector: using Alpaca")
            except Exception as e:
                logger.warning(f"Alpaca init failed: {e} — falling back to yfinance")
        else:
            logger.info("MarketDataConnector: no Alpaca keys — using yfinance")

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        interval: str = "1h",
        period: str = "60d",
        source: str = "auto",
    ) -> pd.DataFrame:
        """
        interval: '1m' | '5m' | '15m' | '1h' | '4h' | '1d'
        period:   '5d' | '60d' | '1y' | '2y' | '10y' | 'max'
        """
        if self._alpaca_client is not None and source in ("auto", "alpaca"):
            df = self._alpaca_ohlcv(ticker, interval, period)
            if df is not None and not df.empty:
                return df
            logger.warning(f"{ticker}: Alpaca empty — falling back to yfinance")

        return self._yf_ohlcv(ticker, interval, period)

    # ── Alpaca ────────────────────────────────────────────────────────────────
    def _alpaca_ohlcv(self, ticker: str, interval: str, period: str) -> pd.DataFrame | None:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            tf_amount, tf_unit_str = _TF_MAP.get(interval, ("1", "Hour"))
            unit_map = {
                "Minute": TimeFrameUnit.Minute,
                "Hour":   TimeFrameUnit.Hour,
                "Day":    TimeFrameUnit.Day,
            }
            tf = TimeFrame(int(tf_amount), unit_map[tf_unit_str])

            start = _period_to_start(period)
            logger.info(f"Fetching {ticker} [{interval}] via Alpaca (from {start.date()})")

            from alpaca.data.enums import DataFeed
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                start=start,
                end=datetime.datetime.now(datetime.timezone.utc),
                adjustment="all",
                feed=DataFeed.IEX,   # free tier — SIP requires paid subscription
            )
            bars = self._alpaca_client.get_stock_bars(req)
            df = bars.df

            if df is None or df.empty:
                return None

            # Multi-index (symbol, timestamp) → flatten
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level="symbol")

            df.index = pd.to_datetime(df.index, utc=True)
            df.columns = [c.lower() for c in df.columns]

            # Keep only OHLCV
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep].copy()
            df.dropna(inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            logger.warning(f"{ticker}: Alpaca fetch error: {e}")
            return None

    # ── yfinance fallback ─────────────────────────────────────────────────────
    def _yf_ohlcv(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        import yfinance as yf
        _MAX_RETRIES = 3
        for attempt in range(_MAX_RETRIES):
            try:
                logger.info(f"Fetching {ticker} [{interval}] via yfinance")
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True)
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
            except Exception as e:
                logger.warning(f"{ticker}: yfinance attempt {attempt + 1}/{_MAX_RETRIES}: {e}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2.0 * (2 ** attempt) + random.uniform(0.5, 1.5))
        return pd.DataFrame()

    # ── Real-time quote ────────────────────────────────────────────────────────
    def get_quote(self, ticker: str) -> dict:
        if self._alpaca_client is not None:
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
                quote = self._alpaca_client.get_stock_latest_quote(req)[ticker]
                mid = (quote.ask_price + quote.bid_price) / 2
                return {
                    "ticker": ticker,
                    "price":  mid,
                    "bid":    quote.bid_price,
                    "ask":    quote.ask_price,
                    "volume": None,
                }
            except Exception as e:
                logger.warning(f"Alpaca quote {ticker}: {e}")

        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        return {
            "ticker": ticker,
            "price":  info.last_price,
            "bid":    getattr(info, "bid", None),
            "ask":    getattr(info, "ask", None),
            "volume": info.three_month_average_volume,
        }

    # ── Order book ────────────────────────────────────────────────────────────
    def get_order_book(self, ticker: str) -> dict:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        return {"bids": tk.bids, "asks": tk.asks}

    # ── Batch fetch ───────────────────────────────────────────────────────────
    def batch_ohlcv(
        self, tickers: list[str], interval: str = "1h", period: str = "60d"
    ) -> dict[str, pd.DataFrame]:
        result = {}
        for t in tickers:
            try:
                result[t] = self.get_ohlcv(t, interval, period)
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Failed {t}: {e}")
        return result
