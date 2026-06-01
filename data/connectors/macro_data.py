"""
Macro / fundamental data connector.
Sources: FRED (rates, CPI), yfinance (earnings), NewsAPI (sentiment).
"""
from __future__ import annotations
import os
import time
import datetime
import requests
import pandas as pd
from utils.logger import logger

FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
_FRED_TTL    = 3600   # cache FRED responses 1 hour — macro data is daily


class MacroDataConnector:
    _fred_key_warned = False
    _snapshot_cache: dict = {"ts": 0.0, "data": {}}   # class-level shared cache

    def __init__(self):
        self.fred_key = os.getenv("FRED_API_KEY", "")
        self.news_key = os.getenv("NEWS_API_KEY", "")

    # ── FRED time-series ───────────────────────────────────────────────────────
    def get_fred_series(self, series_id: str, limit: int = 252) -> pd.Series:
        """
        Common series:
          DFF    — Fed Funds Rate
          T10Y2Y — 10Y-2Y yield spread (recession proxy)
          CPIAUCSL — CPI
          UNRATE — Unemployment
        """
        if not self.fred_key:
            if not MacroDataConnector._fred_key_warned:
                logger.info("No FRED_API_KEY — macro features disabled (optionnel)")
                MacroDataConnector._fred_key_warned = True
            return pd.Series(dtype=float)
        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        s = pd.Series(
            {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
            name=series_id,
        )
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    # ── Macro context snapshot ─────────────────────────────────────────────────
    def get_macro_snapshot(self) -> dict:
        """Returns dict of current key macro indicators. Cached 1 hour."""
        if time.time() - MacroDataConnector._snapshot_cache["ts"] < _FRED_TTL:
            return MacroDataConnector._snapshot_cache["data"]

        snap = {}
        series = {"fed_rate": "DFF", "yield_spread": "T10Y2Y", "cpi": "CPIAUCSL"}
        for name, sid in series.items():
            try:
                s = self.get_fred_series(sid, limit=5)
                snap[name] = float(s.iloc[-1]) if not s.empty else None
            except Exception as e:
                logger.error(f"FRED {sid}: {e}")
                snap[name] = None

        # Only cache if at least one value fetched successfully
        if any(v is not None for v in snap.values()):
            MacroDataConnector._snapshot_cache = {"ts": time.time(), "data": snap}
            logger.debug(f"FRED macro snapshot refreshed: {snap}")
        else:
            logger.debug("FRED snapshot all None — not cached, will retry next cycle")
        return snap

    # ── News sentiment score ──────────────────────────────────────────────────
    def get_news_sentiment(self, query: str, days_back: int = 3) -> float:
        """
        Returns sentiment score in [-1, +1].
        Crude polarity via keyword counting (no extra NLP lib dependency).
        Upgrade to transformers pipeline when GPU available.
        """
        if not self.news_key:
            return 0.0
        from_dt = (datetime.date.today() - datetime.timedelta(days=days_back)).isoformat()
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query, "from": from_dt, "sortBy": "relevancy",
            "pageSize": 20, "apiKey": self.news_key,
        }
        try:
            articles = requests.get(url, params=params, timeout=10).json().get("articles", [])
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return 0.0

        pos = {"growth", "beat", "profit", "surge", "gain", "strong", "bullish", "up"}
        neg = {"loss", "decline", "miss", "weak", "bearish", "down", "recession", "fear"}
        scores = []
        for art in articles:
            text = (art.get("title", "") + " " + art.get("description", "")).lower()
            words = set(text.split())
            p = len(words & pos)
            n = len(words & neg)
            if p + n > 0:
                scores.append((p - n) / (p + n))
        return float(sum(scores) / len(scores)) if scores else 0.0
