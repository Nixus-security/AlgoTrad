"""
Market psychology connector.
Sources:
  - StockTwits  (free, pas d'auth — sentiment finance-specific)
  - Google News RSS (gratuit, sans auth)
  - Reddit r/wallstreetbets + r/pennystocks + r/stocks (gratuit, JSON public)
Computed: FOMO score, panic score, squeeze probability, euphoria, fear,
          crowd acceleration.
"""
from __future__ import annotations
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
from utils.logger import logger

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
_HEADERS        = {"User-Agent": "AlgoTrad/1.0 (research)"}


@dataclass
class SentimentData:
    ticker: str
    bullish_pct: float           # 0–1 fraction of bullish StockTwits messages
    message_volume: int          # number of recent messages fetched
    fomo_score: float            # 0–1: price acceleration + volume spike
    panic_score: float           # 0–1: sharp drop + high volume
    squeeze_probability: float   # 0–1: short float % + RVOL
    euphoria_score: float        # 0–1: overbought + high bull sentiment
    fear_score: float            # 0–1: oversold + bear sentiment
    crowd_acceleration: float    # 0–1: how fast crowd sentiment is moving


class SentimentConnector:

    # ── Google News RSS ───────────────────────────────────────────────────────
    def _google_news_rss(self, ticker: str) -> list[str]:
        """Titres de news Google News RSS — gratuit, zéro auth."""
        try:
            q   = urllib.parse.quote(f"{ticker} stock")
            url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = resp.read()
            tree   = ET.fromstring(raw)
            titles = [
                item.findtext("title", "")
                for item in tree.findall(".//item")
            ]
            result = [t for t in titles if t][:15]
            logger.debug(f"Google News {ticker}: {len(result)} titres")
            return result
        except Exception as e:
            logger.debug(f"Google News {ticker}: {e}")
            return []

    # ── Reddit ────────────────────────────────────────────────────────────────
    def _reddit(self, ticker: str) -> list[str]:
        """Posts Reddit (WSB + pennystocks + stocks) — API publique JSON."""
        try:
            subs = "wallstreetbets+pennystocks+stocks+smallcapstocks"
            url  = (
                f"https://www.reddit.com/r/{subs}/search.json"
                f"?q={urllib.parse.quote(ticker)}&sort=new&limit=15&t=day"
            )
            resp = requests.get(url, headers=_HEADERS, timeout=8)
            if resp.status_code != 200:
                return []
            posts  = resp.json().get("data", {}).get("children", [])
            titles = [
                p["data"]["title"]
                for p in posts
                if p.get("data", {}).get("title")
            ]
            logger.debug(f"Reddit {ticker}: {len(titles)} posts")
            return titles[:12]
        except Exception as e:
            logger.debug(f"Reddit {ticker}: {e}")
            return []

    # ── Agrégation de textes pour Gemini ─────────────────────────────────────
    def get_news_texts(self, ticker: str) -> list[str]:
        """Retourne tous les textes (news + reddit) pour analyse Gemini."""
        news   = self._google_news_rss(ticker)
        reddit = self._reddit(ticker)
        return news + reddit

    # ── StockTwits ────────────────────────────────────────────────────────────
    def _stocktwits(self, ticker: str) -> tuple[float, int]:
        try:
            # Strip yfinance suffixes (=X for forex, -B etc.)
            clean = ticker.replace("=X", "").replace("-", ".")
            resp = requests.get(
                f"{STOCKTWITS_BASE}/streams/symbol/{clean}.json",
                timeout=8,
            )
            if resp.status_code != 200:
                return 0.5, 0
            messages = resp.json().get("messages", [])
            if not messages:
                return 0.5, 0
            bulls = sum(
                1 for m in messages
                if (m.get("entities") or {}).get("sentiment", {})
                and m["entities"]["sentiment"].get("basic") == "Bullish"
            )
            bears = sum(
                1 for m in messages
                if (m.get("entities") or {}).get("sentiment", {})
                and m["entities"]["sentiment"].get("basic") == "Bearish"
            )
            total = bulls + bears or 1
            return float(bulls / total), len(messages)
        except Exception as e:
            logger.debug(f"StockTwits {ticker}: {e}")
            return 0.5, 0

    # ── Main entry ────────────────────────────────────────────────────────────
    def compute(
        self,
        ticker: str,
        df: pd.DataFrame,
        micro=None,   # MicrostructureData
    ) -> SentimentData:
        bull_pct, msg_vol = self._stocktwits(ticker)

        # FOMO: recent price acceleration + volume spike
        if len(df) >= 5:
            ret = df["close"].pct_change().dropna().tail(5)
            has_vol = "volume" in df.columns and not df["volume"].isna().all()
            vol_tail = df["volume"].replace(0, np.nan).tail(5) if has_vol else pd.Series([1.0] * 5)
            price_accel = float(ret.mean() / max(ret.std(), 1e-9))
            vol_mean = vol_tail.mean()
            vol_spike = float(vol_tail.iloc[-1] / vol_mean) if vol_mean and vol_mean > 0 else 1.0
            fomo = float(np.clip(price_accel * 0.5 + (vol_spike - 1) * 0.5, 0, 1))
            panic = float(np.clip(-price_accel * 0.5 + (vol_spike - 1) * 0.3, 0, 1))
        else:
            fomo = panic = 0.0

        # Squeeze: short interest + high RVOL
        short_pct = getattr(micro, "short_interest_pct", 0.0) or 0.0
        rvol = getattr(micro, "rvol", 1.0) or 1.0
        squeeze = float(np.clip(
            (short_pct / 0.30) * 0.6 + (min(rvol, 5) / 5.0) * 0.4,
            0, 1,
        ))

        # Price rank in 20-bar window
        if len(df) >= 20:
            lo = df["close"].tail(20).min()
            hi = df["close"].tail(20).max()
            price_rank = float((df["close"].iloc[-1] - lo) / max(hi - lo, 1e-9))
        else:
            price_rank = 0.5

        euphoria = float(bull_pct * 0.5 + price_rank * 0.5)
        fear = float((1 - bull_pct) * 0.5 + (1 - price_rank) * 0.5)
        crowd_accel = float(abs(bull_pct - 0.5) * 2)

        return SentimentData(
            ticker=ticker,
            bullish_pct=bull_pct,
            message_volume=msg_vol,
            fomo_score=fomo,
            panic_score=panic,
            squeeze_probability=squeeze,
            euphoria_score=euphoria,
            fear_score=fear,
            crowd_acceleration=crowd_accel,
        )
