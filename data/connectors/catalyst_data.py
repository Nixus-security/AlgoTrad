"""
Catalyst data: Fear & Greed index + news spike volume.
Sources: alternative.me (Fear & Greed — free), NewsAPI.
"""
from __future__ import annotations
import os
import datetime
import requests
from dataclasses import dataclass
from utils.logger import logger

_FNG_URL = "https://api.alternative.me/fng/?limit=1"
_HEADERS = {"User-Agent": "AlgoTrad algotrad-bot@example.com"}

_TICKER_TO_NAME: dict[str, str] = {}  # map ticker → news search term (equity names added as needed)


@dataclass
class CatalystData:
    ticker: str
    earnings_days_away: int | None
    has_recent_earnings: bool
    analyst_sentiment: float
    recent_upgrades: int
    recent_downgrades: int
    sec_8k_score: float
    news_spike_score: float          # 0–1: news volume last 24h
    catalyst_score: float            # composite 0–1
    fear_greed_value: int            # 0 (extreme fear) – 100 (extreme greed)
    fear_greed_label: str            # e.g. "Fear", "Greed"


class CatalystConnector:

    def __init__(self):
        self.news_key = os.getenv("NEWS_API_KEY", "")
        self._fng_cache: dict = {"ts": 0.0, "value": 50, "label": "Neutral"}

    def get(self, ticker: str) -> CatalystData:
        fng_value, fng_label = self._fear_greed()
        news_score = self._news_spike(ticker)

        # Fear & Greed as catalyst signal:
        # Extreme fear (0-25) → potential reversal → mild positive catalyst
        # Extreme greed (75-100) → overbought risk → mild negative catalyst
        if fng_value <= 25:
            fng_catalyst = 0.4   # contrarian: fear = opportunity
        elif fng_value >= 75:
            fng_catalyst = 0.3   # greed = caution signal
        else:
            fng_catalyst = 0.1

        catalyst_score = float(min(0.5 * fng_catalyst + 0.5 * news_score, 1.0))

        return CatalystData(
            ticker=ticker,
            earnings_days_away=None,
            has_recent_earnings=False,
            analyst_sentiment=0.0,
            recent_upgrades=0,
            recent_downgrades=0,
            sec_8k_score=0.0,
            news_spike_score=news_score,
            catalyst_score=catalyst_score,
            fear_greed_value=fng_value,
            fear_greed_label=fng_label,
        )

    def _fear_greed(self) -> tuple[int, str]:
        """Fear & Greed index via alternative.me — cached 30 min."""
        import time
        if time.time() - self._fng_cache["ts"] < 1800:
            return self._fng_cache["value"], self._fng_cache["label"]
        try:
            resp = requests.get(_FNG_URL, headers=_HEADERS, timeout=8)
            resp.raise_for_status()
            data = resp.json()["data"][0]
            value = int(data["value"])
            label = data["value_classification"]
            self._fng_cache = {"ts": time.time(), "value": value, "label": label}
            logger.info(f"Fear & Greed: {value} ({label})")
            return value, label
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return self._fng_cache["value"], self._fng_cache["label"]

    def _news_spike(self, ticker: str) -> float:
        if not self.news_key:
            return 0.0
        name = _TICKER_TO_NAME.get(ticker, ticker.replace("-USD", ""))
        try:
            yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": name, "pageSize": 1, "from": yesterday,
                        "apiKey": self.news_key},
                timeout=8,
            )
            total = resp.json().get("totalResults", 0)
            return float(min(total / 100.0, 1.0))
        except Exception:
            return 0.0
