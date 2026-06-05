"""
Market watcher — Gold (GC=F), Nasdaq-100 (QQQ), S&P 500 (SPY), Nasdaq Futures (NQ=F).
is_market_open() checks NYSE/CME overlap window (Mon-Fri, 14-20 UTC).
"""
from __future__ import annotations
from datetime import datetime, timezone
from utils.logger import logger

SYMBOLS = ["QQQ", "SPY", "NQ=F"]

# NYSE/CME active window: 14-20 UTC covers both Gold NY session and QQQ NYSE hours
_MARKET_OPEN_UTC_START  = 14
_MARKET_OPEN_UTC_END    = 20
_MARKET_OPEN_WEEKDAYS   = {0, 1, 2, 3, 4}   # Mon–Fri


class MarketWatcher:

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def get_candidates(self) -> list[str]:
        logger.info(f"MarketWatcher: symbols → {SYMBOLS}")
        return list(SYMBOLS)

    def is_market_open(self) -> bool:
        now = datetime.now(timezone.utc)
        if now.weekday() not in _MARKET_OPEN_WEEKDAYS:
            return False
        return _MARKET_OPEN_UTC_START <= now.hour < _MARKET_OPEN_UTC_END


# Backward compatibility
CryptoWatcher  = MarketWatcher
SmallCapScanner = MarketWatcher
