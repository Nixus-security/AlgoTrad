"""
Crypto watcher — fixed BTC-USD and SOL-USD symbols.
No dynamic scanning: crypto markets are 24/7 and always liquid.
is_market_open() always returns True.
"""
from __future__ import annotations
from utils.logger import logger

CRYPTO_SYMBOLS = ["BTC-USD", "SOL-USD"]


class CryptoWatcher:

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def get_candidates(self) -> list[str]:
        logger.info(f"CryptoWatcher: symbols → {CRYPTO_SYMBOLS}")
        return list(CRYPTO_SYMBOLS)

    def is_market_open(self) -> bool:
        return True  # crypto is 24/7


# Alias for backward compatibility if anything imports SmallCapScanner
SmallCapScanner = CryptoWatcher
