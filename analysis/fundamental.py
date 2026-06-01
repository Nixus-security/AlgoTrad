"""
Fundamental / macro signal module.
Combines macro snapshot + news sentiment into a direction score.
"""
from __future__ import annotations
from dataclasses import dataclass
from data.connectors.macro_data import MacroDataConnector
from utils.logger import logger


@dataclass
class FundamentalSignal:
    direction: str
    strength: float
    sentiment_score: float   # News polarity [-1, +1]
    macro_score: float       # Macro health [-1, +1]
    details: dict


class FundamentalAnalyzer:
    def __init__(self):
        self.macro = MacroDataConnector()

    def get_signal(self, ticker: str) -> FundamentalSignal:
        snap = self.macro.get_macro_snapshot()
        sentiment = self.macro.get_news_sentiment(ticker)

        macro_score = self._macro_score(snap)
        combined = 0.6 * macro_score + 0.4 * sentiment

        if combined > 0.2:
            direction, strength = "BUY", min(combined, 1.0)
        elif combined < -0.2:
            direction, strength = "SELL", min(abs(combined), 1.0)
        else:
            direction, strength = "NEUTRAL", 0.5

        return FundamentalSignal(
            direction=direction,
            strength=strength,
            sentiment_score=sentiment,
            macro_score=macro_score,
            details=snap,
        )

    # ── Macro health heuristic ────────────────────────────────────────────────
    @staticmethod
    def _macro_score(snap: dict) -> float:
        score = 0.0
        fed_rate = snap.get("fed_rate")
        spread = snap.get("yield_spread")

        # Rising rates → bearish for risk assets (equities + crypto)
        if fed_rate is not None:
            if fed_rate < 3.0:
                score += 0.3
            elif fed_rate > 5.5:
                score -= 0.3

        # Inverted yield curve → recession risk → bearish
        if spread is not None:
            if spread > 0:
                score += 0.3
            else:
                score -= 0.4  # Inversion → strong bearish signal

        return max(-1.0, min(score, 1.0))
