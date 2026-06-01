"""
NLP sentiment via FinBERT (HuggingFace transformers).
Falls back to keyword scoring if transformers unavailable or TF is loaded.
FinBERT: ProsusAI/finbert — finance-specific BERT.

NOTE: PyTorch + TensorFlow in same process → segfault.
      Keyword fallback is always active when TF is loaded (default case).
      To run FinBERT: use a dedicated process without TensorFlow.
"""
from __future__ import annotations
import sys
from utils.logger import logger

FINBERT_MODEL = "ProsusAI/finbert"

_BULLISH = {
    "surge", "rally", "beat", "record", "growth", "profit", "strong",
    "bullish", "gain", "breakout", "upgrade", "buy", "outperform",
    "soar", "jump", "climb", "rise", "upside", "positive", "higher",
    "exceed", "guidance", "boost", "expansion", "momentum",
}
_BEARISH = {
    "drop", "fall", "miss", "loss", "weak", "bearish", "down", "decline",
    "downgrade", "sell", "underperform", "lower", "plunge", "crash",
    "slump", "tumble", "recession", "layoff", "cut", "warning",
    "shortfall", "disappointing", "negative", "risk",
}


class NLPSentimentAnalyzer:

    def __init__(self):
        self._pipe = None
        # Only attempt FinBERT when TF is absent — avoids TF+PyTorch segfault
        if "tensorflow" not in sys.modules:
            self._load()
        else:
            logger.info("NLP: keyword fallback active (TF+PyTorch conflict avoided)")

    def _load(self):
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "text-classification",
                model=FINBERT_MODEL,
                top_k=None,
                device=-1,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded")
        except ImportError:
            logger.info("NLP: transformers not installed — keyword fallback")
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e} — keyword fallback")

    # ── Public interface ──────────────────────────────────────────────────────
    def score(self, texts: list[str]) -> float:
        """Returns sentiment in [-1, +1]. Positive = bullish."""
        if not texts:
            return 0.0
        return self._bert_score(texts) if self._pipe else self._keyword_score(texts)

    # ── FinBERT scoring ───────────────────────────────────────────────────────
    def _bert_score(self, texts: list[str]) -> float:
        scores: list[float] = []
        for text in texts[:20]:
            try:
                result = self._pipe(text[:512])[0]
                lmap = {r["label"].lower(): r["score"] for r in result}
                scores.append(lmap.get("positive", 0) - lmap.get("negative", 0))
            except Exception:
                scores.append(0.0)
        return float(sum(scores) / len(scores)) if scores else 0.0

    # ── Keyword fallback ──────────────────────────────────────────────────────
    @staticmethod
    def _keyword_score(texts: list[str]) -> float:
        scores: list[float] = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & _BULLISH)
            neg = len(words & _BEARISH)
            if pos + neg > 0:
                scores.append((pos - neg) / (pos + neg))
        return float(sum(scores) / len(scores)) if scores else 0.0
