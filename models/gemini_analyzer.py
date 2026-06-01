"""
Gemini Flash NLP analyzer — sentiment structuré + détection de catalyst pour small caps.
Falls back automatiquement sur keyword scoring si GEMINI_API_KEY non défini.

Sources analysées:
  - Google News RSS (gratuit, sans auth)
  - StockTwits headlines
  - Reddit r/wallstreetbets + r/pennystocks
  - Textes additionnels passés à analyze()

Catalysts détectés:
  Bullish: earnings_beat, fda_approval, partnership, buyout, upgrade, short_squeeze
  Bearish: earnings_miss, fda_rejection, dilution, downgrade
"""
from __future__ import annotations
import os
import json
import time
import requests
from dataclasses import dataclass
from utils.logger import logger

# API REST Gemini — zéro dépendance extra, pas de conflit protobuf
_GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={key}"
)

# Catalysts qui boostent / réduisent la confiance du signal
BULLISH_CATALYSTS = {
    "earnings_beat", "fda_approval", "partnership",
    "buyout", "upgrade", "short_squeeze",
}
BEARISH_CATALYSTS = {
    "earnings_miss", "fda_rejection", "dilution", "downgrade",
}

_BULLISH_KW = {
    "surge", "rally", "beat", "record", "growth", "profit", "approval",
    "bullish", "gain", "breakout", "upgrade", "buy", "outperform",
    "soar", "jump", "squeeze", "partner", "acquire", "buyout", "fda",
    "clearance", "approved", "milestone", "contract", "deal",
}
_BEARISH_KW = {
    "drop", "fall", "miss", "loss", "weak", "bearish", "decline",
    "downgrade", "sell", "plunge", "crash", "dilut", "offering",
    "warning", "layoff", "cut", "disappoint", "reject", "failure",
    "lawsuit", "fraud", "investigation", "sec", "subpoena",
}

_GEMINI_PROMPT = """\
Analyze these financial news/social media texts about stock ticker {ticker}.
Be concise and objective. Return ONLY valid JSON, no markdown, no explanation.

{{
  "sentiment_score": <float, -1.0=very bearish to +1.0=very bullish>,
  "catalyst_type": <one of: "earnings_beat","earnings_miss","fda_approval","fda_rejection","partnership","dilution","buyout","upgrade","downgrade","short_squeeze","pump","general_news","none">,
  "confidence": <float 0.0 to 1.0>,
  "summary": "<max 20 words describing the key catalyst>"
}}

Texts about {ticker}:
{texts}"""


@dataclass
class GeminiResult:
    ticker: str
    sentiment_score: float    # -1 (très bearish) → +1 (très bullish)
    catalyst_type: str        # ex: "fda_approval"
    confidence: float         # 0-1
    summary: str
    source_count: int
    is_gemini: bool           # True = Gemini API, False = keyword fallback


class GeminiAnalyzer:

    def __init__(self, cfg: dict):
        self._api_key      = os.getenv("GEMINI_API_KEY", "")
        gcfg               = cfg.get("gemini", {})
        self._model_name   = gcfg.get("model", "gemini-1.5-flash")
        self._enabled      = gcfg.get("enabled", True)
        self._cache_sec    = gcfg.get("cache_minutes", 30) * 60
        self._max_texts    = gcfg.get("max_texts_per_ticker", 12)
        self._cache: dict[str, tuple[float, GeminiResult]] = {}
        self._model        = None

        if self._api_key and self._enabled:
            self._load()
        else:
            logger.info("GeminiAnalyzer: pas de GEMINI_API_KEY — keyword fallback actif")

    # ── Vérifie clé via endpoint /models (zéro conflit protobuf) ────────────
    def _load(self):
        try:
            # /models?key=... retourne 200 si clé valide, 400/403 sinon
            test = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models"
                f"?key={self._api_key}",
                timeout=8,
            )
            if test.status_code == 200:
                # Vérifie que le modèle configuré existe, sinon prend premier disponible
                available = [
                    m["name"].split("/")[-1]
                    for m in test.json().get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                if self._model_name in available:
                    chosen = self._model_name
                elif available:
                    # Préfère gemini-2.0-flash ou gemini-1.5-flash si disponible
                    for pref in ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"):
                        if pref in available:
                            chosen = pref
                            break
                    else:
                        chosen = available[0]
                    logger.info(f"GeminiAnalyzer: modèle {self._model_name} non dispo → {chosen}")
                    self._model_name = chosen
                else:
                    logger.warning("GeminiAnalyzer: aucun modèle generateContent disponible")
                    return
                self._model = True
                logger.info(f"GeminiAnalyzer: {self._model_name} prêt (REST API)")
            else:
                logger.warning(
                    f"GeminiAnalyzer: clé API invalide ou quota dépassé "
                    f"(HTTP {test.status_code}) — keyword fallback"
                )
        except Exception as e:
            logger.warning(f"GeminiAnalyzer: connexion failed: {e}")

    # ── Public ────────────────────────────────────────────────────────────────
    def analyze(self, ticker: str, texts: list[str]) -> GeminiResult:
        """Analyse une liste de titres/posts. Résultat mis en cache 30 min."""
        cached = self._cache.get(ticker)
        if cached and time.time() - cached[0] < self._cache_sec:
            return cached[1]

        if not texts:
            result = GeminiResult(
                ticker=ticker, sentiment_score=0.0, catalyst_type="none",
                confidence=0.2, summary="Aucune news trouvée",
                source_count=0, is_gemini=False,
            )
        elif self._model:
            result = self._gemini_analyze(ticker, texts)
        else:
            result = self._keyword_analyze(ticker, texts)

        self._cache[ticker] = (time.time(), result)
        logger.info(
            f"GeminiAnalyzer {ticker}: score={result.sentiment_score:+.2f} "
            f"catalyst={result.catalyst_type} conf={result.confidence:.0%} "
            f"({'Gemini' if result.is_gemini else 'keywords'}) "
            f"sources={result.source_count}"
        )
        return result

    def catalyst_confidence_boost(self, result: GeminiResult) -> float:
        """Multiplicateur de confiance selon catalyst. 1.0 = pas de changement."""
        if result.catalyst_type in BULLISH_CATALYSTS and result.confidence > 0.6:
            return 1.15    # +15% pour catalyst haussier fort
        if result.catalyst_type in BEARISH_CATALYSTS and result.confidence > 0.6:
            return 0.85    # -15% pour risque baissier fort
        return 1.0

    # ── Gemini REST API (zéro SDK, zéro conflit protobuf) ────────────────────
    def _gemini_analyze(self, ticker: str, texts: list[str]) -> GeminiResult:
        combined = "\n---\n".join(texts[: self._max_texts])[:3500]
        prompt   = _GEMINI_PROMPT.format(ticker=ticker, texts=combined)

        url     = _GEMINI_REST_URL.format(model=self._model_name, key=self._api_key)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            raw  = (
                resp.json()
                ["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
            )
            # Nettoie markdown si Gemini l'ajoute
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)

            return GeminiResult(
                ticker=ticker,
                sentiment_score=float(max(-1.0, min(1.0,
                    data.get("sentiment_score", 0.0)))),
                catalyst_type=str(data.get("catalyst_type", "none")),
                confidence=float(max(0.0, min(1.0,
                    data.get("confidence", 0.5)))),
                summary=str(data.get("summary", ""))[:200],
                source_count=len(texts),
                is_gemini=True,
            )

        except Exception as e:
            logger.debug(f"GeminiAnalyzer {ticker} REST error: {e} → keyword fallback")
            return self._keyword_analyze(ticker, texts)

    # ── Keyword fallback (zéro API) ───────────────────────────────────────────
    def _keyword_analyze(self, ticker: str, texts: list[str]) -> GeminiResult:
        combined = " ".join(texts).lower()
        words    = set(combined.split())

        pos = len(words & _BULLISH_KW)
        neg = len(words & _BEARISH_KW)
        total = pos + neg or 1
        score = round((pos - neg) / total, 3)

        # Détection catalyst par mots-clés
        catalyst = "none"
        if "fda" in combined and any(w in combined for w in ("approv", "clear", "grant")):
            catalyst = "fda_approval"
        elif "fda" in combined and any(w in combined for w in ("reject", "fail", "refuse")):
            catalyst = "fda_rejection"
        elif any(w in combined for w in ("dilut", "offering", "share issu")):
            catalyst = "dilution"
        elif any(w in combined for w in ("partner", "agreement", "collaborat", "deal")):
            catalyst = "partnership"
        elif any(w in combined for w in ("squeeze", "short interest", "gamma")):
            catalyst = "short_squeeze"
        elif any(w in combined for w in ("beat", "exceed", "surpass", "record earn")):
            catalyst = "earnings_beat"
        elif any(w in combined for w in ("miss", "disappoint", "below expect")):
            catalyst = "earnings_miss"
        elif any(w in combined for w in ("acqui", "buyout", "takeover", "merger")):
            catalyst = "buyout"
        elif pos + neg >= 3:
            catalyst = "general_news"

        return GeminiResult(
            ticker=ticker,
            sentiment_score=score,
            catalyst_type=catalyst,
            confidence=min(0.65, 0.25 + (pos + neg) * 0.05),
            summary=f"Keywords: {pos} bullish, {neg} bearish ({len(texts)} sources)",
            source_count=len(texts),
            is_gemini=False,
        )
