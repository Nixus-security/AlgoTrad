"""
Gradient boosting ensemble: XGBoost + LightGBM.
Inputs:  feature vector from FeatureEngine (19 features).
Outputs: prob_up_1h, prob_down_1h, volatility_expected, confidence,
         fake_breakout_prob, squeeze_prob.
Falls back to heuristic when no trained model exists.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
from dataclasses import dataclass
from utils.logger import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed — pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("lightgbm not installed — pip install lightgbm")


@dataclass
class EnsemblePrediction:
    prob_up_1h: float
    prob_down_1h: float
    prob_neutral: float
    volatility_expected: float
    confidence: float
    fake_breakout_prob: float
    squeeze_prob: float


class EnsemblePredictor:

    def __init__(self):
        self._xgb: object | None = None
        self._lgb: object | None = None
        self._load()

    def _load(self):
        for fname, attr in [("xgb_model.pkl", "_xgb"), ("lgb_model.pkl", "_lgb")]:
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        setattr(self, attr, pickle.load(f))
                    logger.info(f"Loaded {fname}")
                except Exception as e:
                    logger.warning(f"Could not load {fname}: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, features: list[float]) -> EnsemblePrediction:
        X = np.nan_to_num(
            np.array(features, dtype=np.float32).reshape(1, -1), nan=0.0
        )
        probs: list[float] = []

        if self._xgb is not None and XGB_AVAILABLE:
            try:
                probs.append(float(self._xgb.predict_proba(X)[0][1]))
            except Exception as e:
                logger.debug(f"XGB predict: {e}")

        if self._lgb is not None and LGB_AVAILABLE:
            try:
                # validate_features=False : évite warning si modèle entraîné
                # avec DataFrame mais predict reçoit numpy array
                probs.append(float(
                    self._lgb.predict_proba(X, validate_features=False)[0][1]
                ))
            except Exception as e:
                logger.debug(f"LGB predict: {e}")

        if not probs:
            return self._heuristic(features)

        prob_up = float(np.mean(probs))
        deviation = abs(prob_up - 0.5)
        return EnsemblePrediction(
            prob_up_1h=prob_up,
            prob_down_1h=1.0 - prob_up,
            prob_neutral=max(0.0, 1.0 - deviation * 2),
            volatility_expected=deviation * 0.04,
            confidence=0.5 + deviation,
            fake_breakout_prob=self._fake_prob_from_features(features),
            squeeze_prob=self._squeeze_prob_from_features(features),
        )

    # ── Feature-based fallback ────────────────────────────────────────────────
    @staticmethod
    def _heuristic(features: list[float]) -> EnsemblePrediction:
        # features order from FeatureSet.array:
        # [rvol, spread_pct, liquidity, gap_pct, gap_str, vwap_pos,
        #  mom_1h, mom_str, wick_up, wick_dn, vol_z, vol_trend,
        #  bk_str, fake_bk, vol_pct, catalyst, sentiment, fomo, squeeze]
        def _f(i: int) -> float:
            return float(features[i]) if i < len(features) else 0.0

        rvol = _f(0)
        gap_pct = _f(3)
        vwap_pos = _f(5)
        mom = _f(6)
        bk_str = _f(12)
        sentiment = _f(16)
        fomo = _f(17)
        squeeze = _f(18)

        bull = (
            min(rvol / 2.0, 1.0) * 0.20
            + (1.0 if gap_pct > 0 else 0.0) * 0.10
            + vwap_pos * 0.20
            + (1.0 if mom > 0 else 0.0) * 0.15
            + bk_str * 0.10
            + ((sentiment + 1) / 2) * 0.15
            + fomo * 0.10
        )
        prob_up = float(np.clip(bull, 0.1, 0.9))
        deviation = abs(prob_up - 0.5)
        return EnsemblePrediction(
            prob_up_1h=prob_up,
            prob_down_1h=1.0 - prob_up,
            prob_neutral=max(0.0, 1.0 - deviation * 2),
            volatility_expected=deviation * 0.04,
            confidence=0.5 + deviation,
            fake_breakout_prob=_f(13),
            squeeze_prob=squeeze,
        )

    @staticmethod
    def _fake_prob_from_features(features: list[float]) -> float:
        return float(features[13]) if len(features) > 13 else 0.3

    @staticmethod
    def _squeeze_prob_from_features(features: list[float]) -> float:
        return float(features[18]) if len(features) > 18 else 0.2

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        X: (n_samples, n_features) float32
        y: (n_samples,) binary int  1=up, 0=down/neutral
        """
        results: dict[str, str] = {}
        os.makedirs(MODEL_DIR, exist_ok=True)

        if XGB_AVAILABLE:
            try:
                model = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42, n_jobs=-1,
                )
                model.fit(X, y)
                self._xgb = model
                with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "wb") as f:
                    pickle.dump(model, f)
                results["xgb"] = "trained"
                logger.info("XGBoost trained and saved")
            except Exception as e:
                logger.error(f"XGBoost training: {e}")

        if LGB_AVAILABLE:
            try:
                model = lgb.LGBMClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbose=-1,
                )
                model.fit(X, y)
                self._lgb = model
                with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "wb") as f:
                    pickle.dump(model, f)
                results["lgb"] = "trained"
                logger.info("LightGBM trained and saved")
            except Exception as e:
                logger.error(f"LightGBM training: {e}")

        return results
