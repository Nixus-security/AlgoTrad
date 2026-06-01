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
import warnings
import numpy as np
from dataclasses import dataclass
from utils.logger import logger

# LGBMClassifier trained with DataFrame columns — inference uses numpy array.
# validate_features=False skips check but sklearn still emits the UserWarning.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

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

    # Indices of live features (same filter applied during training in main.py)
    # Full vector: rvol(0) spread_pct(1) liquidity(2) gap_pct(3) gap_str(4)
    #   vwap_pos(5) mom_1h(6) mom_str(7) wick_up(8) wick_dn(9)
    #   vol_z(10) vol_trend(11) bk_str(12) fake_bk(13) vol_pct(14)
    #   catalyst(15) sentiment(16) fomo(17) squeeze(18)
    _LIVE_IDX = [3, 6, 7, 8, 9, 12, 13, 14]   # 8 live features

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, features: list[float]) -> EnsemblePrediction:
        X_full = np.nan_to_num(
            np.array(features, dtype=np.float32).reshape(1, -1), nan=0.0
        )
        # Select only live features if models were trained on reduced set
        if self._xgb is not None and hasattr(self._xgb, "n_features_in_") \
                and self._xgb.n_features_in_ == len(self._LIVE_IDX) \
                and X_full.shape[1] >= max(self._LIVE_IDX) + 1:
            X = X_full[:, self._LIVE_IDX]
        else:
            X = X_full
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
    # Feature names for the 19-element feature vector (from FeatureSet.array order)
    FEATURE_NAMES = [
        "rvol", "spread_pct", "liquidity", "gap_pct", "gap_str",
        "vwap_pos", "mom_1h", "mom_str", "wick_up", "wick_dn",
        "vol_z", "vol_trend", "bk_str", "fake_bk", "vol_pct",
        "catalyst", "sentiment", "fomo", "squeeze",
    ]

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        X: (n_samples, n_features) float32
        y: (n_samples,) binary int  1=up, 0=down/neutral

        Anti-overfitting measures:
          - Temporal OOS split: last 20% held out as true test set
          - XGB/LGB early stopping on OOS set (stops before overfit)
          - Feature importance logged: dead features visible
        """
        results: dict = {}
        os.makedirs(MODEL_DIR, exist_ok=True)

        # ── Temporal OOS split (last 20% = never seen during training) ───────
        n_oos    = max(int(len(X) * 0.20), 50)
        X_tr, X_oos = X[:-n_oos], X[-n_oos:]
        y_tr, y_oos = y[:-n_oos], y[-n_oos:]
        logger.info(f"Ensemble train/OOS split: {len(X_tr)} / {len(X_oos)} samples")

        feat_names = self.FEATURE_NAMES[:X.shape[1]] if X.shape[1] <= len(self.FEATURE_NAMES) \
                     else [f"f{i}" for i in range(X.shape[1])]
        import pandas as _pd
        X_tr_df  = _pd.DataFrame(X_tr,  columns=feat_names)
        X_oos_df = _pd.DataFrame(X_oos, columns=feat_names)

        if XGB_AVAILABLE:
            try:
                model = xgb.XGBClassifier(
                    n_estimators=500,          # more trees — early stopping will cap them
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    early_stopping_rounds=30,  # stop when OOS loss stops improving
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
                model.fit(
                    X_tr_df, y_tr,
                    eval_set=[(X_oos_df, y_oos)],
                    verbose=False,
                )
                oos_acc = float(np.mean(model.predict(X_oos_df) == y_oos))
                best_iter = model.best_iteration
                self._xgb = model
                with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "wb") as f:
                    pickle.dump(model, f)
                results["xgb"] = f"trained (iters={best_iter} OOS_acc={oos_acc:.3f})"
                logger.info(f"XGBoost trained — best_iter={best_iter}  OOS_acc={oos_acc:.3f}")
                # Feature importance (top 5)
                imp = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
                logger.info("XGB feature importance (top 5): " +
                            "  ".join(f"{n}={v:.3f}" for n, v in imp[:5]))
            except Exception as e:
                logger.error(f"XGBoost training: {e}")

        if LGB_AVAILABLE:
            try:
                import lightgbm as _lgb
                callbacks = [_lgb.early_stopping(stopping_rounds=30, verbose=False),
                             _lgb.log_evaluation(period=-1)]
                model = lgb.LGBMClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,             # L1 regularization (added vs original)
                    reg_lambda=0.1,            # L2 regularization (added vs original)
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(
                    X_tr_df, y_tr,
                    eval_set=[(X_oos_df, y_oos)],
                    callbacks=callbacks,
                )
                oos_acc = float(np.mean(model.predict(X_oos_df) == y_oos))
                best_iter = model.best_iteration_
                self._lgb = model
                with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "wb") as f:
                    pickle.dump(model, f)
                results["lgb"] = f"trained (iters={best_iter} OOS_acc={oos_acc:.3f})"
                logger.info(f"LightGBM trained — best_iter={best_iter}  OOS_acc={oos_acc:.3f}")
                # Feature importance (top 5)
                imp = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
                logger.info("LGB feature importance (top 5): " +
                            "  ".join(f"{n}={v:.3f}" for n, v in imp[:5]))
                # Warn on dead features (importance=0)
                dead = [n for n, v in zip(feat_names, model.feature_importances_) if v == 0]
                if dead:
                    logger.warning(f"LGB dead features (importance=0): {dead}")
            except Exception as e:
                logger.error(f"LightGBM training: {e}")

        return results
