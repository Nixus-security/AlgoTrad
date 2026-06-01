"""
Signal fusion engine — combines TA + ML(LSTM) + Ensemble(XGB/LGB) +
Statistical + Fundamental + NLP sentiment into a final TradeSignal.
Quantum optimizer supplies dynamic weights.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from analysis.technical import TechnicalSignal
from analysis.statistical import StatisticalSignal
from analysis.fundamental import FundamentalSignal
from utils.logger import logger


@dataclass
class TradeSignal:
    ticker: str
    direction: str           # "BUY" | "SELL"
    confidence: float        # 0–1
    price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    strategy: str
    sharpe_estimate: float
    drawdown_estimate: float
    quantum_up_prob: float
    # New fields
    fake_breakout_prob: float = 0.0
    squeeze_prob: float = 0.0
    volatility_expected: float = 0.0
    position_size_pct: float = 0.0   # suggested % of capital
    composite_score: float = 0.0
    breakdown: dict = field(default_factory=dict)


DIRECTION_MAP = {"BUY": 1.0, "NEUTRAL": 0.5, "SELL": 0.0}


class SignalFusion:

    def __init__(self, cfg: dict, quantum_optimizer):
        self.cfg = cfg
        self.qopt = quantum_optimizer
        self.weights = cfg["signal_weights"]

    # ── Main fusion entry ─────────────────────────────────────────────────────
    def fuse(
        self,
        ticker: str,
        ta_signal: TechnicalSignal,
        ml_direction: str,
        ml_confidence: float,
        stat_signal: StatisticalSignal,
        fund_signal: FundamentalSignal,
        price: float,
        timeframe: str,
        strategy: str,
        historical_sharpe: float = 0.0,
        # New inputs
        ensemble_pred=None,   # EnsemblePrediction | None
        nlp_score: float = 0.0,
        features=None,        # FeatureSet | None
    ) -> TradeSignal | None:

        # ── Per-module scores in [0, 1] ───────────────────────────────────────
        scores = {
            "technical": DIRECTION_MAP[ta_signal.direction] * ta_signal.strength,
            "ml_prediction": DIRECTION_MAP[ml_direction] * ml_confidence,
            "statistical": DIRECTION_MAP[stat_signal.direction] * stat_signal.strength,
            "fundamental": DIRECTION_MAP[fund_signal.direction] * fund_signal.strength,
        }

        # Ensemble model contributes as extra score
        ensemble_score = 0.5
        if ensemble_pred is not None:
            ensemble_score = float(ensemble_pred.prob_up_1h)
            scores["ensemble"] = ensemble_score

        # NLP sentiment blended in
        nlp_mapped = float(np.clip((nlp_score + 1) / 2, 0, 1))  # [-1,1] → [0,1]
        if abs(nlp_score) > 0.1:
            scores["nlp"] = nlp_mapped

        # Dynamic weight optimisation via QAOA
        sharpes = {k: max(0.1, v) for k, v in scores.items()}
        try:
            self.weights = self.qopt.optimise_weights(scores, sharpes)
        except Exception as e:
            logger.warning(f"Weight optimisation failed: {e} — defaults")

        # Weighted final score (use only the 4 base modules for weighting
        # unless the weight dict has ensemble/nlp keys)
        base_keys = [k for k in scores if k in self.weights]
        if not base_keys:
            base_keys = list(scores.keys())
        total_w = sum(self.weights.get(k, 1 / len(base_keys)) for k in base_keys) or 1.0
        final_score = sum(
            scores[k] * self.weights.get(k, 1 / len(base_keys)) / total_w
            for k in base_keys
        )

        # Quantum walk probability (10% blend)
        qwalk = self.qopt.quantum_random_walk(n_steps=8)
        q_up_prob = qwalk.get("up", 0.5)
        final_score = 0.9 * final_score + 0.1 * q_up_prob

        # Direction decision
        if final_score >= 0.60:
            direction = "BUY"
            confidence = float(final_score)
        elif final_score <= 0.40:
            direction = "SELL"
            confidence = float(1.0 - final_score)
        else:
            logger.info(f"{ticker}: score={final_score:.3f} → NEUTRAL, no signal")
            return None

        # Stop / target
        atr = ta_signal.atr if ta_signal.atr else price * 0.01
        atr_mult = self.cfg["risk"]["atr_stop_multiplier"]
        if direction == "BUY":
            stop = price - atr * atr_mult
            target = price + atr * atr_mult * 2
        else:
            stop = price + atr * atr_mult
            target = price - atr * atr_mult * 2

        # Sharpe: use actual returns when provided, else proxy
        if historical_sharpe != 0.0:
            sharpe_est = historical_sharpe
        else:
            sharpe_est = float(np.mean(list(sharpes.values())) * 2.0)

        dd_est = (1.0 - confidence) * 0.05

        # Pull extra outputs from ensemble
        fake_bk = 0.0
        squeeze = 0.0
        vol_exp = 0.0
        pos_size = 0.0
        if ensemble_pred is not None:
            fake_bk = float(ensemble_pred.fake_breakout_prob)
            squeeze = float(ensemble_pred.squeeze_prob)
            vol_exp = float(ensemble_pred.volatility_expected)
        if features is not None:
            fake_bk = max(fake_bk, float(features.fake_breakout_prob))
            squeeze = max(squeeze, float(features.squeeze_prob))

        # Quantum position sizing
        expected_ret = abs(final_score - 0.5) * 0.04
        pos_size = self.qopt.size_position(
            expected_return=expected_ret,
            volatility=max(vol_exp, 0.005),
            max_position=self.cfg["risk"].get("max_position_pct", 0.02),
        )

        return TradeSignal(
            ticker=ticker,
            direction=direction,
            confidence=round(confidence, 4),
            price=price,
            stop_loss=round(stop, 4),
            take_profit=round(target, 4),
            timeframe=timeframe,
            strategy=strategy,
            sharpe_estimate=round(sharpe_est, 2),
            drawdown_estimate=round(dd_est, 4),
            quantum_up_prob=round(q_up_prob, 3),
            fake_breakout_prob=round(fake_bk, 3),
            squeeze_prob=round(squeeze, 3),
            volatility_expected=round(vol_exp, 4),
            position_size_pct=round(pos_size, 4),
            composite_score=round(final_score, 4),
            breakdown={
                "ta_dir": ta_signal.direction,
                "ml_dir": ml_direction,
                "stat_dir": stat_signal.direction,
                "fund_dir": fund_signal.direction,
                "ensemble_prob_up": round(ensemble_score, 3),
                "nlp_score": round(nlp_score, 3),
                "weights": self.weights,
                "final_score": round(final_score, 4),
                "rsi": round(ta_signal.rsi, 1),
                "z_score": round(stat_signal.z_score, 2),
                "sentiment": round(fund_signal.sentiment_score, 2),
                "hurst": round(stat_signal.hurst, 3),
            },
        )
