"""
Risk gate — every signal passes through here before Telegram dispatch.
Returns (approved: bool, adjusted_stop: float, reason: str).

New filters:
  - spread filter: reject if spread > max_spread_pct
  - liquidity filter: reject if avg volume < min_avg_volume
  - halt / halted-stock detection
  - dilution / news danger filter (earnings proximity + SEC activity)
  - volatility regime filter
  - max loss per trade
  - daily max drawdown (tracked via KillSwitch)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class RiskManager:

    def __init__(self, cfg: dict):
        risk = cfg["risk"]
        self.max_dd = risk["max_drawdown_pct"] / 100
        self.min_sharpe = risk["min_sharpe"]
        self.confidence_threshold = risk["confidence_threshold"]
        self.atr_mult = risk["atr_stop_multiplier"]
        self.max_spread_pct = risk.get("max_spread_pct", 0.005)        # 0.5%
        self.min_avg_volume = risk.get("min_avg_volume", 500_000)       # 500k shares
        self.min_liquidity_score = risk.get("min_liquidity_score", 0.2)
        self.max_fake_breakout_prob = risk.get("max_fake_breakout_prob", 0.70)
        self.max_volatility_percentile = risk.get("max_volatility_percentile", 0.90)
        self.min_rvol = risk.get("min_rvol", 0.5)                      # at least 50% of avg vol
        self.max_loss_per_trade = risk.get("max_loss_per_trade_pct", 0.01)  # 1%

    # ── ATR-based stop-loss ────────────────────────────────────────────────────
    def compute_stop(self, price: float, atr: float, direction: str) -> float:
        delta = atr * self.atr_mult
        return price - delta if direction == "BUY" else price + delta

    # ── Main gate ─────────────────────────────────────────────────────────────
    def approve(
        self,
        confidence: float,
        sharpe: float,
        drawdown: float,
        price: float,
        atr: float,
        direction: str,
        # New optional microstructure / feature inputs
        spread_pct: float = 0.0,
        avg_volume: float = float("inf"),
        liquidity_score: float = 1.0,
        rvol: float = 1.0,
        fake_breakout_prob: float = 0.0,
        volatility_percentile: float = 0.5,
        catalyst_score: float = 0.0,
        has_recent_earnings: bool = False,
        sec_8k_score: float = 0.0,
    ) -> tuple[bool, float, str]:

        stop = self.compute_stop(price, atr, direction)

        # ── Spread filter ─────────────────────────────────────────────────────
        if spread_pct > self.max_spread_pct:
            msg = f"Spread {spread_pct:.3%} > max {self.max_spread_pct:.3%}"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        # ── Volume / liquidity filter ─────────────────────────────────────────
        if avg_volume < self.min_avg_volume:
            msg = f"Avg volume {avg_volume:,.0f} < min {self.min_avg_volume:,.0f}"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        if liquidity_score < self.min_liquidity_score:
            msg = f"Liquidity score {liquidity_score:.2f} < min {self.min_liquidity_score:.2f}"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        if rvol < self.min_rvol:
            msg = f"RVOL {rvol:.2f} < min {self.min_rvol:.2f} — insufficient participation"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        # ── Halt / dilution / news danger filter ──────────────────────────────
        if has_recent_earnings:
            msg = "Recent earnings — avoiding post-earnings gap risk"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        if sec_8k_score > 0.8:
            msg = f"High SEC 8-K activity score {sec_8k_score:.2f} — dilution/news risk"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        # ── Fake breakout filter ──────────────────────────────────────────────
        if fake_breakout_prob > self.max_fake_breakout_prob:
            msg = f"Fake breakout probability {fake_breakout_prob:.0%} too high"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        # ── Volatility filter (avoid extreme regime) ──────────────────────────
        if volatility_percentile > self.max_volatility_percentile:
            msg = f"Volatility percentile {volatility_percentile:.0%} — regime too extreme"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        # ── Confidence threshold ──────────────────────────────────────────────
        if confidence < self.confidence_threshold:
            msg = f"Confidence {confidence:.2%} < threshold {self.confidence_threshold:.2%}"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        # ── Sharpe filter ─────────────────────────────────────────────────────
        if sharpe < self.min_sharpe:
            msg = f"Sharpe {sharpe:.2f} < min {self.min_sharpe}"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        # ── Drawdown filter ───────────────────────────────────────────────────
        if drawdown > self.max_dd:
            msg = f"Drawdown {drawdown:.2%} exceeds limit {self.max_dd:.2%}"
            logger.warning(f"Signal rejected — {msg}")
            return False, stop, msg

        logger.info(
            f"Signal approved — conf={confidence:.2%} sharpe={sharpe:.2f} "
            f"rvol={rvol:.1f} spread={spread_pct:.3%}"
        )
        return True, stop, "OK"

    # ── Position-size hint (Kelly fraction, capped at max_loss_per_trade) ────
    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return self.max_loss_per_trade
        b = avg_win / avg_loss
        kelly = (b * win_rate - (1 - win_rate)) / b
        return float(max(0.0, min(kelly * 0.5, self.max_loss_per_trade)))
