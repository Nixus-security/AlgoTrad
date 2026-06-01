"""
Adaptive Parameter Store.
Persists parameter adjustments (from TradeAnalyzer) to JSON.
Strategies read from this store before each analysis cycle.

Adjustment types:
  min_confluence_delta       → +1 or +2 (raise entry bar)
  breakout_vol_mult_delta    → +0.25 (need more volume proof)
  blocked_setups             → ["poc_breakout"] (disable setup type)
  blocked_sessions           → ["london"] (skip session)
  blacklisted_tickers        → ["UPST"] (skip ticker for 7 days)
  max_hold_bars_delta        → -5 (exit sooner)
  vwap_tol_delta             → -0.0002 (tighter proximity filter)

All adjustments expire after `_expires_in_days` (default 7).
On expiry, parameters revert to cfg defaults.

JSON structure (adaptive_params.json):
{
  "swing": {
    "min_confluence_delta": 1,
    "blocked_setups": ["poc_breakout"],
    "blacklisted_tickers": ["UPST"],
    "blocked_sessions": [],
    "breakout_vol_mult_delta": 0.25,
    "vwap_tol_delta": 0.0,
    "max_hold_bars_delta": 0
  },
  "day_trading": { ... },
  "scalping_hfq": { ... },
  "_expires_at": 1234567890.0,
  "_generated_at": 1234567890.0
}
"""
from __future__ import annotations
import json
import os
import time
from typing import Any
from utils.logger import logger

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "adaptive_params.json")

_DEFAULTS: dict[str, dict] = {
    "swing": {
        "min_confluence_delta":     0,
        "breakout_vol_mult_delta":  0.0,
        "blocked_setups":           [],
        "blocked_sessions":         [],
        "blacklisted_tickers":      [],
        "max_hold_bars_delta":      0,
        "vwap_tol_delta":           0.0,
    },
    "day_trading": {
        "min_confluence_delta":     0,
        "breakout_vol_mult_delta":  0.0,
        "blocked_setups":           [],
        "blocked_sessions":         [],
        "blacklisted_tickers":      [],
        "max_hold_bars_delta":      0,
        "vwap_tol_delta":           0.0,
    },
    "scalping_hfq": {
        "min_confluence_delta":     0,
        "breakout_vol_mult_delta":  0.0,
        "blocked_setups":           [],
        "blocked_sessions":         [],
        "blacklisted_tickers":      [],
        "max_hold_bars_delta":      0,
        "vwap_tol_delta":           0.0,
    },
}


class AdaptiveParams:
    """
    Reads / writes / expires adaptive parameter adjustments.
    Thread-safe via file reload on every access.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
        self._cache: dict = {}
        self._load()

    # ── Read ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(PARAMS_PATH):
            self._cache = {}
            return
        try:
            with open(PARAMS_PATH) as f:
                self._cache = json.load(f)
        except Exception as e:
            logger.warning(f"AdaptiveParams: load error: {e}")
            self._cache = {}

    def _get_fresh(self) -> dict:
        """Reload from disk, check expiry."""
        self._load()
        expires_at = self._cache.get("_expires_at", 0.0)
        if expires_at and time.time() > expires_at:
            logger.info("AdaptiveParams: adjustments expired — reverting to defaults")
            self._cache = {}
            # Remove stale file
            try:
                os.remove(PARAMS_PATH)
            except Exception:
                pass
        return self._cache

    def get_strategy(self, strategy: str) -> dict:
        """Return current adjustments for a strategy (or defaults if none/expired)."""
        data = self._get_fresh()
        adj = data.get(strategy, {})
        defaults = _DEFAULTS.get(strategy, {})
        # Merge with defaults (defaults fill missing keys)
        return {**defaults, **adj}

    def is_setup_blocked(self, strategy: str, setup_type: str) -> bool:
        adj = self.get_strategy(strategy)
        return setup_type in adj.get("blocked_setups", [])

    def is_session_blocked(self, strategy: str, session: str) -> bool:
        adj = self.get_strategy(strategy)
        return session in adj.get("blocked_sessions", [])

    def is_ticker_blacklisted(self, strategy: str, ticker: str) -> bool:
        adj = self.get_strategy(strategy)
        return ticker in adj.get("blacklisted_tickers", [])

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply(self, adjustments: dict, expires_in_days: float = 7.0) -> None:
        """
        Apply new adjustments dict (output of TradeAnalyzer.analyse().adjustments).
        Merges with existing adjustments (new values take precedence).
        """
        self._load()

        for key, val in adjustments.items():
            if key.startswith("_"):
                continue  # skip metadata keys
            if key not in self._cache:
                self._cache[key] = {}
            # Lists: union (add new items)
            for list_key in ["blocked_setups", "blocked_sessions", "blacklisted_tickers"]:
                if list_key in val:
                    existing = self._cache[key].get(list_key, [])
                    merged   = list(set(existing + val[list_key]))
                    self._cache[key][list_key] = merged
            # Scalars: accumulate deltas (but cap to reasonable range)
            for delta_key in ["min_confluence_delta", "breakout_vol_mult_delta",
                               "max_hold_bars_delta", "vwap_tol_delta"]:
                if delta_key in val:
                    prev = self._cache[key].get(delta_key, 0)
                    new_val = prev + val[delta_key]
                    # Caps
                    if delta_key == "min_confluence_delta":
                        new_val = max(0, min(new_val, 3))   # max +3 extra confluence
                    self._cache[key][delta_key] = new_val

        self._cache["_generated_at"] = time.time()
        self._cache["_expires_at"]   = time.time() + expires_in_days * 86400

        self._save()
        logger.info(
            f"AdaptiveParams: adjustments applied for "
            f"{[k for k in adjustments if not k.startswith('_')]}  "
            f"expires in {expires_in_days:.0f} days"
        )

    def reset(self, strategy: str | None = None) -> None:
        """Reset adjustments (all strategies or specific one)."""
        self._load()
        if strategy:
            self._cache.pop(strategy, None)
            logger.info(f"AdaptiveParams: reset {strategy}")
        else:
            self._cache = {}
            try:
                os.remove(PARAMS_PATH)
            except Exception:
                pass
            logger.info("AdaptiveParams: all adjustments reset")
        if self._cache:
            self._save()

    # ── Patch strategy instances ──────────────────────────────────────────────

    def patch_strategy(self, strategy_obj, strategy_type: str) -> None:
        """
        Directly update a strategy instance's thresholds with current adjustments.
        Call before each strategy cycle.
        """
        adj = self.get_strategy(strategy_type)

        # min_confluence
        delta_conf = int(adj.get("min_confluence_delta", 0))
        if delta_conf != 0 and hasattr(strategy_obj, "min_confluence"):
            orig = getattr(strategy_obj, "_base_min_confluence",
                           strategy_obj.min_confluence)
            strategy_obj._base_min_confluence = orig
            strategy_obj.min_confluence       = orig + delta_conf

        # breakout_vol_mult
        delta_vol = float(adj.get("breakout_vol_mult_delta", 0.0))
        if delta_vol != 0 and hasattr(strategy_obj, "breakout_vol_mult"):
            orig = getattr(strategy_obj, "_base_breakout_vol_mult",
                           strategy_obj.breakout_vol_mult)
            strategy_obj._base_breakout_vol_mult = orig
            strategy_obj.breakout_vol_mult       = round(orig + delta_vol, 3)

        # vwap_tol
        delta_tol = float(adj.get("vwap_tol_delta", 0.0))
        if delta_tol != 0 and hasattr(strategy_obj, "vwap_tol"):
            orig = getattr(strategy_obj, "_base_vwap_tol", strategy_obj.vwap_tol)
            strategy_obj._base_vwap_tol = orig
            strategy_obj.vwap_tol       = max(0.0001, orig + delta_tol)

        # max_hold_bars (scalping)
        delta_hold = int(adj.get("max_hold_bars_delta", 0))
        if delta_hold != 0 and hasattr(strategy_obj, "tp_ticks"):
            # For scalping, shorter hold → reduce tp_ticks
            pass   # handled at analysis level

        # Blocked setups / sessions / tickers — injected directly into strategy
        blocked_setups = adj.get("blocked_setups", [])
        if hasattr(strategy_obj, "_blocked_setups"):
            strategy_obj._blocked_setups = blocked_setups
            if blocked_setups:
                logger.info(f"AdaptiveParams [{strategy_type}]: blocked setups={blocked_setups}")

        blocked_sessions = adj.get("blocked_sessions", [])
        if hasattr(strategy_obj, "_blocked_sessions"):
            strategy_obj._blocked_sessions = blocked_sessions
            if blocked_sessions:
                logger.info(f"AdaptiveParams [{strategy_type}]: blocked sessions={blocked_sessions}")

        blacklisted = adj.get("blacklisted_tickers", [])
        if hasattr(strategy_obj, "_blacklisted_tickers"):
            strategy_obj._blacklisted_tickers = blacklisted
            if blacklisted:
                logger.info(f"AdaptiveParams [{strategy_type}]: blacklisted tickers={blacklisted}")

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        data = self._get_fresh()
        if not data:
            return "AdaptiveParams: aucun ajustement actif (paramètres par défaut)"

        import datetime
        exp = data.get("_expires_at", 0)
        exp_str = (
            datetime.datetime.fromtimestamp(exp).strftime("%d/%m %H:%M")
            if exp else "N/A"
        )
        lines = [f"AdaptiveParams (expire: {exp_str})"]
        for st in ["swing", "day_trading", "scalping_hfq"]:
            adj = data.get(st)
            if adj:
                lines.append(f"  {st}: {adj}")
        return "\n".join(lines)

    # ── Persist ───────────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            with open(PARAMS_PATH, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"AdaptiveParams: save error: {e}")
