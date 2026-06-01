"""
Shared Volume Profile analytics engine.
Used by all three trading strategies (Swing, DayTrading, Scalping HFQ).

Exports:
  compute_volume_profile(df)  → (poc, vah, val)
  compute_vwap(df)            → pd.Series
  compute_volume_delta(df)    → pd.Series  (buy_vol - sell_vol per bar)
  compute_cvd(df)             → pd.Series  (cumulative volume delta)
  compute_orderflow_score(df) → float [-1, +1]
  detect_absorption(df)       → (bull_absorb: bool, bear_absorb: bool)

Volume Delta approximation from OHLCV (no tick data required):
  buy_vol  = volume × (close − low)  / (high − low)
  sell_vol = volume × (high − close) / (high − low)
  delta    = buy_vol − sell_vol
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ─── Volume Profile ───────────────────────────────────────────────────────────

def compute_volume_profile(
    df: pd.DataFrame,
    n_bins: int = 100,
) -> tuple[float, float, float]:
    """
    Compute Volume Profile: POC, VAH, VAL from OHLCV data.

    Algorithm:
      1. Split price range [low.min, high.max] into n_bins equal bins.
      2. Distribute each candle's volume uniformly across its H-L range.
      3. POC = bin with highest cumulative volume.
      4. VAH / VAL = bounds of Value Area (70% of total volume) expanded
         outward from POC by greedy bin selection.

    Returns: (poc, vah, val)
    """
    price_low  = float(df["low"].min())
    price_high = float(df["high"].max())
    if price_high <= price_low:
        mid = (price_high + price_low) / 2.0
        return mid, price_high, price_low

    bin_size = (price_high - price_low) / n_bins
    bins     = np.linspace(price_low, price_high, n_bins + 1)
    vol_hist = np.zeros(n_bins)

    for _, row in df.iterrows():
        lo  = float(row["low"])
        hi  = float(row["high"])
        vol = float(row["volume"])
        if vol <= 0 or hi <= lo:
            continue
        lo_idx = max(0, int((lo - price_low) / bin_size))
        hi_idx = min(n_bins - 1, int((hi - price_low) / bin_size))
        n_span = max(hi_idx - lo_idx + 1, 1)
        vol_hist[lo_idx : hi_idx + 1] += vol / n_span

    # POC
    poc_idx = int(np.argmax(vol_hist))
    poc     = float(bins[poc_idx] + bin_size / 2.0)

    # Value Area (70%)
    total_vol  = vol_hist.sum()
    target     = total_vol * 0.70
    cumvol     = vol_hist[poc_idx]
    lo_idx     = poc_idx
    hi_idx     = poc_idx

    while cumvol < target:
        can_low  = lo_idx > 0
        can_high = hi_idx < n_bins - 1
        if not can_low and not can_high:
            break
        add_low  = vol_hist[lo_idx - 1] if can_low  else 0.0
        add_high = vol_hist[hi_idx + 1] if can_high else 0.0
        if add_high >= add_low:
            hi_idx  += 1
            cumvol  += add_high
        else:
            lo_idx  -= 1
            cumvol  += add_low

    vah = float(bins[hi_idx + 1])
    val = float(bins[lo_idx])
    return poc, vah, val


# ─── VWAP ─────────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP anchored to start of df.
    Typical price = (H + L + C) / 3.
    Returns pd.Series indexed same as df.
    """
    tp      = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tpv = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (cum_tpv / cum_vol).rename("vwap")


# ─── Volume Delta ─────────────────────────────────────────────────────────────

def compute_volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    Volume Delta per bar.
    Positive = net buying pressure; Negative = net selling.
    """
    hl       = (df["high"] - df["low"]).replace(0, np.nan)
    buy_vol  = df["volume"] * (df["close"] - df["low"])  / hl
    sell_vol = df["volume"] * (df["high"] - df["close"]) / hl
    half     = df["volume"] * 0.5
    delta    = buy_vol.fillna(half) - sell_vol.fillna(half)
    return delta.rename("volume_delta")


# ─── CVD ──────────────────────────────────────────────────────────────────────

def compute_cvd(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume Delta — running sum of per-bar Volume Delta."""
    return compute_volume_delta(df).cumsum().rename("cvd")


# ─── Orderflow Score ──────────────────────────────────────────────────────────

def compute_orderflow_score(df: pd.DataFrame, window: int = 5) -> float:
    """
    Composite Orderflow score in [-1, +1].

    Components (equal weight):
      1. VD ratio:   fraction of recent bars with positive VD, normalised to [-1, +1]
      2. CVD norm:   slope of CVD over window, normalised by total volume
      3. Wick score: (lower wick avg − upper wick avg) per typical bar range
         Positive = price repeatedly rejects lower wick = buying pressure

    Positive → bullish institutional pressure
    Negative → bearish institutional pressure
    """
    if len(df) < window:
        return 0.0

    recent = df.tail(window)
    vd     = compute_volume_delta(recent)

    # Component 1 — VD ratio
    buy_bars  = int((vd > 0).sum())
    sell_bars = window - buy_bars
    vd_ratio  = (buy_bars - sell_bars) / window   # in [-1, +1]

    # Component 2 — CVD momentum
    full_cvd = compute_cvd(df)
    n_back   = min(window + 1, len(full_cvd))
    slope    = float(full_cvd.iloc[-1]) - float(full_cvd.iloc[-n_back])
    tot_vol  = float(df["volume"].tail(window).sum())
    cvd_norm = float(np.clip(slope / (tot_vol + 1e-9), -1.0, 1.0))

    # Component 3 — Wick imbalance (requires close ≥ open or close < open)
    hl = (recent["high"] - recent["low"]).replace(0, np.nan)
    body_low  = recent[["close", "open"]].min(axis=1)
    body_high = recent[["close", "open"]].max(axis=1)
    lower_wick = (body_low  - recent["low"])  / hl
    upper_wick = (recent["high"] - body_high) / hl
    wick_score = float((lower_wick.mean() - upper_wick.mean()))

    composite = float(np.clip((vd_ratio + cvd_norm + wick_score) / 3.0, -1.0, 1.0))
    return composite


# ─── Absorption Detection ─────────────────────────────────────────────────────

def detect_absorption(
    df: pd.DataFrame,
    n_bars: int = 3,
    spike_mult: float = 2.0,
) -> tuple[bool, bool]:
    """
    Detect absorption at key price levels.

    Bullish absorption:
      High volume + net selling (delta < 0) + price holds / rises
      → Sellers absorbed by waiting buyers

    Bearish absorption:
      High volume + net buying (delta > 0) + price holds / falls
      → Buyers absorbed by waiting sellers

    Returns: (bullish_absorption, bearish_absorption)
    """
    if len(df) < n_bars + 5:
        return False, False

    baseline = df.iloc[-(n_bars + 5) : -n_bars]
    recent   = df.tail(n_bars)

    avg_vol    = float(baseline["volume"].mean())
    vol_spike  = float(recent["volume"].mean()) >= spike_mult * max(avg_vol, 1.0)

    vd         = compute_volume_delta(recent)
    net_delta  = float(vd.sum())
    price_chg  = float(recent["close"].iloc[-1]) - float(recent["close"].iloc[0])

    # Bullish: big vol, sellers dominate delta, but price flat or up
    bull = vol_spike and net_delta < 0 and price_chg >= -float(recent["close"].mean()) * 0.001
    # Bearish: big vol, buyers dominate delta, but price flat or down
    bear = vol_spike and net_delta > 0 and price_chg <=  float(recent["close"].mean()) * 0.001

    return bool(bull), bool(bear)
