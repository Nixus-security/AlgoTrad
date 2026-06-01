"""
Test runner for the three trading strategies.
Fetches REAL market data from yfinance — no mock, no broker, no Telegram.

Usage:
  python test_strategies.py               # test all three strategies
  python test_strategies.py --strategy 1  # swing only
  python test_strategies.py --strategy 2  # day trading only
  python test_strategies.py --strategy 3  # scalping only
  python test_strategies.py --force       # lower confluence to 0 to force signals
  python test_strategies.py --broker      # also run paper broker simulation

Outputs:
  - Console: signal details per ticker
  - logs/test_run.log
"""
from __future__ import annotations
import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd

# Ensure project root in path
sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import logger, setup_logger
setup_logger("DEBUG")

# ── Load config ────────────────────────────────────────────────────────────────
with open("config/settings.yaml") as f:
    CFG = yaml.safe_load(f)

from data.connectors.market_data import MarketDataConnector
from strategies.volume_profile import (
    compute_volume_profile, compute_vwap,
    compute_volume_delta, compute_cvd,
    compute_orderflow_score, detect_absorption,
)
from strategies.swing_trading     import SwingTradingStrategy, SwingSignal
from strategies.day_trading_forex import DayTradingForexStrategy, DayTradingSignal, PAIR_META
from strategies.scalping_hfq      import ScalpingHFQStrategy, ScalpSignal
from utils.paper_broker           import PaperBroker

MARKET = MarketDataConnector()
SEP    = "─" * 62


# ── Helpers ────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _print_swing(sig: SwingSignal) -> None:
    rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
    print(
        f"  {'🟢' if sig.direction == 'BUY' else '🔴'} {sig.ticker}  "
        f"{sig.direction}  setup={sig.setup_type}\n"
        f"  Entry  : {sig.entry:.4f}\n"
        f"  SL     : {sig.stop_loss:.4f}   (−{abs(sig.entry - sig.stop_loss):.4f})\n"
        f"  TP     : {sig.take_profit:.4f}  (+{abs(sig.take_profit - sig.entry):.4f})\n"
        f"  R:R    : 1:{rr:.2f}\n"
        f"  Risk   : {sig.risk_amount:.2f}$  |  Shares: {sig.position_size_shares:.1f}\n"
        f"  Conf   : {sig.confidence:.1%}  |  Confluence: {sig.confluence_score}/4\n"
        f"  POC={sig.poc:.4f}  VAH={sig.vah:.4f}  VAL={sig.val:.4f}\n"
        f"  VWAP={sig.vwap:.4f}  ATR={sig.atr:.4f}  Vol×={sig.vol_ratio:.2f}\n"
        f"  CVD div={sig.cvd_divergent}  VD conf={sig.volume_delta_confirming}  "
        f"Balancing={sig.is_balancing}"
    )


def _print_dt(sig: DayTradingSignal) -> None:
    rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
    print(
        f"  {'🟢' if sig.direction == 'BUY' else '🔴'} {sig.ticker}  "
        f"{sig.direction}  setup={sig.setup_type}  session={sig.session}\n"
        f"  Entry  : {sig.entry:.5f}\n"
        f"  SL     : {sig.stop_loss:.5f}   (pip risk: {sig.pip_risk:.1f})\n"
        f"  TP     : {sig.take_profit:.5f}\n"
        f"  R:R    : 1:{rr:.2f}  |  Lot: {sig.lot_size}\n"
        f"  Conf   : {sig.confidence:.1%}  |  Confluence: {sig.confluence_score}/5\n"
        f"  CVD bias: {sig.cvd_bias_4h}  |  OF score: {sig.orderflow_score:+.3f}\n"
        f"  POC_4H={sig.poc_4h:.5f}  VWAP_4H={sig.vwap_4h:.5f}  VWAP_1H={sig.vwap_1h:.5f}\n"
        + (f"  DXY bias: {sig.dxy_bias}  slope={sig.dxy_slope_pct:+.3f}%"
           if sig.dxy_bias is not None else "")
    )


def _print_scalp(sig: ScalpSignal) -> None:
    rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
    print(
        f"  {'🟢' if sig.direction == 'BUY' else '🔴'} {sig.ticker}  "
        f"{sig.direction}  setup={sig.setup_type}\n"
        f"  Entry  : {sig.entry:.2f}\n"
        f"  SL     : {sig.stop_loss:.2f}  ({sig.sl_ticks} ticks)\n"
        f"  TP     : {sig.take_profit:.2f}  ({sig.tp_ticks} ticks)\n"
        f"  R:R    : 1:{rr:.2f}  |  Contracts: {sig.contracts}\n"
        f"  Conf   : {sig.confidence:.1%}  |  Vol spike: {sig.volume_spike_ratio:.1f}×\n"
        f"  POC={sig.poc:.2f}  VWAP={sig.vwap:.2f}  VAH={sig.vah:.2f}  VAL={sig.val:.2f}\n"
        f"  CVD={sig.cvd_at_signal:.0f}  Absorb bull={sig.absorption_bull}  bear={sig.absorption_bear}"
    )


def _vp_report(df: pd.DataFrame, label: str) -> None:
    """Print Volume Profile summary for a dataframe."""
    poc, vah, val = compute_volume_profile(df.tail(20))
    vwap = float(compute_vwap(df).iloc[-1])
    cvd  = float(compute_cvd(df).iloc[-1])
    vd   = float(compute_volume_delta(df).iloc[-1])
    of   = compute_orderflow_score(df)
    price= float(df["close"].iloc[-1])
    bull_abs, bear_abs = detect_absorption(df)
    print(
        f"  [{label}] price={price:.4f}  POC={poc:.4f}  VAH={vah:.4f}  VAL={val:.4f}\n"
        f"           VWAP={vwap:.4f}  CVD={cvd:+.0f}  VD(last)={vd:+.0f}  OF={of:+.3f}\n"
        f"           Absorption bull={bull_abs}  bear={bear_abs}"
    )


# ── Strategy 1: Swing Trading ──────────────────────────────────────────────────

def test_swing(force: bool = False, run_broker: bool = False) -> None:
    _header("STRATÉGIE 1 — SWING TRADING  (Small/Mid Cap, 1D)")

    cfg = CFG.copy()
    if force:
        cfg.setdefault("strategies", {}).setdefault("swing", {})["min_confluence"] = 0
        print("  ⚠️  Mode --force: confluence réduit à 0 (affiche tout setup partiel)\n")

    strategy = SwingTradingStrategy(cfg)
    broker   = PaperBroker(cfg) if run_broker else None

    tickers: list[str] = CFG.get("strategies", {}).get("swing", {}).get("tickers", [
        "AFRM", "SOFI", "UPST", "RIVN", "HOOD", "IONQ", "MARA",
    ])

    signals_found = 0
    for ticker in tickers:
        try:
            print(f"\n  Fetching {ticker} [1D 90d] ...", end="", flush=True)
            df = MARKET.get_ohlcv(ticker, "1d", "90d")
            if df is None or len(df) < 25:
                print(" insufficient data — skip")
                continue
            print(f" {len(df)} bars", end="")

            # VP report
            _vp_report(df, "VP")

            sig = strategy.analyse(ticker, df)
            if sig:
                # ── Assertions ────────────────────────────────────────────────
                assert sig.direction in ("BUY", "SELL"),          f"direction invalid: {sig.direction}"
                assert sig.confidence >= 0.50,                    f"confidence too low: {sig.confidence}"
                assert sig.risk_amount > 0,                       f"risk_amount zero/negative"
                assert sig.stop_loss != sig.entry,                f"SL == entry"
                assert sig.take_profit != sig.entry,              f"TP == entry"
                if sig.direction == "BUY":
                    assert sig.stop_loss < sig.entry,             f"BUY: SL {sig.stop_loss} >= entry {sig.entry}"
                    assert sig.take_profit > sig.entry,           f"BUY: TP {sig.take_profit} <= entry {sig.entry}"
                else:
                    assert sig.stop_loss > sig.entry,             f"SELL: SL {sig.stop_loss} <= entry {sig.entry}"
                    assert sig.take_profit < sig.entry,           f"SELL: TP {sig.take_profit} >= entry {sig.entry}"
                rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
                assert rr >= 1.5,                                 f"R:R {rr:.2f} < 1.5"
                # ──────────────────────────────────────────────────────────────
                _print_swing(sig)
                signals_found += 1
                if broker:
                    pos = broker.execute(sig, "swing")
                    print(f"  → PAPER FILL @ {pos.entry_price:.4f}  id={pos.id}")
            else:
                price = float(df["close"].iloc[-1])
                poc, vah, val = compute_volume_profile(df.tail(20))
                dist_val = (price - val) / val * 100
                dist_vah = (price - vah) / vah * 100
                print(
                    f"\n  No signal — price={price:.4f}  "
                    f"dist_VAL={dist_val:+.1f}%  dist_VAH={dist_vah:+.1f}%"
                )
        except Exception as e:
            print(f"\n  ERROR {ticker}: {e}")

    print(f"\n  → {signals_found}/{len(tickers)} signal(s) found")
    if broker:
        eq = broker.equity("swing")
        print(f"  Paper equity: {eq.capital:.2f}$  trades={eq.total_trades}")


# ── Strategy 2: Day Trading Forex + Gold ──────────────────────────────────────

def test_daytrading(force: bool = False, run_broker: bool = False) -> None:
    _header("STRATÉGIE 2 — DAY TRADING  (Forex + Gold, 4H→1H)")

    cfg = CFG.copy()
    if force:
        cfg.setdefault("strategies", {}).setdefault("day_trading", {})["min_confluence"] = 0
        print("  ⚠️  Mode --force: confluence réduit à 0\n")

    strategy = DayTradingForexStrategy(cfg)
    broker   = PaperBroker(cfg) if run_broker else None

    # Override session check in force mode
    if force:
        strategy.active_session = lambda: (True, "ny")  # type: ignore

    tickers: list[str] = CFG.get("strategies", {}).get("day_trading", {}).get("tickers", [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F",
    ])

    signals_found = 0
    for ticker in tickers:
        try:
            print(f"\n  Fetching {ticker} [4H + 1H] ...", end="", flush=True)
            df_4h = MARKET.get_ohlcv(ticker, "4h", "60d")
            df_1h = MARKET.get_ohlcv(ticker, "1h", "30d")
            if df_4h is None or df_1h is None or len(df_4h) < 25 or len(df_1h) < 20:
                print(" insufficient data — skip")
                continue
            print(f" 4H:{len(df_4h)} bars  1H:{len(df_1h)} bars")

            _vp_report(df_4h.tail(20), "4H VP")
            _vp_report(df_1h.tail(20), "1H VP")

            # Fetch DXY for Gold (inverse correlation filter)
            df_dxy = None
            if ticker == "GC=F":
                try:
                    df_dxy = MARKET.get_ohlcv("DX-Y.NYB", "1h", "10d")
                    if df_dxy is not None and len(df_dxy) >= 8:
                        dxy_now   = float(df_dxy["close"].iloc[-1])
                        dxy_slope = (dxy_now - float(df_dxy["close"].iloc[-6])) / dxy_now * 100
                        print(f"  DXY: {dxy_now:.2f}  slope(6h)={dxy_slope:+.3f}%  "
                              f"({'bearish→Gold bullish' if dxy_slope < 0 else 'bullish→Gold bearish'})")
                    else:
                        df_dxy = None
                except Exception:
                    df_dxy = None

            sig = strategy.analyse(ticker, df_4h, df_1h, df_dxy_1h=df_dxy)
            if sig:
                # ── Assertions ────────────────────────────────────────────────
                assert sig.direction in ("BUY", "SELL"),   f"direction: {sig.direction}"
                assert sig.confidence >= 0.50,             f"confidence: {sig.confidence}"
                assert sig.lot_size > 0,                   f"lot_size zero"
                assert sig.pip_risk > 0,                   f"pip_risk zero"
                if sig.direction == "BUY":
                    assert sig.stop_loss < sig.entry,      f"BUY: SL >= entry"
                    assert sig.take_profit > sig.entry,    f"BUY: TP <= entry"
                else:
                    assert sig.stop_loss > sig.entry,      f"SELL: SL <= entry"
                    assert sig.take_profit < sig.entry,    f"SELL: TP >= entry"
                rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
                assert rr >= 1.5,                          f"R:R {rr:.2f} < 1.5"
                # ──────────────────────────────────────────────────────────────
                _print_dt(sig)
                signals_found += 1
                if broker:
                    pos = broker.execute(sig, "day_trading")
                    print(f"  → PAPER FILL @ {pos.entry_price:.5f}  id={pos.id}")
            else:
                active, session = strategy.active_session()
                print(f"  No signal — session={session} active={active}")
        except Exception as e:
            print(f"\n  ERROR {ticker}: {e}")

    print(f"\n  → {signals_found}/{len(tickers)} signal(s) found")
    if broker:
        eq = broker.equity("day_trading")
        print(f"  Paper equity: {eq.capital:.2f}$  trades={eq.total_trades}")


# ── Strategy 3: Scalping HFQ ──────────────────────────────────────────────────

def test_scalping(force: bool = False, run_broker: bool = False) -> None:
    _header("STRATÉGIE 3 — SCALPING HFQ  (ES=F / SPX, 1m)")

    cfg = CFG.copy()
    if force:
        cfg.setdefault("strategies", {}).setdefault("scalping_hfq", {})["min_vol_ratio"] = 0.0
        cfg["strategies"]["scalping_hfq"]["vwap_tol_pts"] = 999.0   # match any price
        print("  ⚠️  Mode --force: vol_ratio=0 et vwap_tol=999 (force signal sur premier bar)\n")

    strategy = ScalpingHFQStrategy(cfg)
    broker   = PaperBroker(cfg) if run_broker else None

    ticker = CFG.get("strategies", {}).get("scalping_hfq", {}).get("ticker", "ES=F")

    try:
        print(f"  Fetching {ticker} [1m 7d] ...", end="", flush=True)
        df = MARKET.get_ohlcv(ticker, "1m", "7d")
        if df is None or len(df) < 15:
            print(" insufficient data")
            return
        print(f" {len(df)} bars")

        _vp_report(df.tail(30), "Session VP")

        # Print CVD trend
        cvd = compute_cvd(df)
        print(
            f"  CVD(now)={float(cvd.iloc[-1]):+.0f}  "
            f"CVD(10 bars ago)={float(cvd.iloc[-11]):+.0f}  "
            f"slope={float(cvd.iloc[-1]) - float(cvd.iloc[-11]):+.0f}"
        )

        # Simulate market-open (force bypass in test)
        from strategies import scalping_hfq as _shfq
        _shfq._market_open = lambda: True  # type: ignore  — bypass for testing

        # Run analyse on last 40 bars to find any signal
        signals_found = 0
        for i in range(max(15, len(df) - 40), len(df)):
            df_window = df.iloc[:i + 1]
            strategy._bars_since_open = AVOID_OPEN_BARS + 1   # bypass open filter
            sig = strategy.analyse.__func__(strategy, ticker, df_window)  # type: ignore
            if sig:
                # ── Assertions ────────────────────────────────────────────────
                assert sig.direction in ("BUY", "SELL"),   f"direction: {sig.direction}"
                assert sig.confidence >= 0.50,             f"confidence: {sig.confidence}"
                assert sig.contracts > 0,                  f"contracts zero"
                if sig.direction == "BUY":
                    assert sig.stop_loss < sig.entry,      f"BUY: SL >= entry"
                    assert sig.take_profit > sig.entry,    f"BUY: TP <= entry"
                else:
                    assert sig.stop_loss > sig.entry,      f"SELL: SL <= entry"
                    assert sig.take_profit < sig.entry,    f"SELL: TP >= entry"
                # ──────────────────────────────────────────────────────────────
                _print_scalp(sig)
                signals_found += 1
                if broker and signals_found == 1:
                    pos = broker.execute(sig, "scalping_hfq")
                    print(f"  → PAPER FILL @ {pos.entry_price:.2f}  id={pos.id}")
                break   # show first signal only

        if signals_found == 0:
            price = float(df["close"].iloc[-1])
            poc, vah, val = compute_volume_profile(df.tail(30))
            print(
                f"\n  No signal on last 40 bars — price={price:.2f}  "
                f"POC={poc:.2f}  VWAP={float(compute_vwap(df).iloc[-1]):.2f}"
            )
            print(
                "  Tip: try --force to bypass vol/proximity filters"
            )

    except Exception as e:
        print(f"\n  ERROR {ticker}: {e}")
        import traceback; traceback.print_exc()

    if broker:
        eq = broker.equity("scalping_hfq")
        print(f"  Paper equity: {eq.capital:.2f}$  trades={eq.total_trades}")


# ── AVOID_OPEN_BARS constant needed ───────────────────────────────────────────
from strategies.scalping_hfq import AVOID_OPEN_BARS


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test the three trading strategies")
    parser.add_argument("--strategy", type=int, choices=[1, 2, 3],
                        help="1=Swing  2=DayTrading  3=Scalping  (default: all)")
    parser.add_argument("--force",  action="store_true",
                        help="Lower confluence/filters to 0 — forces signal generation")
    parser.add_argument("--broker", action="store_true",
                        help="Also simulate paper broker execution on each signal")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Test single ticker only (e.g. GC=F, EURUSD=X)")
    args = parser.parse_args()

    print("\n" + "═" * 62)
    print("  ALGOTRAD — Strategy Test Runner")
    print("  Data source : yfinance (real market data)")
    print(f"  Force mode  : {'ON — filters bypassed' if args.force else 'OFF'}")
    print(f"  Broker sim  : {'ON — paper execution' if args.broker else 'OFF'}")
    if args.ticker:
        print(f"  Ticker filter: {args.ticker}")
    print("═" * 62)

    # Inject single-ticker filter into CFG for day trading
    if args.ticker:
        CFG.setdefault("strategies", {}).setdefault("day_trading", {})["tickers"] = [args.ticker]
        CFG.setdefault("strategies", {}).setdefault("swing", {})["tickers"] = [args.ticker]

    run_all = args.strategy is None
    if run_all or args.strategy == 1:
        test_swing(force=args.force, run_broker=args.broker)
    if run_all or args.strategy == 2:
        test_daytrading(force=args.force, run_broker=args.broker)
    if run_all or args.strategy == 3:
        test_scalping(force=args.force, run_broker=args.broker)

    print("\n" + "═" * 62)
    print("  Test complete. Check logs/test_run.log for full output.")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
