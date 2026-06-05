"""
Test runner — Gold (GC=F) + Nasdaq (QQQ) day trading strategies.
Fetches REAL market data from yfinance — no mock, no broker, no Telegram.

Usage:
  python test_strategies.py                 # test both strategies
  python test_strategies.py --strategy gold  # Gold only
  python test_strategies.py --strategy nasdaq # Nasdaq only
  python test_strategies.py --force          # lower confluence to 0 to force signals
  python test_strategies.py --broker         # also run paper broker simulation
"""
from __future__ import annotations
import argparse
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import logger, setup_logger
setup_logger("DEBUG")

with open("config/settings.yaml") as f:
    CFG = yaml.safe_load(f)

from data.connectors.market_data import MarketDataConnector
from strategies.volume_profile import (
    compute_volume_profile, compute_vwap,
    compute_volume_delta, compute_cvd,
    compute_orderflow_score,
)
from strategies.day_trading_forex import DayTradingStrategy, DayTradingSignal
from utils.paper_broker import PaperBroker

MARKET = MarketDataConnector()
SEP    = "─" * 62


def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _print_dt(sig: DayTradingSignal) -> None:
    rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
    dxy_line = (f"  DXY bias: {sig.dxy_bias}  slope={sig.dxy_slope_pct:+.3f}%"
                if sig.dxy_bias is not None else "")
    print(
        f"  {'🟢' if sig.direction == 'BUY' else '🔴'} {sig.ticker}  "
        f"{sig.direction}  setup={sig.setup_type}  session={sig.session}\n"
        f"  Entry  : {sig.entry:.5f}\n"
        f"  SL     : {sig.stop_loss:.5f}   (pip risk: {sig.pip_risk:.1f})\n"
        f"  TP     : {sig.take_profit:.5f}\n"
        f"  R:R    : 1:{rr:.2f}  |  Lot: {sig.lot_size}\n"
        f"  Conf   : {sig.confidence:.1%}  |  Confluence: {sig.confluence_score}/5\n"
        f"  CVD bias: {sig.cvd_bias_4h}  |  OF score: {sig.orderflow_score:+.3f}\n"
        f"  POC_4H={sig.poc_4h:.5f}  POC_1H={sig.poc_1h:.5f}\n"
        f"  VWAP_4H={sig.vwap_4h:.5f}  VWAP_1H={sig.vwap_1h:.5f}"
        + ("\n" + dxy_line if dxy_line else "")
    )


def _vp_report(df, label: str) -> None:
    poc, vah, val = compute_volume_profile(df.tail(20))
    vwap  = float(compute_vwap(df).iloc[-1])
    cvd   = float(compute_cvd(df).iloc[-1])
    vd    = float(compute_volume_delta(df).iloc[-1])
    of    = compute_orderflow_score(df)
    price = float(df["close"].iloc[-1])
    print(
        f"  [{label}] price={price:.4f}  POC={poc:.4f}  VAH={vah:.4f}  VAL={val:.4f}\n"
        f"           VWAP={vwap:.4f}  CVD={cvd:+.0f}  VD(last)={vd:+.0f}  OF={of:+.3f}"
    )


# ── Test Gold (GC=F) ──────────────────────────────────────────────────────────

def test_gold(force: bool = False, run_broker: bool = False) -> None:
    _header("GOLD DAY TRADING  (GC=F, 4H→1H, London + NY)")

    cfg = CFG.copy()
    if force:
        cfg.setdefault("strategies", {}).setdefault("gold", {})["min_confluence"] = 0
        print("  ⚠️  Mode --force: confluence réduit à 0\n")

    strategy = DayTradingStrategy(cfg, "gold")
    broker   = PaperBroker(cfg) if run_broker else None
    ticker   = cfg.get("strategies", {}).get("gold", {}).get("ticker", "GC=F")

    if force:
        strategy.active_session = lambda t=None: (True, "ny")  # type: ignore

    try:
        print(f"  Fetching {ticker} [4H + 1H] ...", end="", flush=True)
        df_4h = MARKET.get_ohlcv(ticker, "4h", "60d")
        df_1h = MARKET.get_ohlcv(ticker, "1h", "30d")
        if df_4h is None or df_1h is None or len(df_4h) < 25 or len(df_1h) < 20:
            print(" insufficient data")
            return
        print(f" 4H:{len(df_4h)} bars  1H:{len(df_1h)} bars")

        _vp_report(df_4h.tail(20), "4H VP")
        _vp_report(df_1h.tail(20), "1H VP")

        df_dxy = None
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
            assert sig.direction in ("BUY", "SELL"), f"direction invalid"
            assert sig.confidence >= 0.50,           f"confidence too low"
            assert sig.lot_size > 0,                 f"lot_size zero"
            if sig.direction == "BUY":
                assert sig.stop_loss < sig.entry,    f"BUY: SL >= entry"
                assert sig.take_profit > sig.entry,  f"BUY: TP <= entry"
            else:
                assert sig.stop_loss > sig.entry,    f"SELL: SL <= entry"
                assert sig.take_profit < sig.entry,  f"SELL: TP >= entry"
            rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
            assert rr >= 1.5, f"R:R {rr:.2f} < 1.5"
            _print_dt(sig)
            if broker:
                pos = broker.execute(sig, "day_trading")
                print(f"  → PAPER FILL @ {pos.entry_price:.5f}  id={pos.id}")
        else:
            active, session = strategy.active_session(ticker)
            print(f"  No signal — session={session} active={active}")
    except Exception as e:
        print(f"\n  ERROR {ticker}: {e}")
        import traceback; traceback.print_exc()


# ── Test Nasdaq (QQQ) ─────────────────────────────────────────────────────────

def test_nasdaq(force: bool = False, run_broker: bool = False) -> None:
    _header("NASDAQ DAY TRADING  (QQQ, 4H→1H, NYSE session)")

    cfg = CFG.copy()
    if force:
        cfg.setdefault("strategies", {}).setdefault("nasdaq", {})["min_confluence"] = 0
        print("  ⚠️  Mode --force: confluence réduit à 0\n")

    strategy = DayTradingStrategy(cfg, "nasdaq")
    broker   = PaperBroker(cfg) if run_broker else None
    ticker   = cfg.get("strategies", {}).get("nasdaq", {}).get("ticker", "QQQ")

    if force:
        strategy.active_session = lambda t=None: (True, "ny")  # type: ignore

    try:
        print(f"  Fetching {ticker} [4H + 1H] ...", end="", flush=True)
        df_4h = MARKET.get_ohlcv(ticker, "4h", "60d")
        df_1h = MARKET.get_ohlcv(ticker, "1h", "30d")
        if df_4h is None or df_1h is None or len(df_4h) < 25 or len(df_1h) < 20:
            print(" insufficient data")
            return
        print(f" 4H:{len(df_4h)} bars  1H:{len(df_1h)} bars")

        _vp_report(df_4h.tail(20), "4H VP")
        _vp_report(df_1h.tail(20), "1H VP")

        sig = strategy.analyse(ticker, df_4h, df_1h)
        if sig:
            assert sig.direction in ("BUY", "SELL"), f"direction invalid"
            assert sig.confidence >= 0.50,           f"confidence too low"
            assert sig.lot_size > 0,                 f"lot_size zero"
            if sig.direction == "BUY":
                assert sig.stop_loss < sig.entry,    f"BUY: SL >= entry"
                assert sig.take_profit > sig.entry,  f"BUY: TP <= entry"
            else:
                assert sig.stop_loss > sig.entry,    f"SELL: SL <= entry"
                assert sig.take_profit < sig.entry,  f"SELL: TP >= entry"
            rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
            assert rr >= 1.5, f"R:R {rr:.2f} < 1.5"
            _print_dt(sig)
            if broker:
                pos = broker.execute(sig, "day_trading")
                print(f"  → PAPER FILL @ {pos.entry_price:.5f}  id={pos.id}")
        else:
            active, session = strategy.active_session(ticker)
            print(f"  No signal — session={session} active={active}")
    except Exception as e:
        print(f"\n  ERROR {ticker}: {e}")
        import traceback; traceback.print_exc()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AlgoTrad — Gold + Nasdaq strategy tester")
    parser.add_argument("--strategy", choices=["gold", "nasdaq"],
                        help="gold | nasdaq  (default: both)")
    parser.add_argument("--force",  action="store_true",
                        help="Lower confluence to 0 — forces signal generation")
    parser.add_argument("--broker", action="store_true",
                        help="Simulate paper broker execution on each signal")
    args = parser.parse_args()

    print("\n" + "═" * 62)
    print("  ALGOTRAD — Gold + Nasdaq Day Trading Test Runner")
    print("  Data source : yfinance (real market data)")
    print(f"  Force mode  : {'ON — filters bypassed' if args.force else 'OFF'}")
    print(f"  Broker sim  : {'ON — paper execution' if args.broker else 'OFF'}")
    print("═" * 62)

    run_all = args.strategy is None
    if run_all or args.strategy == "gold":
        test_gold(force=args.force, run_broker=args.broker)
    if run_all or args.strategy == "nasdaq":
        test_nasdaq(force=args.force, run_broker=args.broker)

    print("\n" + "═" * 62)
    print("  Test complete.")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
