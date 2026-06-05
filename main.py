"""
AlgoTrad — Main entry point.
Two strategies: Day Trading Gold (GC=F) + Day Trading Nasdaq (QQQ).
Max 3 trades/day per strategy (6 total).

Pipeline per asset:
  market data → microstructure → catalyst → sentiment → features
  → LSTM + Ensemble(XGB/LGB) + NLP
  → Quantum weight optimisation
  → Signal fusion → Risk gate → Kill switch → Dispatch

Usage:
  python main.py                    # Live signal generation loop
  python main.py --paper            # Paper-trading dry-run
  python main.py --backtest GC=F    # Run backtest
  python main.py --train            # (Re)train all ML models
"""
from __future__ import annotations
import argparse
import signal as _signal
import time
import os
import yaml
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

SLIPPAGE_PCT = 0.0008   # 0.08% per trade

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

from utils.logger import logger, setup_logger
setup_logger(CFG["logging"]["level"])

# ── Connectors ────────────────────────────────────────────────────────────────
from data.connectors.market_data import MarketDataConnector
from data.connectors.market_microstructure import MicrostructureConnector
from data.connectors.catalyst_data import CatalystConnector
from data.connectors.sentiment_data import SentimentConnector
from data.connectors.scanner import MarketWatcher

# ── Processing ────────────────────────────────────────────────────────────────
from data.preprocessor import Preprocessor
from features.feature_engine import FeatureEngine

# ── Analysis ──────────────────────────────────────────────────────────────────
from analysis.technical import TechnicalAnalyzer
from analysis.statistical import StatisticalAnalyzer
from analysis.fundamental import FundamentalAnalyzer

# ── Models ────────────────────────────────────────────────────────────────────
from models.ml_model import MLPredictor
from models.ensemble import EnsemblePredictor
from models.nlp_sentiment import NLPSentimentAnalyzer
from models.gemini_analyzer import GeminiAnalyzer
from models.quantum_optimizer import QuantumOptimizer
from models.signal_fusion import SignalFusion

# ── Risk & dispatch ───────────────────────────────────────────────────────────
from strategies.strategy_selector import StrategySelector
from strategies.day_trading_forex  import DayTradingStrategy, DayTradingSignal
from utils.risk_manager    import RiskManager
from utils.kill_switch     import KillSwitch
from utils.pnl_journal     import PnLJournal
from utils.paper_broker    import PaperBroker
from utils.adaptive_params import AdaptiveParams
from analysis.trade_analyzer import TradeAnalyzer
from alerts.bot import TelegramAlerter
from backtesting.engine import BacktestEngine


# ── Feature columns for LSTM ──────────────────────────────────────────────────
LSTM_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "log_ret", "ret_5", "ret_20", "vol_20", "atr", "vol_ratio",
    "rsi_14", "macd_hist", "bb_pct", "gap_pct", "vwap_dev",
]


# ── Component initialisation ──────────────────────────────────────────────────
def build_components() -> dict:
    return {
        "market":       MarketDataConnector(),
        "micro":        MicrostructureConnector(),
        "catalyst":     CatalystConnector(),
        "sentiment":    SentimentConnector(),
        "preprocessor": Preprocessor(lookback=CFG["ml"]["lookback_window"]),
        "features":     FeatureEngine(),
        "ta":           TechnicalAnalyzer(),
        "stat":         StatisticalAnalyzer(),
        "fund":         FundamentalAnalyzer(),
        "lstm":         MLPredictor(CFG),
        "ensemble":     EnsemblePredictor(),
        "nlp":          NLPSentimentAnalyzer(),
        "gemini":       GeminiAnalyzer(CFG),
        "qopt":         QuantumOptimizer(CFG),
        "fusion":       SignalFusion(CFG, QuantumOptimizer(CFG)),
        "selector":     StrategySelector(CFG),
        # Three independent day trading strategies (separate daily counters each)
        "nasdaq_dt":    DayTradingStrategy(CFG, "nasdaq"),  # QQQ
        "spy_dt":       DayTradingStrategy(CFG, "spy"),     # SPY
        "nq_dt":        DayTradingStrategy(CFG, "nq"),      # NQ=F
        "paper_broker":    PaperBroker(CFG),
        "adaptive_params": AdaptiveParams(),
        "trade_analyzer":  TradeAnalyzer(CFG),
        "risk":            RiskManager(CFG),
        "kill":         KillSwitch(
            max_daily_loss_pct=CFG["risk"].get("max_daily_drawdown_pct", 3.0) / 100,
            max_daily_trades=CFG["risk"].get("max_signals_per_day", 12),
        ),
        "telegram":     TelegramAlerter(CFG),
        "scanner":      MarketWatcher(CFG),
        "journal":      PnLJournal(),
    }


# ── Single-asset analysis ─────────────────────────────────────────────────────
def analyse_asset(
    ticker: str,
    C: dict,
    paper_mode: bool = False,
    signal_dedup: dict | None = None,
    dedup_seconds: float = 4 * 3600,
) -> tuple[bool, float]:
    """Returns (signal_generated, composite_score)."""
    # ── 1. Daily data ─────────────────────────────────────────────────────────
    df_daily = C["market"].get_ohlcv(ticker, "1d", "30d")
    if df_daily is None or len(df_daily) < 2:
        logger.warning(f"{ticker}: insufficient daily data")
        return False, 0.0
    strategy, timeframe = C["selector"].select(df_daily)

    daily_ret = np.log(df_daily["close"] / df_daily["close"].shift(1)).dropna()
    historical_sharpe = (
        float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))
        if len(daily_ret) >= 5 else 0.0
    )

    # ── 2. Intraday data ──────────────────────────────────────────────────────
    df_raw = C["market"].get_ohlcv(ticker, timeframe, "60d")
    if df_raw is None or len(df_raw) < 80:
        logger.warning(f"{ticker}: insufficient intraday data")
        return False, 0.0
    df = C["preprocessor"].transform(df_raw)
    if len(df) < 60:
        logger.warning(f"{ticker}: insufficient data after preprocessing")
        return False, 0.0

    # ── 3. Microstructure ─────────────────────────────────────────────────────
    micro = C["micro"].get(ticker)

    # ── 4. Catalyst ───────────────────────────────────────────────────────────
    catalyst = C["catalyst"].get(ticker)

    # ── 5. Sentiment ──────────────────────────────────────────────────────────
    sentiment = C["sentiment"].compute(ticker, df, micro)

    # ── 6. Feature engineering ────────────────────────────────────────────────
    feat = C["features"].compute(df, df_daily, micro, catalyst, sentiment)

    # ── 7. Technical + Statistical + Fundamental ──────────────────────────────
    ta_sig   = C["ta"].get_signal(df)
    stat_sig = C["stat"].get_signal(df)
    fund_sig = C["fund"].get_signal(ticker)

    # ── 8. LSTM prediction ────────────────────────────────────────────────────
    available_cols = [c for c in LSTM_FEATURE_COLS if c in df.columns]
    _thr = CFG["ml"].get("label_threshold", 0.003)
    _hor = CFG["ml"].get("label_horizon", 3)
    X, _ = C["preprocessor"].build_sequences(df, available_cols, threshold=_thr, horizon=_hor)
    if len(X) == 0:
        logger.warning(f"{ticker}: not enough data for LSTM sequence")
        return False, 0.0
    lstm_dir, lstm_conf = C["lstm"].predict(X[-1:])

    # ── 9. Ensemble prediction (XGB + LGB) ────────────────────────────────────
    ensemble_pred = C["ensemble"].predict(feat.array)

    # ── 10. NLP sentiment — Gemini Flash ─────────────────────────────────────
    news_texts    = C["sentiment"].get_news_texts(ticker)
    gemini_result = C["gemini"].analyze(ticker, news_texts)
    nlp_score     = float(gemini_result.sentiment_score)
    _catalyst_mult = C["gemini"].catalyst_confidence_boost(gemini_result)
    if gemini_result.catalyst_type != "none":
        logger.info(
            f"{ticker}: catalyst={gemini_result.catalyst_type} "
            f"nlp={nlp_score:+.2f} boost={_catalyst_mult:.2f}x "
            f"— {gemini_result.summary}"
        )

    # ── 11. Signal fusion ─────────────────────────────────────────────────────
    price = float(df["close"].iloc[-1])
    signal = C["fusion"].fuse(
        ticker=ticker,
        ta_signal=ta_sig,
        ml_direction=lstm_dir,
        ml_confidence=lstm_conf,
        stat_signal=stat_sig,
        fund_signal=fund_sig,
        price=price,
        timeframe=timeframe,
        strategy=strategy,
        historical_sharpe=historical_sharpe,
        ensemble_pred=ensemble_pred,
        nlp_score=nlp_score,
        features=feat,
    )

    if signal is None:
        logger.info(f"{ticker}: no actionable signal")
        return False, 0.0

    if _catalyst_mult != 1.0:
        signal.confidence = float(min(0.99, signal.confidence * _catalyst_mult))

    # ── 12. Risk gate ─────────────────────────────────────────────────────────
    approved, adjusted_stop, reason = C["risk"].approve(
        confidence=signal.confidence,
        sharpe=signal.sharpe_estimate,
        drawdown=signal.drawdown_estimate,
        price=price,
        atr=ta_sig.atr,
        direction=signal.direction,
        spread_pct=micro.spread_pct,
        avg_volume=micro.avg_volume_30d,
        liquidity_score=micro.liquidity_score,
        rvol=feat.rvol,
        fake_breakout_prob=signal.fake_breakout_prob,
        volatility_percentile=feat.volatility_percentile,
        catalyst_score=catalyst.catalyst_score,
        has_recent_earnings=catalyst.has_recent_earnings,
        sec_8k_score=catalyst.sec_8k_score,
    )
    if not approved:
        logger.info(f"{ticker}: signal blocked by risk gate — {reason}")
        return False, signal.composite_score

    signal.stop_loss = adjusted_stop

    # ── 13. Kill switch ───────────────────────────────────────────────────────
    alive, ks_reason = C["kill"].check()
    if not alive:
        logger.critical(f"Kill switch active — {ks_reason}")
        return False, signal.composite_score

    # ── 14. Signal dedup ──────────────────────────────────────────────────────
    dedup_key = f"{ticker}:{signal.direction}"
    if signal_dedup is not None:
        last_sent = signal_dedup.get(dedup_key, 0.0)
        if time.time() - last_sent < dedup_seconds:
            logger.info(f"{ticker}: duplicate signal suppressed (same dir within {dedup_seconds/3600:.0f}h)")
            return False, signal.composite_score

    # ── 15. Dispatch ──────────────────────────────────────────────────────────
    slip_factor     = (1 + SLIPPAGE_PCT) if signal.direction == "BUY" else (1 - SLIPPAGE_PCT)
    entry_with_slip = price * slip_factor

    if paper_mode:
        logger.info(
            f"[PAPER] {ticker} {signal.direction} @ {entry_with_slip:.4f} "
            f"(slip={SLIPPAGE_PCT:.2%}) "
            f"conf={signal.confidence:.1%} "
            f"stop={signal.stop_loss:.4f} tp={signal.take_profit:.4f} "
            f"pos={signal.position_size_pct:.2%}"
        )
        C["telegram"].send_signal(signal, paper=True, entry_price=entry_with_slip)
    else:
        C["telegram"].send_signal(signal, paper=False, entry_price=entry_with_slip)

    if signal_dedup is not None:
        signal_dedup[dedup_key] = time.time()

    C["journal"].record_signal(
        ticker=ticker,
        direction=signal.direction,
        entry_price=entry_with_slip,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        strategy=signal.strategy,
        confidence=signal.confidence,
        position_size_pct=signal.position_size_pct,
    )

    if not paper_mode:
        C["kill"].record_trade(pnl_pct=0.0)
    return True, signal.composite_score


# ── ML training ───────────────────────────────────────────────────────────────
def train_models(C: dict) -> None:
    all_tickers = C["scanner"].get_candidates() if "scanner" in C else \
        CFG["assets"].get("commodities", []) + CFG["assets"].get("equities", [])
    X_lstm_all, y_lstm_all = [], []
    X_ens_all, y_ens_all = [], []

    for ticker in all_tickers:
        try:
            df_raw = C["market"].get_ohlcv(ticker, "1h", "max")
            df = C["preprocessor"].transform(df_raw)
            df_daily = C["market"].get_ohlcv(ticker, "1d", "60d")
            micro = C["micro"].get(ticker)

            available_cols = [c for c in LSTM_FEATURE_COLS if c in df.columns]
            _thr = CFG["ml"].get("label_threshold", 0.003)
            _hor = CFG["ml"].get("label_horizon", 3)
            X, y = C["preprocessor"].build_sequences(df, available_cols, threshold=_thr, horizon=_hor)
            if len(X) > 0:
                X_lstm_all.append(X)
                y_lstm_all.append(y)

            feat = C["features"].compute(df, df_daily, micro)
            closes = df["close"].values
            labels = np.array([1 if closes[i + 1] > closes[i] else 0
                               for i in range(len(closes) - 1)])
            feat_rows = []
            for i in range(len(df) - 1):
                sub_df = df.iloc[max(0, i - 60):i + 1]
                sub_daily = df_daily.iloc[-30:] if len(df_daily) >= 30 else df_daily
                f = C["features"].compute(sub_df, sub_daily, micro)
                feat_rows.append(f.array)
            if feat_rows:
                X_ens_all.append(np.array(feat_rows))
                y_ens_all.append(labels[:len(feat_rows)])

        except Exception as e:
            logger.error(f"Train data {ticker}: {e}")

    if X_lstm_all:
        X_cat = np.concatenate(X_lstm_all)
        y_cat = np.concatenate(y_lstm_all)
        logger.info(f"Training LSTM on {len(X_cat)} sequences")
        metrics = C["lstm"].train(X_cat, y_cat)
        logger.info(f"LSTM training complete: {metrics}")
    else:
        logger.error("No LSTM training data collected")

    if X_ens_all:
        X_e = np.concatenate(X_ens_all).astype(np.float32)
        y_e = np.concatenate(y_ens_all)
        _LIVE_FEATURES = [3, 6, 7, 8, 9, 12, 13, 14]
        if X_e.shape[1] > max(_LIVE_FEATURES):
            X_e = X_e[:, _LIVE_FEATURES]
        logger.info(f"Training ensemble on {len(X_e)} samples")
        results = C["ensemble"].train(X_e, y_e)
        logger.info(f"Ensemble training: {results}")
    else:
        logger.warning("No ensemble training data — skipped")

    _STAMP = os.path.join(os.path.dirname(__file__), "models", "last_train.txt")
    try:
        open(_STAMP, "w").write(str(time.time()))
    except Exception:
        pass


# ── DayTradingSignal → TradeSignal adapter ────────────────────────────────────
def _daytrading_to_trade(sig: DayTradingSignal) -> "TradeSignal":
    from models.signal_fusion import TradeSignal
    rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
    return TradeSignal(
        ticker            = sig.ticker,
        direction         = sig.direction,
        confidence        = sig.confidence,
        price             = sig.entry,
        stop_loss         = sig.stop_loss,
        take_profit       = sig.take_profit,
        timeframe         = "1h",
        strategy          = f"daytrading_{sig.setup_type}_{sig.session}",
        sharpe_estimate   = round(sig.confidence * 2.0, 2),
        drawdown_estimate = round((1 - sig.confidence) * 0.05, 4),
        quantum_up_prob   = 0.5,
        fake_breakout_prob= 0.0,
        squeeze_prob      = 0.0,
        volatility_expected= sig.atr_1h / max(sig.entry, 1e-9),
        position_size_pct = round(sig.risk_amount / 8871.0, 4),
        composite_score   = sig.confidence,
        breakdown         = {
            **sig.breakdown,
            "setup":      sig.setup_type,
            "confluence": sig.confluence_score,
            "session":    sig.session,
            "poc_4h":     sig.poc_4h,
            "poc_1h":     sig.poc_1h,
            "vah_4h":     sig.vah_4h,
            "val_4h":     sig.val_4h,
            "vwap_4h":    sig.vwap_4h,
            "vwap_1h":    sig.vwap_1h,
            "cvd_bias":   sig.cvd_bias_4h,
            "of_score":   sig.orderflow_score,
            "lot_size":   sig.lot_size,
            "pip_risk":   sig.pip_risk,
            "rr_actual":  round(rr, 2),
            "ta_dir":     sig.direction,
            "ml_dir":     sig.direction,
            "stat_dir":   sig.direction,
            "fund_dir":   sig.direction,
            "rsi":        50.0,
            "hurst":      0.5,
            "z_score":    0.0,
            "sentiment":  0.0,
        },
    )


# ── Generic day trading cycle (Gold / Nasdaq / SPY / NQ=F) ───────────────────
def run_dt_cycle(
    C: dict,
    strategy_key: str,      # "gold" | "nasdaq" | "spy" | "nq"
    instance_key: str,      # "gold_dt" | "nasdaq_dt" | "spy_dt" | "nq_dt"
    label: str,             # display label for logs
    paper: bool = False,
) -> int:
    """Generic day trading cycle. Returns number of signals dispatched."""
    strategy: DayTradingStrategy = C[instance_key]

    if not strategy.can_trade_today():
        logger.debug(f"{label}: daily limit reached — skip")
        return 0

    st_cfg = CFG.get("strategies", {}).get(strategy_key, {})
    ticker = st_cfg.get("ticker", "")
    if not ticker:
        return 0

    # DXY for Gold (optional, controlled by use_dxy flag)
    df_dxy_1h = None
    if st_cfg.get("use_dxy", False):
        try:
            df_dxy_1h = C["market"].get_ohlcv("DX-Y.NYB", "1h", "10d")
            if df_dxy_1h is not None and len(df_dxy_1h) < 8:
                df_dxy_1h = None
        except Exception as _e:
            logger.warning(f"{label}: DXY fetch failed ({_e})")

    exec_tf = st_cfg.get("timeframes", {}).get("execution", "1h")
    # yfinance: 15m max period=60d, 1h max period=730d
    exec_period = "60d" if exec_tf == "15m" else "30d"

    dispatched = 0
    try:
        df_4h = C["market"].get_ohlcv(ticker, "4h", "60d")
        df_1h = C["market"].get_ohlcv(ticker, exec_tf, exec_period)
        if df_4h is None or df_1h is None or len(df_4h) < 25 or len(df_1h) < 20:
            return 0

        sig = strategy.analyse(ticker, df_4h, df_1h, df_dxy_1h=df_dxy_1h)
        if sig is None:
            return 0

        trade_sig = _daytrading_to_trade(sig)
        if paper:
            pos = C["paper_broker"].execute(sig, "day_trading")
            if pos is None:
                return 0
            logger.info(
                f"[PAPER EXEC][{label}] {ticker} {sig.direction} "
                f"fill={pos.entry_price:.5f} (slip={pos.slippage_pct:.3%}) "
                f"SL={sig.stop_loss:.5f}  TP={sig.take_profit:.5f} "
                f"session={sig.session}  lot={sig.lot_size}  id={pos.id}"
            )
        C["telegram"].send_signal(trade_sig, paper=paper, entry_price=sig.entry)
        C["journal"].record_signal(
            ticker           = sig.ticker,
            direction        = sig.direction,
            entry_price      = sig.entry,
            stop_loss        = sig.stop_loss,
            take_profit      = sig.take_profit,
            strategy         = trade_sig.strategy,
            confidence       = sig.confidence,
            position_size_pct= trade_sig.position_size_pct,
        )
        strategy.record_trade()
        dispatched = 1
        logger.info(
            f"{label} [{ticker}]: {sig.direction} {sig.setup_type} "
            f"session={sig.session} conf={sig.confluence_score} "
            f"pip_risk={sig.pip_risk:.1f}"
        )
    except Exception as e:
        logger.error(f"{label} [{ticker}]: {e}")

    return dispatched


# Convenience wrappers (readability in run_live)
def run_gold_cycle(C, paper=False):   return run_dt_cycle(C, "gold",   "gold_dt",   "GOLD",   paper)
def run_nasdaq_cycle(C, paper=False): return run_dt_cycle(C, "nasdaq", "nasdaq_dt", "NASDAQ", paper)
def run_spy_cycle(C, paper=False):    return run_dt_cycle(C, "spy",    "spy_dt",    "SPY",    paper)
def run_nq_cycle(C, paper=False):     return run_dt_cycle(C, "nq",     "nq_dt",     "NQ",     paper)


# ── Adaptive feedback: analyse losses → patch strategies ──────────────────────
def run_trade_analysis(C: dict, paper: bool = True) -> None:
    analyzer: TradeAnalyzer  = C["trade_analyzer"]
    adaptive: AdaptiveParams = C["adaptive_params"]

    report = analyzer.analyse(n_recent=100)

    if not report.strategies:
        logger.info("TradeAnalyzer: insufficient data — skipping")
        return

    if report.adjustments:
        adaptive.apply(report.adjustments, expires_in_days=7)

    for strategy_type, strategy_obj in [
        ("day_trading", C["nasdaq_dt"]),
        ("day_trading", C["spy_dt"]),
        ("day_trading", C["nq_dt"]),
    ]:
        adaptive.patch_strategy(strategy_obj, strategy_type)

    logger.info(adaptive.summary())

    msg = analyzer.format_telegram_report(report)
    try:
        C["telegram"]._send_raw(msg)
    except Exception as e:
        logger.warning(f"TradeAnalyzer Telegram: {e}")


# ── Paper broker update cycle ─────────────────────────────────────────────────
def run_paper_broker_update(C: dict, paper: bool = True) -> int:
    """Returns number of positions closed this cycle."""
    broker: PaperBroker = C["paper_broker"]
    open_pos = broker.open_positions()
    if not open_pos:
        return 0

    TF_MAP = {
        "day_trading": ("1h", "7d"),
    }
    ohlcv_map: dict[str, dict] = {}
    fetched: set[str] = set()
    for pos in open_pos:
        key = f"{pos.ticker}:{pos.strategy_type}"
        if key in fetched:
            continue
        tf, period = TF_MAP.get(pos.strategy_type, ("1h", "7d"))
        try:
            df = C["market"].get_ohlcv(pos.ticker, tf, period)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                ohlcv_map[pos.ticker] = {
                    "high":  float(last["high"]),
                    "low":   float(last["low"]),
                    "close": float(last["close"]),
                }
        except Exception as e:
            logger.warning(f"PaperBroker update: OHLCV fetch {pos.ticker} [{tf}]: {e}")
        fetched.add(key)

    if not ohlcv_map:
        return 0

    closed = broker.update(ohlcv_map=ohlcv_map, telegram=C["telegram"], paper=paper)

    for pos in closed:
        C["kill"].record_trade(pnl_pct=pos.pnl_pct)
        sign = "+" if pos.pnl_usd >= 0 else ""
        logger.info(
            f"[PAPER CLOSE] {pos.strategy_type} {pos.direction} {pos.ticker} "
            f"→ {sign}{pos.pnl_usd:.2f}$ ({pos.r_multiple:+.2f}R) [{pos.exit_reason}]"
        )

    if closed:
        summary = broker.equity_summary()
        for st, eq in summary.items():
            if eq["trades"] > 0:
                logger.info(
                    f"[PAPER EQUITY] {st}: capital={eq['capital']:.2f}$ "
                    f"roi={eq['roi_pct']:+.2f}%  WR={eq['win_rate']:.1f}%  "
                    f"DD={eq['max_dd_usd']:.2f}$"
                )

    return len(closed)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_live(paper: bool = False) -> None:
    C        = build_components()
    mode_str = "PAPER" if paper else "LIVE"
    logger.info(f"AlgoTrad starting — mode={mode_str} — strategies: Gold (GC=F) + Nasdaq (QQQ)")

    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt
    _signal.signal(_signal.SIGTERM, _sigterm_handler)

    C["telegram"].send_startup(mode_str)

    use_scanner   = CFG.get("scanner", {}).get("enabled", False)
    retrain_every = CFG["ml"]["retrain_interval_hours"] * 3600
    signal_dedup: dict[str, float] = {}
    DEDUP_SECONDS = 4 * 3600
    _STAMP = os.path.join(os.path.dirname(__file__), "models", "last_train.txt")
    try:
        last_train = float(open(_STAMP).read().strip())
    except Exception:
        last_train = 0.0
    last_day            = -1
    _cycles             = 0
    _signals_today      = 0
    _last_heartbeat_day = -1
    _cycle_prices: dict[str, float] = {}
    _last_analysis_day  = -1
    _closed_trades_since_analysis = 0

    # Apply existing adaptive adjustments at startup
    adaptive: AdaptiveParams = C["adaptive_params"]
    for _st, _so in [
        ("day_trading", C["nasdaq_dt"]),
        ("day_trading", C["spy_dt"]),
        ("day_trading", C["nq_dt"]),
    ]:
        adaptive.patch_strategy(_so, _st)
    logger.info(f"Startup adaptive params: {adaptive.summary()}")

    try:
      while True:
        import datetime
        today = datetime.date.today().toordinal()
        if today != last_day:
            C["kill"].reset_daily()
            _signals_today = 0
            last_day = today

        # Daily heartbeat
        if today != _last_heartbeat_day:
            pnl_stats = C["journal"].get_stats()
            C["telegram"].send_daily_summary(
                signals_today=_signals_today,
                cycles=_cycles,
                pnl_stats=pnl_stats,
                mode=mode_str,
            )
            if paper:
                eq_summary = C["paper_broker"].equity_summary()
                lines = ["📊 *Paper Equity — QQQ + SPY + NQ*\n━━━━━━━━━━━━━━━━━━━━━━"]
                labels = {"day_trading": "Day Trading (QQQ/SPY/NQ=F)"}
                for st, eq in eq_summary.items():
                    sign = "+" if eq["roi_pct"] >= 0 else ""
                    lines.append(
                        f"*{labels.get(st, st)}*\n"
                        f"  Capital : `{eq['capital']:.2f}$`  ROI : `{sign}{eq['roi_pct']:.2f}%`\n"
                        f"  Trades : `{eq['trades']}`  WR : `{eq['win_rate']:.1f}%`  DD : `{eq['max_dd_usd']:.2f}$`"
                    )
                C["telegram"]._send_raw("\n".join(lines))
            _last_heartbeat_day = today

        # Kill switch check
        alive, reason = C["kill"].check()
        if not alive:
            logger.critical(f"Kill switch — {reason}. Sleeping until next cycle.")
            time.sleep(300)
            continue

        # Periodic ML retraining
        if time.time() - last_train > retrain_every:
            logger.info("Scheduled ML retraining...")
            C["telegram"].send_status("🔄 Ré-entraînement ML démarré — analyse suspendue ~20min")
            train_models(C)
            last_train = time.time()
            try:
                open(_STAMP, "w").write(str(last_train))
            except Exception:
                pass
            C["telegram"].send_status("✅ Ré-entraînement ML terminé — reprise de l'analyse")

        # Ticker list: Gold + Nasdaq
        if use_scanner:
            all_tickers = C["scanner"].get_candidates()
            if not all_tickers:
                logger.warning("Scanner returned 0 candidates — skipping cycle")
                C["telegram"].send_error("Scanner: 0 candidats — cycle ignoré")
                time.sleep(300)
                continue
        else:
            all_tickers = (
                CFG["assets"].get("commodities", []) +
                CFG["assets"].get("equities", [])
            )

        # Per-ticker ML/fusion analysis
        cycle_scores: dict[str, float] = {}
        _cycle_errors = 0
        _cycle_prices.clear()
        for ticker in all_tickers:
            try:
                generated, score = analyse_asset(
                    ticker, C, paper_mode=paper,
                    signal_dedup=signal_dedup,
                    dedup_seconds=DEDUP_SECONDS,
                )
                if generated:
                    _signals_today += 1
                if score > 0:
                    cycle_scores[ticker] = score
                try:
                    _df = C["market"].get_ohlcv(ticker, "1d", "5d")
                    if _df is not None and len(_df) > 0:
                        _cycle_prices[ticker] = float(_df["close"].iloc[-1])
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Error analysing {ticker}: {e}")
                _cycle_errors += 1
            time.sleep(2)

        # Journal: update open positions
        if _cycle_prices:
            closed_trades = C["journal"].update_positions(_cycle_prices)
            for trade in closed_trades:
                C["kill"].record_trade(pnl_pct=trade["pnl_pct"])
                sign = "+" if trade["pnl_pct"] >= 0 else ""
                logger.info(
                    f"Position closed: {trade['direction']} {trade['ticker']} "
                    f"{sign}{trade['pnl_pct']:.2%} ({trade['exit_reason']})"
                )

        if _cycle_errors >= len(all_tickers):
            C["telegram"].send_error(
                f"Cycle dégradé: {_cycle_errors}/{len(all_tickers)} tickers en erreur",
                critical=True,
            )

        # Multi-asset quantum ranking
        if cycle_scores:
            top = C["qopt"].rank_signals(cycle_scores, top_k=3)
            logger.info(f"Top ranked setups this cycle: {top}")

        # ── Strategy 1: Day Trading Nasdaq QQQ ────────────────────────────────
        if CFG.get("strategies", {}).get("nasdaq", {}).get("enabled", False):
            try:
                _signals_today += run_nasdaq_cycle(C, paper=paper)
            except Exception as e:
                logger.error(f"Nasdaq DT cycle error: {e}")

        # ── Strategy 3: Day Trading SPY ────────────────────────────────────────
        if CFG.get("strategies", {}).get("spy", {}).get("enabled", False):
            try:
                _signals_today += run_spy_cycle(C, paper=paper)
            except Exception as e:
                logger.error(f"SPY DT cycle error: {e}")

        # ── Strategy 4: Day Trading NQ=F ───────────────────────────────────────
        if CFG.get("strategies", {}).get("nq", {}).get("enabled", False):
            try:
                _signals_today += run_nq_cycle(C, paper=paper)
            except Exception as e:
                logger.error(f"NQ DT cycle error: {e}")

        # ── Paper broker: check open positions for TP/SL ──────────────────────
        if paper:
            try:
                closed_this_cycle = run_paper_broker_update(C, paper=True)
                _closed_trades_since_analysis += closed_this_cycle
            except Exception as e:
                logger.error(f"Paper broker update error: {e}")

        # ── Trade analysis: every 10 closes OR once per day ───────────────────
        if _closed_trades_since_analysis >= 10 or today != _last_analysis_day:
            try:
                run_trade_analysis(C, paper=paper)
                _last_analysis_day = today
                _closed_trades_since_analysis = 0
            except Exception as e:
                logger.error(f"Trade analysis error: {e}")

        _cycles += 1
        logger.info("Cycle complete — waiting 5 minutes")
        time.sleep(300)

    except KeyboardInterrupt:
        logger.info("AlgoTrad arrêté par l'utilisateur (Ctrl+C)")
        C["telegram"].send_shutdown(
            reason="Arrêt manuel (Ctrl+C)",
            signals_today=_signals_today,
            cycles=_cycles,
        )
    except Exception as e:
        logger.critical(f"AlgoTrad crash inattendu: {e}")
        C["telegram"].send_shutdown(
            reason=f"CRASH: {str(e)[:200]}",
            signals_today=_signals_today,
            cycles=_cycles,
        )
        C["telegram"].send_error(str(e), critical=True)
        raise


def run_backtest(ticker: str) -> None:
    day_trade_tickers = (
        CFG["assets"].get("commodities", []) +
        CFG["assets"].get("equities", [])
    )
    if ticker in day_trade_tickers:
        from backtesting.backtest_daytrading import BacktestDayTrading
        _ticker_map = {"QQQ": "nasdaq", "SPY": "spy", "NQ=F": "nq"}
        strategy_key = _ticker_map.get(ticker, "nasdaq")
        results = BacktestDayTrading(CFG).run(ticker, strategy_key=strategy_key, plot=True)
    else:
        results = BacktestEngine(CFG).run(ticker, plot=True)
    print("\n" + "=" * 50)
    print(f"BACKTEST: {ticker}")
    for k, v in results.items():
        print(f"  {k:<24}: {v}")
    print("=" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlgoTrad — Gold + Nasdaq Day Trading")
    parser.add_argument("--backtest", metavar="TICKER")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.backtest)
    elif args.train:
        train_models(build_components())
    else:
        run_live(paper=args.paper)
