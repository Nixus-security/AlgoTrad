"""
AlgoTrad — Main entry point. Crypto mode: BTC-USD + SOL-USD, 24/7.
Pipeline per asset:
  market data → microstructure → catalyst → sentiment → features
  → LSTM + Ensemble(XGB/LGB) + NLP
  → Quantum weight optimisation + feature selection
  → Signal fusion → Risk gate → Kill switch
  → Quantum multi-asset ranking + position sizing → Dispatch

Usage:
  python main.py                    # Live signal generation loop
  python main.py --paper            # Paper-trading dry-run (signals to console)
  python main.py --backtest BTC-USD # Run backtest
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

# ── Slippage simulation (paper mode) ─────────────────────────────────────────
SLIPPAGE_PCT = 0.0008   # 0.08% par trade — realistic pour crypto spot

# ── Régime BTC — cache 30 min ─────────────────────────────────────────────────
# BTC trend used as regime indicator for both BTC and SOL (high correlation).
_btc_cache: dict = {"ts": 0.0, "above_ma20": True}
_BTC_CACHE_SEC = 1800  # 30 min


def _btc_regime() -> bool:
    """Retourne True si BTC-USD au-dessus de sa MA20 (régime haussier crypto)."""
    global _btc_cache
    if time.time() - _btc_cache["ts"] < _BTC_CACHE_SEC:
        return _btc_cache["above_ma20"]
    try:
        btc = yf.download("BTC-USD", period="30d", interval="1d",
                          auto_adjust=True, progress=False)
        if btc.empty or len(btc) < 20:
            logger.warning("BTC regime: insufficient data — fail-closed (no signal)")
            return False   # fail-closed: block signals on uncertainty
        close = btc["Close"].squeeze()
        above = bool(float(close.iloc[-1]) > float(close.rolling(20).mean().iloc[-1]))
        _btc_cache = {"ts": time.time(), "above_ma20": above}
        logger.info(f"Régime BTC: {'HAUSSIER' if above else 'BAISSIER'} "
                    f"(price={close.iloc[-1]:.2f} vs MA20={close.rolling(20).mean().iloc[-1]:.2f})")
        return above
    except Exception as _e:
        logger.warning(f"BTC regime fetch failed ({_e}) — fail-closed (no signal)")
        return False   # fail-closed: never trade on yfinance error

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
from data.connectors.scanner import CryptoWatcher

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
from strategies.strategy_selector  import StrategySelector
from strategies.swing_trading      import SwingTradingStrategy, SwingSignal
from strategies.day_trading_forex  import DayTradingForexStrategy, DayTradingSignal
from strategies.scalping_hfq       import ScalpingHFQStrategy, ScalpSignal
from utils.risk_manager     import RiskManager
from utils.kill_switch      import KillSwitch
from utils.pnl_journal      import PnLJournal
from utils.paper_broker     import PaperBroker
from utils.adaptive_params  import AdaptiveParams
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
        # ── Three new strategies ───────────────────────────────────────────
        "swing":        SwingTradingStrategy(CFG),
        "daytrading":   DayTradingForexStrategy(CFG),
        "scalping_hfq": ScalpingHFQStrategy(CFG),
        # Paper broker: simulates execution (slippage, TP/SL, timeout, equity)
        "paper_broker":    PaperBroker(CFG),
        # Adaptive feedback loop: loss analysis → parameter adjustments
        "adaptive_params": AdaptiveParams(),
        "trade_analyzer":  TradeAnalyzer(CFG),
        "risk":            RiskManager(CFG),
        "kill":         KillSwitch(
            # settings.yaml: max_daily_drawdown_pct = 3.0 (%) → divide by 100 → 0.03
            # Default 3.0 not 0.02 — 0.02/100=0.0002% would trigger at $1.77 loss
            max_daily_loss_pct=CFG["risk"].get("max_daily_drawdown_pct", 3.0) / 100,
            max_daily_trades=CFG["risk"].get("max_signals_per_day", 20),
        ),
        "telegram":     TelegramAlerter(CFG),
        "scanner":      CryptoWatcher(CFG),
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
    """
    Returns (signal_generated, composite_score).
    score used for multi-stock ranking.
    """
    # ── 1. Daily data — strategy selection + Sharpe ───────────────────────────
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

    # ── 5. Sentiment (StockTwits + computed metrics) ──────────────────────────
    sentiment = C["sentiment"].compute(ticker, df, micro)

    # ── 6. Feature engineering ────────────────────────────────────────────────
    feat = C["features"].compute(df, df_daily, micro, catalyst, sentiment)

    # ── 7. Technical + Statistical + Fundamental signals ─────────────────────
    ta_sig = C["ta"].get_signal(df)
    stat_sig = C["stat"].get_signal(df)
    fund_sig = C["fund"].get_signal(ticker)

    # ── 8. LSTM prediction ────────────────────────────────────────────────────
    available_cols = [c for c in LSTM_FEATURE_COLS if c in df.columns]
    _thr = CFG["ml"].get("label_threshold", 0.003)
    _hor = CFG["ml"].get("label_horizon", 3)
    X, _ = C["preprocessor"].build_sequences(df, available_cols,
                                              threshold=_thr, horizon=_hor)
    if len(X) == 0:
        logger.warning(f"{ticker}: not enough data for LSTM sequence")
        return False, 0.0
    lstm_dir, lstm_conf = C["lstm"].predict(X[-1:])

    # ── 9. Ensemble prediction (XGB + LGB) ────────────────────────────────────
    ensemble_pred = C["ensemble"].predict(feat.array)

    # ── 10. NLP sentiment — Gemini Flash + Google News RSS + Reddit ──────────
    news_texts    = C["sentiment"].get_news_texts(ticker)
    gemini_result = C["gemini"].analyze(ticker, news_texts)
    # Score Gemini en [-1,+1] — complété par keyword NLP si pas de clé API
    nlp_score     = float(gemini_result.sentiment_score)
    # Boost/réduction confiance selon catalyst détecté
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

    # Applique boost catalyst Gemini sur la confidence finale
    if _catalyst_mult != 1.0:
        signal.confidence = float(min(0.99, signal.confidence * _catalyst_mult))

    # ── 12. Régime marché (BTC MA20) — bloque BUY en régime baissier crypto ────
    btc_bull = _btc_regime()
    if not btc_bull and signal.direction == "BUY":
        logger.info(f"{ticker}: BUY bloqué — BTC régime BAISSIER (sous MA20)")
        return False, signal.composite_score

    # ── 13. Risk gate ─────────────────────────────────────────────────────────
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

    # ── 14. Signal dedup — skip if same ticker+direction dispatched recently ──
    dedup_key = f"{ticker}:{signal.direction}"
    if signal_dedup is not None:
        last_sent = signal_dedup.get(dedup_key, 0.0)
        if time.time() - last_sent < dedup_seconds:
            logger.info(f"{ticker}: duplicate signal suppressed (same dir within {dedup_seconds/3600:.0f}h)")
            return False, signal.composite_score

    # ── 15. Dispatch ──────────────────────────────────────────────────────────
    # Slippage réaliste : BUY paie plus cher, SELL reçoit moins
    slip_factor     = (1 + SLIPPAGE_PCT) if signal.direction == "BUY" else (1 - SLIPPAGE_PCT)
    entry_with_slip = price * slip_factor

    if paper_mode:
        logger.info(
            f"[PAPER] {ticker} {signal.direction} @ {entry_with_slip:.4f} "
            f"(slip={SLIPPAGE_PCT:.2%}) "
            f"conf={signal.confidence:.1%} "
            f"stop={signal.stop_loss:.4f} tp={signal.take_profit:.4f} "
            f"pos={signal.position_size_pct:.2%} "
            f"fake_bk={signal.fake_breakout_prob:.0%} "
            f"squeeze={signal.squeeze_prob:.0%}"
        )
        # Envoi Telegram en mode paper (message clairement labellé [PAPER])
        C["telegram"].send_signal(signal, paper=True, entry_price=entry_with_slip)
    else:
        C["telegram"].send_signal(signal, paper=False, entry_price=entry_with_slip)

    if signal_dedup is not None:
        signal_dedup[dedup_key] = time.time()

    # ── Journal: record open position ─────────────────────────────────────────
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
        C["kill"].record_trade(pnl_pct=0.0)  # actual P&L wired in when live broker added
    return True, signal.composite_score


# ── ML training ───────────────────────────────────────────────────────────────
def train_models(C: dict) -> None:
    # Use crypto symbols for training data
    all_tickers = C["scanner"].get_candidates() if "scanner" in C else \
        CFG["assets"].get("crypto", [])
    X_lstm_all, y_lstm_all = [], []
    X_ens_all, y_ens_all = [], []

    for ticker in all_tickers:
        try:
            df_raw = C["market"].get_ohlcv(ticker, "1h", "max")
            df = C["preprocessor"].transform(df_raw)
            df_daily = C["market"].get_ohlcv(ticker, "1d", "60d")
            micro = C["micro"].get(ticker)

            # LSTM sequences
            available_cols = [c for c in LSTM_FEATURE_COLS if c in df.columns]
            _thr = CFG["ml"].get("label_threshold", 0.003)
            _hor = CFG["ml"].get("label_horizon", 3)
            X, y = C["preprocessor"].build_sequences(df, available_cols,
                                                     threshold=_thr, horizon=_hor)
            if len(X) > 0:
                X_lstm_all.append(X)
                y_lstm_all.append(y)

            # Ensemble feature vectors (one per bar)
            feat = C["features"].compute(df, df_daily, micro)
            # Build simple label: 1 if next bar up, 0 otherwise
            closes = df["close"].values
            labels = np.array([1 if closes[i + 1] > closes[i] else 0
                               for i in range(len(closes) - 1)])
            # One feature row per bar (skip last bar, no label)
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

    # Train LSTM
    if X_lstm_all:
        X_cat = np.concatenate(X_lstm_all)
        y_cat = np.concatenate(y_lstm_all)
        logger.info(f"Training LSTM on {len(X_cat)} sequences")
        metrics = C["lstm"].train(X_cat, y_cat)
        logger.info(f"LSTM training complete: {metrics}")
    else:
        logger.error("No LSTM training data collected")

    # Train Ensemble
    if X_ens_all:
        X_e = np.concatenate(X_ens_all).astype(np.float32)
        y_e = np.concatenate(y_ens_all)
        # Drop dead features identified by LGB (importance=0 on last training run)
        # Feature order: rvol(0) spread_pct(1) liquidity(2) gap_pct(3) gap_str(4)
        #   vwap_pos(5) mom_1h(6) mom_str(7) wick_up(8) wick_dn(9)
        #   vol_z(10) vol_trend(11) bk_str(12) fake_bk(13) vol_pct(14)
        #   catalyst(15) sentiment(16) fomo(17) squeeze(18)
        # Dead: 0,1,2,4,5,10,11,15,16,17,18 → keep: 3,6,7,8,9,12,13,14
        _LIVE_FEATURES = [3, 6, 7, 8, 9, 12, 13, 14]   # gap_pct mom_1h mom_str wick_up wick_dn bk_str fake_bk vol_pct
        if X_e.shape[1] > max(_LIVE_FEATURES):
            X_e = X_e[:, _LIVE_FEATURES]
            logger.info(f"Ensemble: using {len(_LIVE_FEATURES)}/19 live features "
                        f"(dropped 11 zero-importance features from prev training)")
        logger.info(f"Training ensemble on {len(X_e)} samples")
        results = C["ensemble"].train(X_e, y_e)
        logger.info(f"Ensemble training: {results}")
    else:
        logger.warning("No ensemble training data — skipped")

    # Write retrain stamp so --paper doesn't immediately retrain
    _STAMP = os.path.join(os.path.dirname(__file__), "models", "last_train.txt")
    try:
        open(_STAMP, "w").write(str(time.time()))
    except Exception:
        pass


# ── Helpers: signal → TradeSignal adapter (for TelegramAlerter reuse) ────────

def _swing_to_trade(sig: SwingSignal) -> "TradeSignal":
    """Wrap SwingSignal in TradeSignal so existing TelegramAlerter works."""
    from models.signal_fusion import TradeSignal
    rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
    return TradeSignal(
        ticker            = sig.ticker,
        direction         = sig.direction,
        confidence        = sig.confidence,
        price             = sig.entry,
        stop_loss         = sig.stop_loss,
        take_profit       = sig.take_profit,
        timeframe         = "1d",
        strategy          = f"swing_{sig.setup_type}",
        sharpe_estimate   = round(sig.confidence * 2.0, 2),
        drawdown_estimate = round((1 - sig.confidence) * 0.06, 4),
        quantum_up_prob   = 0.5,
        fake_breakout_prob= 0.0,
        squeeze_prob      = 0.0,
        volatility_expected= sig.atr / max(sig.entry, 1.0),
        position_size_pct = round(sig.risk_amount / 8871.0, 4),
        composite_score   = sig.confidence,
        breakdown         = {
            **sig.breakdown,
            "setup":       sig.setup_type,
            "confluence":  sig.confluence_score,
            "poc":         sig.poc,
            "vah":         sig.vah,
            "val":         sig.val,
            "vwap":        sig.vwap,
            "cvd_div":     sig.cvd_divergent,
            "vol_ratio":   sig.vol_ratio,
            "balancing":   sig.is_balancing,
            "pos_shares":  sig.position_size_shares,
            "rr_actual":   round(rr, 2),
            "ta_dir":      sig.direction,
            "ml_dir":      sig.direction,
            "stat_dir":    sig.direction,
            "fund_dir":    sig.direction,
            "rsi":         50.0,
            "hurst":       0.5,
            "z_score":     0.0,
            "sentiment":   0.0,
        },
    )


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


def _scalp_to_trade(sig: ScalpSignal) -> "TradeSignal":
    from models.signal_fusion import TradeSignal
    rr = abs(sig.take_profit - sig.entry) / max(abs(sig.entry - sig.stop_loss), 1e-9)
    return TradeSignal(
        ticker            = sig.ticker,
        direction         = sig.direction,
        confidence        = sig.confidence,
        price             = sig.entry,
        stop_loss         = sig.stop_loss,
        take_profit       = sig.take_profit,
        timeframe         = "1m",
        strategy          = f"scalp_hfq_{sig.setup_type}",
        sharpe_estimate   = round(sig.confidence * 1.5, 2),
        drawdown_estimate = round((1 - sig.confidence) * 0.03, 4),
        quantum_up_prob   = 0.5,
        fake_breakout_prob= 0.0,
        squeeze_prob      = 0.0,
        volatility_expected= sig.atr / max(sig.entry, 1.0),
        position_size_pct = round(sig.risk_amount / 8871.0, 4),
        composite_score   = sig.confidence,
        breakdown         = {
            **sig.breakdown,
            "setup":       sig.setup_type,
            "poc":         sig.poc,
            "vwap":        sig.vwap,
            "contracts":   sig.contracts,
            "sl_ticks":    sig.sl_ticks,
            "tp_ticks":    sig.tp_ticks,
            "vol_spike":   sig.volume_spike_ratio,
            "cvd":         sig.cvd_at_signal,
            "rr_actual":   round(rr, 2),
            "ta_dir":      sig.direction,
            "ml_dir":      sig.direction,
            "stat_dir":    sig.direction,
            "fund_dir":    sig.direction,
            "rsi":         50.0,
            "hurst":       0.5,
            "z_score":     0.0,
            "sentiment":   0.0,
        },
    )


# ── Strategy 1: Swing Trading cycle ──────────────────────────────────────────

def run_swing_cycle(C: dict, paper: bool = False) -> int:
    """
    Scans all small/mid cap tickers daily.
    Returns number of signals dispatched.
    """
    strategy: SwingTradingStrategy = C["swing"]

    if not strategy.can_trade_this_week():
        logger.info("Swing: weekly trade limit reached — skip cycle")
        return 0

    tickers: list[str] = CFG.get("strategies", {}).get("swing", {}).get("tickers", [])
    if not tickers:
        logger.warning("Swing: no tickers configured in settings.yaml → strategies.swing.tickers")
        return 0

    dispatched = 0
    for ticker in tickers:
        try:
            df = C["market"].get_ohlcv(ticker, "1d", "90d")
            if df is None or len(df) < 25:
                continue

            sig = strategy.analyse(ticker, df)
            if sig is None:
                continue

            trade_sig = _swing_to_trade(sig)
            # Paper broker: simulate execution
            if paper:
                pos = C["paper_broker"].execute(sig, "swing")
                logger.info(
                    f"[PAPER EXEC][SWING] {ticker} {sig.direction} "
                    f"fill={pos.entry_price:.4f} (slip={pos.slippage_pct:.3%}) "
                    f"SL={sig.stop_loss:.4f}  TP={sig.take_profit:.4f} "
                    f"setup={sig.setup_type}  conf={sig.confidence:.1%}  id={pos.id}"
                )
            C["telegram"].send_signal(trade_sig, paper=paper, entry_price=sig.entry)
            C["journal"].record_signal(
                ticker          = sig.ticker,
                direction       = sig.direction,
                entry_price     = sig.entry,
                stop_loss       = sig.stop_loss,
                take_profit     = sig.take_profit,
                strategy        = trade_sig.strategy,
                confidence      = sig.confidence,
                position_size_pct= trade_sig.position_size_pct,
            )
            strategy.record_trade()
            dispatched += 1
            logger.info(
                f"Swing [{ticker}]: {sig.direction} {sig.setup_type} "
                f"confluence={sig.confluence_score} conf={sig.confidence:.1%}"
            )
        except Exception as e:
            logger.error(f"Swing [{ticker}]: {e}")

    return dispatched


# ── Strategy 2: Day Trading Forex + Gold cycle ───────────────────────────────

def run_daytrading_cycle(C: dict, paper: bool = False) -> int:
    """
    Checks Forex + Gold pairs during active sessions (London + NY).
    Returns number of signals dispatched.
    """
    strategy: DayTradingForexStrategy = C["daytrading"]

    if not strategy.can_trade_today():
        logger.info("DayTrading: daily trade/loss limit reached — skip cycle")
        return 0

    tickers: list[str] = CFG.get("strategies", {}).get("day_trading", {}).get("tickers", [])
    if not tickers:
        return 0

    # DXY (US Dollar Index) data — fetched once per cycle, used for Gold filter
    _DXY_TICKER = "DX-Y.NYB"
    df_dxy_1h: pd.DataFrame | None = None
    try:
        df_dxy_1h = C["market"].get_ohlcv(_DXY_TICKER, "1h", "10d")
        if df_dxy_1h is not None and len(df_dxy_1h) < 8:
            df_dxy_1h = None   # too few bars — disable filter
    except Exception as _e:
        logger.warning(f"DayTrading: DXY fetch failed ({_e}) — Gold DXY filter disabled")
        df_dxy_1h = None

    dispatched = 0
    for ticker in tickers:
        try:
            df_4h = C["market"].get_ohlcv(ticker, "4h", "60d")
            df_1h = C["market"].get_ohlcv(ticker, "1h", "30d")
            if df_4h is None or df_1h is None:
                continue
            if len(df_4h) < 25 or len(df_1h) < 20:
                continue

            # Pass DXY only for Gold; None for Forex pairs (no effect)
            sig = strategy.analyse(
                ticker, df_4h, df_1h,
                df_dxy_1h=(df_dxy_1h if ticker == "GC=F" else None),
            )
            if sig is None:
                continue

            trade_sig = _daytrading_to_trade(sig)
            if paper:
                pos = C["paper_broker"].execute(sig, "day_trading")
                logger.info(
                    f"[PAPER EXEC][DT] {ticker} {sig.direction} "
                    f"fill={pos.entry_price:.5f} (slip={pos.slippage_pct:.3%}) "
                    f"SL={sig.stop_loss:.5f}  TP={sig.take_profit:.5f} "
                    f"session={sig.session}  lot={sig.lot_size}  id={pos.id}"
                )
            C["telegram"].send_signal(trade_sig, paper=paper, entry_price=sig.entry)
            C["journal"].record_signal(
                ticker          = sig.ticker,
                direction       = sig.direction,
                entry_price     = sig.entry,
                stop_loss       = sig.stop_loss,
                take_profit     = sig.take_profit,
                strategy        = trade_sig.strategy,
                confidence      = sig.confidence,
                position_size_pct= trade_sig.position_size_pct,
            )
            strategy.record_trade()
            dispatched += 1
            logger.info(
                f"DayTrading [{ticker}]: {sig.direction} {sig.setup_type} "
                f"session={sig.session} conf={sig.confluence_score} pip_risk={sig.pip_risk:.1f}"
            )
        except Exception as e:
            logger.error(f"DayTrading [{ticker}]: {e}")

    return dispatched


# ── Strategy 3: Scalping HFQ cycle ───────────────────────────────────────────

def run_scalping_cycle(C: dict, paper: bool = False) -> int:
    """
    Polls ES futures on 1m bars. Called every loop iteration.
    Returns number of signals dispatched this cycle.
    """
    strategy: ScalpingHFQStrategy = C["scalping_hfq"]
    ticker = CFG.get("strategies", {}).get("scalping_hfq", {}).get("ticker", "ES=F")

    try:
        # yfinance: 1m data available for last 7 days only
        df = C["market"].get_ohlcv(ticker, "1m", "7d")
        if df is None or len(df) < 15:
            logger.debug(f"Scalp [{ticker}]: insufficient 1m bars")
            return 0

        sig = strategy.analyse(ticker, df)
        if sig is None:
            return 0

        trade_sig = _scalp_to_trade(sig)
        if paper:
            pos = C["paper_broker"].execute(sig, "scalping_hfq")
            logger.info(
                f"[PAPER EXEC][SCALP] {ticker} {sig.direction} "
                f"fill={pos.entry_price:.2f} (slip={pos.slippage_pct:.3%}) "
                f"SL={sig.stop_loss:.2f}  TP={sig.take_profit:.2f} "
                f"setup={sig.setup_type}  contracts={sig.contracts}  id={pos.id}"
            )
        C["telegram"].send_signal(trade_sig, paper=paper, entry_price=sig.entry)
        C["journal"].record_signal(
            ticker          = sig.ticker,
            direction       = sig.direction,
            entry_price     = sig.entry,
            stop_loss       = sig.stop_loss,
            take_profit     = sig.take_profit,
            strategy        = trade_sig.strategy,
            confidence      = sig.confidence,
            position_size_pct= trade_sig.position_size_pct,
        )
        logger.info(
            f"Scalp [{ticker}]: {sig.direction} {sig.setup_type} "
            f"poc={sig.poc} vwap={sig.vwap} contracts={sig.contracts}"
        )
        return 1

    except Exception as e:
        logger.error(f"Scalp [{ticker}]: {e}")
        return 0


# ── Adaptive feedback: analyse losses → patch strategies ──────────────────────

def run_trade_analysis(C: dict, paper: bool = True) -> None:
    """
    Runs TradeAnalyzer on closed paper trades.
    Applies adjustments to all three strategy instances via AdaptiveParams.
    Sends Telegram report if findings exist.
    Triggered: every 10 closed trades OR once per day minimum.
    """
    analyzer: TradeAnalyzer  = C["trade_analyzer"]
    adaptive: AdaptiveParams = C["adaptive_params"]

    report = analyzer.analyse(n_recent=100)

    if not report.strategies:
        logger.info("TradeAnalyzer: insufficient data — skipping")
        return

    # Apply adjustments to persistent store
    if report.adjustments:
        adaptive.apply(report.adjustments, expires_in_days=7)

    # Patch live strategy instances immediately
    for strategy_type, strategy_obj in [
        ("swing",        C["swing"]),
        ("day_trading",  C["daytrading"]),
        ("scalping_hfq", C["scalping_hfq"]),
    ]:
        adaptive.patch_strategy(strategy_obj, strategy_type)

    # Log summary
    logger.info(adaptive.summary())

    # Telegram report
    msg = analyzer.format_telegram_report(report)
    try:
        C["telegram"]._send_raw(msg)
    except Exception as e:
        logger.warning(f"TradeAnalyzer Telegram: {e}")


# ── Paper broker update cycle ─────────────────────────────────────────────────

def run_paper_broker_update(C: dict, paper: bool = True) -> int:
    """Returns number of positions closed this cycle."""
    """
    Called every main loop iteration in paper mode.
    Fetches latest OHLCV for each open paper position and checks TP/SL/timeout.
    Sends Telegram notification on every close.
    Feeds P&L back to scalping circuit breaker.
    """
    broker: PaperBroker = C["paper_broker"]
    open_pos = broker.open_positions()
    if not open_pos:
        return 0

    # Collect OHLCV for all unique tickers with open positions
    # Use TF matching the strategy type for realistic H/L check
    TF_MAP = {
        "swing":        ("1d", "5d"),
        "day_trading":  ("1h", "7d"),
        "scalping_hfq": ("1m", "7d"),
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

    closed = broker.update(
        ohlcv_map=ohlcv_map,
        telegram=C["telegram"],
        paper=paper,
    )

    for pos in closed:
        # Feed P&L to kill switch (paper losses still count toward daily DD limit)
        C["kill"].record_trade(pnl_pct=pos.pnl_pct)

        # Feed scalp circuit breaker with actual paper P&L
        if pos.strategy_type == "scalping_hfq":
            C["scalping_hfq"].record_result(pnl_usd=pos.pnl_usd)

        sign = "+" if pos.pnl_usd >= 0 else ""
        logger.info(
            f"[PAPER CLOSE] {pos.strategy_type} {pos.direction} {pos.ticker} "
            f"→ {sign}{pos.pnl_usd:.2f}$ ({pos.r_multiple:+.2f}R) [{pos.exit_reason}]"
        )

    # Daily equity summary log
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
    logger.info(f"AlgoTrad starting — mode={mode_str}")

    # ── SIGTERM handler — makes systemd stop send Telegram shutdown alert ─────
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt
    _signal.signal(_signal.SIGTERM, _sigterm_handler)

    # ── Notification démarrage (paper ET live) ────────────────────────────────
    C["telegram"].send_startup(mode_str)

    use_scanner   = CFG.get("scanner", {}).get("enabled", True)
    retrain_every = CFG["ml"]["retrain_interval_hours"] * 3600
    signal_dedup: dict[str, float] = {}
    DEDUP_SECONDS = 4 * 3600
    _STAMP = os.path.join(os.path.dirname(__file__), "models", "last_train.txt")
    try:
        last_train = float(open(_STAMP).read().strip())
    except Exception:
        last_train = 0.0
    last_day            = -1
    _cycles             = 0       # compteur cycles pour le rapport shutdown
    _signals_today      = 0       # compteur signaux du jour
    _last_heartbeat_day = -1      # heartbeat quotidien
    _cycle_prices: dict[str, float] = {}   # prix courants pour journal
    # Three-strategy session tracking
    _last_swing_day       = -1    # swing runs once per day
    _last_scalp_reset     = -1    # scalp session reset at open
    _last_analysis_day    = -1    # trade analysis runs once per day minimum
    _closed_trades_since_analysis = 0   # also trigger every 10 closed trades

    # Apply any existing adaptive adjustments at startup
    adaptive: AdaptiveParams = C["adaptive_params"]
    for _st, _so in [("swing", C["swing"]), ("day_trading", C["daytrading"]),
                     ("scalping_hfq", C["scalping_hfq"])]:
        adaptive.patch_strategy(_so, _st)
    logger.info(f"Startup adaptive params: {adaptive.summary()}")

    try:
      while True:
        # Daily kill-switch counter reset + heartbeat
        import datetime
        today = datetime.date.today().toordinal()
        if today != last_day:
            C["kill"].reset_daily()
            _signals_today = 0
            last_day = today

        # Daily heartbeat — sent once per day at first cycle after midnight
        if today != _last_heartbeat_day:
            pnl_stats = C["journal"].get_stats()
            C["telegram"].send_daily_summary(
                signals_today=_signals_today,
                cycles=_cycles,
                pnl_stats=pnl_stats,
                mode=mode_str,
            )
            # Paper equity summary on heartbeat
            if paper:
                eq_summary = C["paper_broker"].equity_summary()
                lines = ["📊 *Paper Equity des 3 stratégies*\n━━━━━━━━━━━━━━━━━━━━━━"]
                labels = {"swing": "Swing", "day_trading": "Day Trading", "scalping_hfq": "Scalping HFQ"}
                for st, eq in eq_summary.items():
                    sign = "+" if eq["roi_pct"] >= 0 else ""
                    lines.append(
                        f"*{labels.get(st, st)}*\n"
                        f"  Capital : `{eq['capital']:.2f}$`  ROI : `{sign}{eq['roi_pct']:.2f}%`\n"
                        f"  Trades : `{eq['trades']}`  WR : `{eq['win_rate']:.1f}%`  DD : `{eq['max_dd_usd']:.2f}$`"
                    )
                C["telegram"]._send_raw("\n".join(lines))
            _last_heartbeat_day = today

        # Kill switch check before each cycle
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

        # Dynamic ticker list — scanner or static fallback
        if use_scanner:
            all_tickers = C["scanner"].get_candidates()
            if not all_tickers:
                logger.warning("Scanner returned 0 candidates — skipping cycle")
                C["telegram"].send_error("Scanner: 0 candidats trouvés — cycle ignoré")
                time.sleep(300)
                continue
        else:
            all_tickers = CFG["assets"].get("crypto", [])

        # Per-ticker analysis
        cycle_scores: dict[str, float] = {}
        _cycle_errors = 0
        _cycle_prices.clear()
        for ticker in all_tickers:
            try:
                generated, score = analyse_asset(ticker, C, paper_mode=paper,
                                                 signal_dedup=signal_dedup,
                                                 dedup_seconds=DEDUP_SECONDS)
                if generated:
                    _signals_today += 1
                if score > 0:
                    cycle_scores[ticker] = score
                # Cache last price for position tracking
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

        # Journal: update open positions, record P&L for closed ones
        if _cycle_prices:
            closed_trades = C["journal"].update_positions(_cycle_prices)
            for trade in closed_trades:
                C["kill"].record_trade(pnl_pct=trade["pnl_pct"])
                sign = "+" if trade["pnl_pct"] >= 0 else ""
                logger.info(
                    f"Position closed: {trade['direction']} {trade['ticker']} "
                    f"{sign}{trade['pnl_pct']:.2%} ({trade['exit_reason']})"
                )

        # Alerte si trop d'erreurs dans le cycle
        if _cycle_errors >= len(all_tickers) // 2:
            C["telegram"].send_error(
                f"Cycle dégradé: {_cycle_errors}/{len(all_tickers)} tickers en erreur",
                critical=True,
            )

        # Multi-stock quantum ranking (best setups this cycle)
        if cycle_scores:
            top = C["qopt"].rank_signals(cycle_scores, top_k=3)
            logger.info(f"Top ranked setups this cycle: {top}")

        # ── Strategy 1: Swing Trading (once per calendar day) ─────────────────
        if CFG.get("strategies", {}).get("swing", {}).get("enabled", False):
            if today != _last_swing_day:
                try:
                    n_swing = run_swing_cycle(C, paper=paper)
                    _signals_today += n_swing
                    _last_swing_day = today
                    if n_swing:
                        logger.info(f"Swing cycle: {n_swing} signal(s) dispatched")
                except Exception as e:
                    logger.error(f"Swing cycle error: {e}")

        # ── Strategy 2: Day Trading Forex + Gold (every cycle, session-gated) ─
        if CFG.get("strategies", {}).get("day_trading", {}).get("enabled", False):
            try:
                n_dt = run_daytrading_cycle(C, paper=paper)
                _signals_today += n_dt
            except Exception as e:
                logger.error(f"DayTrading cycle error: {e}")

        # ── Strategy 3: Scalping HFQ (every cycle, market-hours-gated) ────────
        if CFG.get("strategies", {}).get("scalping_hfq", {}).get("enabled", False):
            # Reset session at start of each trading day
            if today != _last_scalp_reset:
                C["scalping_hfq"].reset_session()
                _last_scalp_reset = today
            try:
                n_scalp = run_scalping_cycle(C, paper=paper)
                _signals_today += n_scalp
            except Exception as e:
                logger.error(f"Scalping cycle error: {e}")

        # ── Paper broker: check open positions for TP/SL/timeout ─────────────
        if paper:
            try:
                closed_this_cycle = run_paper_broker_update(C, paper=True)
                _closed_trades_since_analysis += closed_this_cycle
            except Exception as e:
                logger.error(f"Paper broker update error: {e}")

        # ── Trade analysis: every 10 closes OR once per day ───────────────────
        _trigger_analysis = (
            _closed_trades_since_analysis >= 10 or
            today != _last_analysis_day
        )
        if _trigger_analysis:
            try:
                run_trade_analysis(C, paper=paper)
                _last_analysis_day = today
                _closed_trades_since_analysis = 0
            except Exception as e:
                logger.error(f"Trade analysis error: {e}")

        _cycles += 1
        scalp_enabled = CFG.get("strategies", {}).get("scalping_hfq", {}).get("enabled", False)

        if scalp_enabled:
            # ── Scalping inner loop: poll every 60s for 5 min ──────────────
            # Replaces time.sleep(300) so scalp runs at 1m frequency,
            # matching the 1m yfinance data granularity.
            logger.info("Cycle complete — scalp inner loop (5 × 60s)")
            for _tick in range(5):
                time.sleep(60)
                if today != _last_scalp_reset:
                    C["scalping_hfq"].reset_session()
                    _last_scalp_reset = today
                try:
                    n_s = run_scalping_cycle(C, paper=paper)
                    _signals_today += n_s
                except Exception as e:
                    logger.error(f"Scalp inner-loop tick {_tick}: {e}")
                if paper:
                    try:
                        n_closed = run_paper_broker_update(C, paper=True)
                        _closed_trades_since_analysis += n_closed
                    except Exception as e:
                        logger.error(f"Paper broker inner-loop: {e}")
        else:
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
    engine = BacktestEngine(CFG)
    results = engine.run(ticker, plot=True)
    print("\n" + "=" * 50)
    print(f"BACKTEST: {ticker}")
    for k, v in results.items():
        print(f"  {k:<24}: {v}")
    print("=" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlgoTrad Signal System")
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
