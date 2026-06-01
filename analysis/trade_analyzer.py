"""
Trade Loss Analyzer — post-mortem automatique des trades perdants.

Lit logs/paper_trades.csv (PaperBroker) + logs/pnl_journal.csv (journal live).
Détecte les patterns récurrents de pertes et génère des ajustements de paramètres
pour chaque stratégie (swing / day_trading / scalping_hfq).

Optionnel: envoie les pertes à Gemini pour explication en langage naturel.

Patterns détectés:
  1. Taux de perte par setup_type   → block ou raise confluence
  2. Taux de perte par confluence   → raise min_confluence
  3. Taux de perte par session      → bloquer session problématique
  4. Volume ratio bas + SL hit      → raise vol threshold
  5. Timeout excessif               → réduire max_hold_bars
  6. CVD non-divergent + SL         → forcer CVD filter
  7. Ticker systematically losing   → blacklist temporaire

Sorties:
  - AnalysisReport (dataclass)
  - dict d'ajustements → AdaptiveParams
  - Message Telegram formaté
  - Explication Gemini (si GEMINI_API_KEY disponible)
"""
from __future__ import annotations
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np
from utils.logger import logger

PAPER_LOG   = os.path.join(os.path.dirname(__file__), "..", "logs", "paper_trades.csv")
JOURNAL_LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "pnl_journal.csv")

# Thresholds for triggering an adjustment
LOSS_RATE_TRIGGER   = 0.60   # > 60% loss rate on a pattern → flag
MIN_SAMPLE          = 3      # min trades to consider a pattern significant
TIMEOUT_RATE_TRIGGER= 0.40   # > 40% trades timeout → tighten hold/proximity


@dataclass
class PatternFinding:
    pattern_id: str          # e.g. "swing_poc_breakout_high_loss"
    strategy:   str          # "swing" | "day_trading" | "scalping_hfq"
    description: str         # human-readable
    loss_rate:  float        # 0–1
    n_trades:   int
    severity:   str          # "low" | "medium" | "high"
    recommendation: str      # concrete parameter change
    adjustment: dict         # {param: new_value}


@dataclass
class StrategyStats:
    strategy: str
    total: int = 0
    wins:   int = 0
    losses: int = 0
    timeouts: int = 0
    eod_closes: int = 0
    total_pnl_usd: float = 0.0
    setup_stats:      dict = field(default_factory=dict)
    confluence_stats: dict = field(default_factory=dict)
    session_stats:    dict = field(default_factory=dict)
    ticker_stats:     dict = field(default_factory=dict)
    vol_ratio_by_outcome: dict = field(default_factory=lambda: {"win": [], "loss": []})

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    @property
    def expectancy_usd(self) -> float:
        """Expected USD per trade."""
        return self.total_pnl_usd / self.total if self.total > 0 else 0.0


@dataclass
class AnalysisReport:
    strategies:   dict[str, StrategyStats]    # strategy → stats
    findings:     list[PatternFinding]
    adjustments:  dict[str, Any]              # ready-to-apply adjustments
    n_trades_analysed: int = 0
    gemini_explanation: str = ""
    generated_at: float = 0.0


class TradeAnalyzer:
    """
    Analyses closed paper trades to detect systematic loss patterns
    and generate parameter adjustments for each strategy.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.gemini_enabled = cfg.get("gemini", {}).get("enabled", False)

    # ── Main entry ────────────────────────────────────────────────────────────

    def analyse(self, n_recent: int = 100) -> AnalysisReport:
        """
        Read paper_trades.csv, compute per-strategy stats, detect patterns.

        n_recent: only look at the last N closed trades (rolling window).
        Returns AnalysisReport with findings and suggested adjustments.
        """
        import time
        df = self._load_trades(n_recent)
        if df is None or df.empty:
            logger.info("TradeAnalyzer: no closed trades found yet")
            return AnalysisReport(
                strategies={}, findings=[], adjustments={},
                n_trades_analysed=0, generated_at=time.time(),
            )

        # Per-strategy stats
        stats: dict[str, StrategyStats] = {}
        for st in df["strategy_type"].unique():
            sub = df[df["strategy_type"] == st]
            stats[st] = self._compute_stats(st, sub)

        # Pattern detection
        findings: list[PatternFinding] = []
        for st, s in stats.items():
            findings.extend(self._detect_patterns(st, s, df[df["strategy_type"] == st]))

        # Translate findings → adjustments
        adjustments = self._build_adjustments(findings, stats)

        # Gemini LLM explanation (optional)
        gemini_text = ""
        if self.gemini_enabled and findings:
            try:
                gemini_text = self._gemini_explain(stats, findings)
            except Exception as e:
                logger.warning(f"TradeAnalyzer Gemini: {e}")

        report = AnalysisReport(
            strategies       = stats,
            findings         = findings,
            adjustments      = adjustments,
            n_trades_analysed= len(df),
            gemini_explanation= gemini_text,
            generated_at     = time.time(),
        )

        logger.info(
            f"TradeAnalyzer: {len(df)} trades analysed  "
            f"{len(findings)} patterns found  "
            f"{len(adjustments)} adjustments generated"
        )
        return report

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_trades(self, n_recent: int) -> pd.DataFrame | None:
        """Load and combine paper_trades.csv (primary) + pnl_journal.csv (fallback)."""
        frames = []

        # Primary: paper_trades.csv (from PaperBroker)
        if os.path.exists(PAPER_LOG):
            try:
                df = pd.read_csv(PAPER_LOG)
                df = df[df["exit_reason"].notna() & (df["exit_reason"] != "")]
                if not df.empty:
                    df["source"] = "paper_broker"
                    # Normalise column names
                    if "strategy_type" not in df.columns:
                        df["strategy_type"] = "unknown"
                    frames.append(df)
            except Exception as e:
                logger.warning(f"TradeAnalyzer: paper_trades.csv read error: {e}")

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp_open", ascending=False)
        combined = combined.head(n_recent)

        # Numeric coercion
        for col in ["r_multiple", "pnl_usd", "hold_bars", "slippage_pct"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

        return combined

    # ── Stats computation ─────────────────────────────────────────────────────

    def _compute_stats(self, strategy: str, df: pd.DataFrame) -> StrategyStats:
        s = StrategyStats(strategy=strategy)
        s.total       = len(df)
        s.wins        = int((df["r_multiple"] > 0).sum())
        s.losses      = int((df["r_multiple"] <= 0).sum())
        s.timeouts    = int((df["exit_reason"] == "TIMEOUT").sum())
        s.eod_closes  = int((df["exit_reason"] == "EOD").sum())
        s.total_pnl_usd = float(df["pnl_usd"].sum())

        # Setup-type stats
        if "exit_reason" in df.columns and "setup_type" in df.columns:
            for setup, sub in df.groupby("setup_type"):
                wins = int((sub["r_multiple"] > 0).sum())
                s.setup_stats[setup] = {
                    "total": len(sub),
                    "wins":  wins,
                    "loss_rate": round(1 - wins / len(sub), 3),
                    "avg_r": round(float(sub["r_multiple"].mean()), 3),
                }

        # Confluence stats (from breakdown column if available)
        # (paper_trades.csv doesn't store breakdown; use r_multiple distribution)

        # Session stats
        if "session" in df.columns:
            for sess, sub in df.groupby("session"):
                wins = int((sub["r_multiple"] > 0).sum())
                s.session_stats[sess] = {
                    "total": len(sub),
                    "loss_rate": round(1 - wins / len(sub), 3) if len(sub) > 0 else 0.0,
                }

        # Ticker stats
        if "ticker" in df.columns:
            for ticker, sub in df.groupby("ticker"):
                wins = int((sub["r_multiple"] > 0).sum())
                s.ticker_stats[ticker] = {
                    "total": len(sub),
                    "loss_rate": round(1 - wins / len(sub), 3) if len(sub) > 0 else 0.0,
                    "pnl_usd": round(float(sub["pnl_usd"].sum()), 2),
                }

        return s

    # ── Pattern detection ─────────────────────────────────────────────────────

    def _detect_patterns(
        self, strategy: str, s: StrategyStats, df: pd.DataFrame,
    ) -> list[PatternFinding]:
        findings: list[PatternFinding] = []

        # ── Pattern 1: Setup type high loss rate ─────────────────────────────
        for setup, stats in s.setup_stats.items():
            if stats["total"] < MIN_SAMPLE:
                continue
            if stats["loss_rate"] > LOSS_RATE_TRIGGER:
                sev = "high" if stats["loss_rate"] > 0.75 else "medium"
                findings.append(PatternFinding(
                    pattern_id  = f"{strategy}_{setup}_high_loss",
                    strategy    = strategy,
                    description = (
                        f"Setup '{setup}' perd {stats['loss_rate']:.0%} des trades "
                        f"({stats['total'] - stats['wins']}/{stats['total']} SL)"
                    ),
                    loss_rate   = stats["loss_rate"],
                    n_trades    = stats["total"],
                    severity    = sev,
                    recommendation = f"Bloquer temporairement '{setup}' ou augmenter confluence",
                    adjustment  = {
                        "blocked_setups": [setup],
                    } if stats["loss_rate"] > 0.75 else {
                        "min_confluence_delta": +1,
                    },
                ))

        # ── Pattern 2: Session high loss rate ────────────────────────────────
        for sess, stats in s.session_stats.items():
            if stats["total"] < MIN_SAMPLE:
                continue
            if stats["loss_rate"] > LOSS_RATE_TRIGGER:
                findings.append(PatternFinding(
                    pattern_id  = f"{strategy}_session_{sess}_high_loss",
                    strategy    = strategy,
                    description = (
                        f"Session '{sess}' : {stats['loss_rate']:.0%} de pertes "
                        f"({stats['total']} trades)"
                    ),
                    loss_rate   = stats["loss_rate"],
                    n_trades    = stats["total"],
                    severity    = "medium",
                    recommendation = f"Éviter session '{sess}' pour {strategy}",
                    adjustment  = {"blocked_sessions": [sess]},
                ))

        # ── Pattern 3: Ticker systematically losing ──────────────────────────
        for ticker, stats in s.ticker_stats.items():
            if stats["total"] < MIN_SAMPLE:
                continue
            if stats["loss_rate"] > 0.75 and stats["pnl_usd"] < -50:
                findings.append(PatternFinding(
                    pattern_id  = f"{strategy}_{ticker}_toxic",
                    strategy    = strategy,
                    description = (
                        f"Ticker '{ticker}' : {stats['loss_rate']:.0%} pertes "
                        f"({stats['pnl_usd']:+.2f}$)"
                    ),
                    loss_rate   = stats["loss_rate"],
                    n_trades    = stats["total"],
                    severity    = "high",
                    recommendation = f"Blacklist temporaire '{ticker}'",
                    adjustment  = {"blacklisted_tickers": [ticker]},
                ))

        # ── Pattern 4: Timeout rate too high ─────────────────────────────────
        if s.total >= MIN_SAMPLE:
            timeout_rate = s.timeouts / s.total
            if timeout_rate > TIMEOUT_RATE_TRIGGER:
                findings.append(PatternFinding(
                    pattern_id  = f"{strategy}_high_timeout",
                    strategy    = strategy,
                    description = (
                        f"{timeout_rate:.0%} des trades expirent par timeout "
                        f"({s.timeouts}/{s.total}) → niveaux trop éloignés"
                    ),
                    loss_rate   = timeout_rate,
                    n_trades    = s.total,
                    severity    = "medium",
                    recommendation = "Réduire vwap_tol ou resserrer max_hold_bars",
                    adjustment  = {"max_hold_bars_delta": -5, "vwap_tol_delta": -0.0002},
                ))

        # ── Pattern 5: Overall win rate too low ──────────────────────────────
        if s.total >= 5 and s.win_rate < 0.35:
            findings.append(PatternFinding(
                pattern_id  = f"{strategy}_global_low_winrate",
                strategy    = strategy,
                description = (
                    f"Win rate global {s.win_rate:.0%} < 35% sur {s.total} trades "
                    f"({s.expectancy_usd:+.2f}$/trade)"
                ),
                loss_rate   = 1 - s.win_rate,
                n_trades    = s.total,
                severity    = "high",
                recommendation = "Augmenter min_confluence +1 et vol threshold",
                adjustment  = {
                    "min_confluence_delta": +1,
                    "breakout_vol_mult_delta": +0.25,
                },
            ))

        return findings

    # ── Build adjustments dict ────────────────────────────────────────────────

    def _build_adjustments(
        self,
        findings: list[PatternFinding],
        stats: dict[str, StrategyStats],
    ) -> dict[str, Any]:
        """
        Merge findings into a structured adjustments dict per strategy.
        Delta adjustments are accumulated across findings.
        """
        import time

        result: dict[str, Any] = {
            "_generated_at": time.time(),
            "_expires_in_days": 7,
        }

        # Group findings by strategy
        by_strategy: dict[str, list[PatternFinding]] = defaultdict(list)
        for f in findings:
            by_strategy[f.strategy].append(f)

        for strategy, fs in by_strategy.items():
            adj: dict[str, Any] = {}

            # Accumulate adjustments
            conf_delta = 0
            vol_delta  = 0.0
            blocked_setups: list[str] = []
            blocked_sessions: list[str] = []
            blacklisted: list[str] = []

            for f in fs:
                a = f.adjustment
                conf_delta       += a.get("min_confluence_delta", 0)
                vol_delta        += a.get("breakout_vol_mult_delta", 0.0)
                blocked_setups   += a.get("blocked_setups", [])
                blocked_sessions += a.get("blocked_sessions", [])
                blacklisted      += a.get("blacklisted_tickers", [])

            if conf_delta != 0:
                adj["min_confluence_delta"] = conf_delta
            if vol_delta != 0:
                adj["breakout_vol_mult_delta"] = round(vol_delta, 3)
            if blocked_setups:
                adj["blocked_setups"] = list(set(blocked_setups))
            if blocked_sessions:
                adj["blocked_sessions"] = list(set(blocked_sessions))
            if blacklisted:
                adj["blacklisted_tickers"] = list(set(blacklisted))

            if adj:
                result[strategy] = adj

        return result

    # ── Gemini LLM explanation ────────────────────────────────────────────────

    def _gemini_explain(
        self,
        stats: dict[str, StrategyStats],
        findings: list[PatternFinding],
    ) -> str:
        """
        Send loss data to Gemini Flash and get natural language explanation.
        Returns formatted text explanation.
        """
        import os
        import json as _json
        import requests

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return ""

        # Build compact context for Gemini
        context = {}
        for st, s in stats.items():
            context[st] = {
                "win_rate":    f"{s.win_rate:.0%}",
                "total":       s.total,
                "pnl_usd":     f"{s.total_pnl_usd:+.2f}$",
                "setup_stats": s.setup_stats,
                "session_stats": s.session_stats,
            }

        pattern_list = [
            {"id": f.pattern_id, "desc": f.description, "severity": f.severity}
            for f in findings
        ]

        prompt = f"""Tu es un analyste quantitatif expert en trading algorithmique.

Voici les statistiques de performance de 3 stratégies de trading (paper trading) :

DONNÉES:
{_json.dumps(context, indent=2, ensure_ascii=False)}

PATTERNS DÉTECTÉS:
{_json.dumps(pattern_list, indent=2, ensure_ascii=False)}

Analyse:
1. Quelle est la cause systémique principale des pertes pour chaque stratégie ?
2. Ces pertes sont-elles dues à une mauvaise logique d'entrée, de timing ou de gestion du risque ?
3. Quels ajustements concrets recommandes-tu ? (seuils, filtres, marchés)

Sois direct et factuel. Max 300 mots. Format: une section par stratégie."""

        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={api_key}"
            )
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            return text.strip()
        except Exception as e:
            logger.warning(f"Gemini trade analysis failed: {e}")
            return ""

    # ── Telegram report formatting ────────────────────────────────────────────

    def format_telegram_report(self, report: AnalysisReport) -> str:
        if not report.strategies:
            return "📊 *Analyse trades* — pas encore assez de trades fermés."

        lines = [
            "📊 *Analyse des Trades Perdants*",
            f"━━━━━━━━━━━━━━━━━━━━━━",
            f"📋 Trades analysés: `{report.n_trades_analysed}`",
            f"🔍 Patterns trouvés: `{len(report.findings)}`",
            "",
        ]

        label_map = {
            "swing": "🔵 Swing Trading",
            "day_trading": "🟡 Day Trading",
            "scalping_hfq": "🔴 Scalping HFQ",
        }

        for st, s in report.strategies.items():
            label = label_map.get(st, st)
            wr_emoji = "✅" if s.win_rate >= 0.5 else "⚠️" if s.win_rate >= 0.35 else "❌"
            lines.append(f"*{label}*")
            lines.append(
                f"  {wr_emoji} WR: `{s.win_rate:.0%}` ({s.wins}W/{s.losses}L)  "
                f"P&L: `{s.total_pnl_usd:+.2f}$`"
            )
            if s.setup_stats:
                worst = max(s.setup_stats.items(), key=lambda x: x[1]["loss_rate"])
                if worst[1]["total"] >= MIN_SAMPLE:
                    lines.append(
                        f"  ⚡ Pire setup: `{worst[0]}` ({worst[1]['loss_rate']:.0%} pertes)"
                    )
            lines.append("")

        if report.findings:
            sev_emoji = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}
            lines.append("*Patterns détectés:*")
            for f in sorted(report.findings, key=lambda x: x.severity == "high", reverse=True)[:5]:
                e = sev_emoji.get(f.severity, "•")
                lines.append(f"  {e} {f.description}")
                lines.append(f"     → _{f.recommendation}_")
            lines.append("")

        if report.adjustments:
            lines.append("*Ajustements appliqués:*")
            for st, adj in report.adjustments.items():
                if st.startswith("_"):
                    continue
                label = label_map.get(st, st)
                lines.append(f"  📐 {label}: `{adj}`")
            lines.append("")

        if report.gemini_explanation:
            lines.append("*🤖 Analyse Gemini:*")
            # Truncate for Telegram (4096 char limit)
            snippet = report.gemini_explanation[:600]
            if len(report.gemini_explanation) > 600:
                snippet += "…"
            lines.append(f"_{snippet}_")

        return "\n".join(lines)
