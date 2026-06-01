"""
AlgoTrad Dashboard v2 — Triple Strategy Monitor
================================================
Tabs : Overview | Swing | Day Trading | Scalping HFQ | Analyse

Sources de données :
  logs/paper_trades.csv        → PaperBroker (trades exécutés, P&L réel en $)
  logs/pnl_journal.csv         → Journal signaux
  config/adaptive_params.json  → Ajustements adaptatifs actifs

Lancement :
    cd ~/AlgoTrad && streamlit run dashboard/app.py
"""
from __future__ import annotations
import json, os, time, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
LOG_DIR       = ROOT / "logs"
PAPER_CSV     = LOG_DIR / "paper_trades.csv"
JOURNAL_CSV   = LOG_DIR / "pnl_journal.csv"
ADAPTIVE_JSON = ROOT / "config" / "adaptive_params.json"
CFG_YAML      = ROOT / "config" / "settings.yaml"
DASH_CFG      = ROOT / "cache" / "dashboard_settings.json"

try:
    with open(CFG_YAML) as f:
        _CFG = yaml.safe_load(f)
    CAPITAL_PER_STRATEGY = float(
        _CFG.get("strategies", {}).get("swing", {}).get("capital", 8_871)
    )
except Exception:
    CAPITAL_PER_STRATEGY = 8_871.0

# ── Palette ────────────────────────────────────────────────────────────────────
BG     = "#060B14"
CARD   = "#0D1526"
BORDER = "#1A2740"
BLUE   = "#3B82F6"
GREEN  = "#10B981"
RED    = "#EF4444"
GOLD   = "#F59E0B"
PURPLE = "#8B5CF6"
TEXT   = "#E2E8F0"
MUTED  = "#64748B"
ACCENT = "#1E3A5F"

STRATEGY_COLOR = {
    "swing":        BLUE,
    "day_trading":  GOLD,
    "scalping_hfq": GREEN,
}
STRATEGY_LABEL = {
    "swing":        "Swing Trading",
    "day_trading":  "Day Trading",
    "scalping_hfq": "Scalping HFQ",
}

# ── SVG icons ──────────────────────────────────────────────────────────────────
def svg(path_d: str, color: str = BLUE, size: int = 20) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f'{path_d}</svg>'
    )

ICO_WALLET   = svg('<rect x="1" y="4" width="22" height="16" rx="2"/><path d="M16 10h2"/>')
ICO_TREND    = svg('<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>', GREEN)
ICO_TARGET   = svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>', GOLD)
ICO_DOWN     = svg('<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>', RED)
ICO_CHART    = svg('<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>', BLUE)
ICO_ALERT    = svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', GOLD)
ICO_SHIELD   = svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>', GREEN)
ICO_CLOCK    = svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>', MUTED)
ICO_LIST     = svg('<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>', MUTED)
ICO_REFRESH  = svg('<polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.51"/>', BLUE)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoTrad v2",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{
      font-family: 'Inter', system-ui, sans-serif;
      background-color: {BG}; color: {TEXT};
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.2rem 1.8rem 3rem; max-width: 1700px; }}
  [data-testid="stSidebar"] {{
      background-color: {CARD}; border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  .metric-card {{
      background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
      padding: 1rem 1.2rem; display: flex; align-items: flex-start;
      gap: 0.8rem; transition: border-color .2s; height: 100%;
  }}
  .metric-card:hover {{ border-color: {BLUE}44; }}
  .metric-icon {{
      background: {ACCENT}; border-radius: 8px; padding: 7px;
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }}
  .metric-label {{
      font-size: 0.70rem; font-weight: 500; color: {MUTED};
      text-transform: uppercase; letter-spacing: .06em; margin-bottom: 0.2rem;
  }}
  .metric-value {{
      font-size: 1.25rem; font-weight: 700; color: {TEXT}; line-height: 1;
  }}
  .metric-sub {{ font-size: 0.73rem; font-weight: 500; margin-top: 0.15rem; }}
  .positive {{ color: {GREEN}; }}
  .negative {{ color: {RED}; }}
  .neutral  {{ color: {MUTED}; }}
  .gold     {{ color: {GOLD}; }}

  .section-title {{
      display: flex; align-items: center; gap: 0.5rem;
      font-size: 0.82rem; font-weight: 600; color: {MUTED};
      text-transform: uppercase; letter-spacing: .07em;
      margin: 1.4rem 0 0.7rem; border-bottom: 1px solid {BORDER};
      padding-bottom: 0.45rem;
  }}
  .strategy-badge {{
      display: inline-flex; align-items: center; gap: 4px;
      padding: 3px 10px; border-radius: 999px;
      font-size: 0.72rem; font-weight: 600;
  }}
  .badge-swing       {{ background: {BLUE}22;   color: {BLUE};   border: 1px solid {BLUE}44; }}
  .badge-day_trading {{ background: {GOLD}22;   color: {GOLD};   border: 1px solid {GOLD}44; }}
  .badge-scalping_hfq{{ background: {GREEN}22;  color: {GREEN};  border: 1px solid {GREEN}44; }}
  .badge-tp          {{ background: {GREEN}22;  color: {GREEN};  }}
  .badge-sl          {{ background: {RED}22;    color: {RED};    }}
  .badge-open        {{ background: {GOLD}22;   color: {GOLD};   }}
  .badge-timeout     {{ background: {MUTED}22;  color: {MUTED};  }}
  .badge-eod         {{ background: {PURPLE}22; color: {PURPLE}; }}

  .adaptive-block {{
      background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
      padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
  }}
  .adaptive-block h4 {{
      font-size: 0.78rem; font-weight: 600; color: {MUTED};
      text-transform: uppercase; letter-spacing: .06em; margin: 0 0 0.5rem;
  }}
  .tag {{
      display: inline-block; padding: 2px 8px; border-radius: 5px;
      font-size: 0.70rem; font-weight: 600; margin: 2px;
      background: {RED}22; color: {RED}; border: 1px solid {RED}33;
  }}
  .tag-warn {{
      background: {GOLD}22; color: {GOLD}; border: 1px solid {GOLD}33;
  }}
  [data-testid="stPlotlyChart"] {{
      border-radius: 12px; background: {CARD};
      border: 1px solid {BORDER}; overflow: hidden;
  }}
  .stButton button {{
      background: {BLUE}; color: white; border: none; border-radius: 7px;
      font-weight: 600; font-size: 0.82rem; padding: 0.4rem 1rem; width: 100%;
  }}
  .stButton button:hover {{ background: #2563EB; }}
  .stTabs [data-baseweb="tab-list"] {{
      background: {CARD}; border-radius: 10px; padding: 4px;
      border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      color: {MUTED}; font-size: 0.82rem; font-weight: 500;
      padding: 6px 16px; border-radius: 7px;
  }}
  .stTabs [aria-selected="true"] {{
      background: {ACCENT}; color: {TEXT} !important; font-weight: 600;
  }}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def mc(icon_svg: str, label: str, value: str, sub: str = "", cls: str = "neutral") -> str:
    return f"""<div class="metric-card">
      <div class="metric-icon">{icon_svg}</div>
      <div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub {cls}">{sub}</div>' if sub else ""}
      </div></div>"""

def sec(icon_svg: str, title: str) -> None:
    st.markdown(f'<div class="section-title">{icon_svg}<span>{title}</span></div>',
                unsafe_allow_html=True)

def _badge(text: str, cls: str) -> str:
    return f'<span class="badge-{cls}" style="display:inline-block;padding:2px 7px;border-radius:4px;font-size:.7rem;font-weight:600">{text}</span>'

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=MUTED),
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False,
)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_paper_trades() -> pd.DataFrame:
    if not PAPER_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PAPER_CSV)
        df = df[df["exit_reason"].notna() & (df["exit_reason"] != "")]
        for col in ["r_multiple", "pnl_usd", "running_equity", "slippage_pct", "hold_bars"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_journal() -> pd.DataFrame:
    if not JOURNAL_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(JOURNAL_CSV)
        for col in ["pnl_pct", "confidence"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_adaptive() -> dict:
    if not ADAPTIVE_JSON.exists():
        return {}
    try:
        with open(ADAPTIVE_JSON) as f:
            return json.load(f)
    except Exception:
        return {}


def strategy_metrics(df: pd.DataFrame, capital: float) -> dict:
    """Compute standard metrics for a strategy's paper trades."""
    if df.empty:
        return dict(total=0, wins=0, losses=0, win_rate=0, pnl_usd=0,
                    roi=0, avg_r=0, max_dd=0, capital=capital,
                    tp_count=0, sl_count=0, to_count=0, eod_count=0)
    wins = int((df["r_multiple"] > 0).sum())
    losses = int((df["r_multiple"] <= 0).sum())
    pnl_usd = float(df["pnl_usd"].sum())
    # Running equity series
    eq = df["running_equity"].iloc[-1] if "running_equity" in df.columns and len(df) > 0 else capital + pnl_usd
    # Max drawdown: peak-to-trough of running equity
    if "running_equity" in df.columns:
        eq_series = pd.concat([pd.Series([capital]), df["running_equity"]])
        peak = eq_series.cummax()
        dd_series = eq_series - peak
        max_dd = float(dd_series.min())
    else:
        max_dd = float(df[df["pnl_usd"] < 0]["pnl_usd"].sum())
    return dict(
        total      = len(df),
        wins       = wins,
        losses     = losses,
        win_rate   = wins / len(df) * 100 if len(df) > 0 else 0,
        pnl_usd    = pnl_usd,
        roi        = pnl_usd / capital * 100,
        avg_r      = float(df["r_multiple"].mean()),
        max_dd     = max_dd,
        capital    = eq,
        tp_count   = int((df["exit_reason"] == "TP").sum()),
        sl_count   = int((df["exit_reason"] == "SL").sum()),
        to_count   = int((df["exit_reason"] == "TIMEOUT").sum()),
        eod_count  = int((df["exit_reason"] == "EOD").sum()),
    )


def equity_fig(df: pd.DataFrame, capital: float, color: str) -> go.Figure:
    """Build equity curve figure from paper_trades running_equity column."""
    fig = go.Figure()

    if df.empty or "running_equity" not in df.columns:
        fig.add_trace(go.Scatter(x=[datetime.datetime.now()], y=[capital],
                                 mode="lines", line=dict(color=color, width=2)))
    else:
        df_s = df.sort_values("timestamp_close")
        xs = [datetime.datetime.now() - datetime.timedelta(hours=1)] + \
             pd.to_datetime(df_s["timestamp_close"], errors="coerce").tolist()
        ys = [capital] + df_s["running_equity"].tolist()
        fill_color = f"rgba({','.join(str(int(int(color[1:3], 16))) for _ in range(1))},0.08)"
        # Convert hex color to rgba for fillcolor (Plotly rejects 8-char hex)
        _h = color.lstrip("#")
        _r, _g, _b = int(_h[0:2], 16), int(_h[2:4], 16), int(_h[4:6], 16)
        fill_color = f"rgba({_r},{_g},{_b},0.10)"

        # Baseline
        fig.add_trace(go.Scatter(x=xs, y=[capital]*len(xs), mode="lines",
                                 line=dict(color="rgba(0,0,0,0)", width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color, width=2.2),
            fill="tonexty", fillcolor=fill_color,
            marker=dict(size=5, color=color),
            hovertemplate="<b>%{x|%d/%m %H:%M}</b><br>$%{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(y=capital, line_dash="dot", line_color=MUTED,
                      annotation_text=f"Départ {capital:,.0f}$",
                      annotation_font_color=MUTED,
                      annotation_position="bottom right")

    y_vals = [capital]
    if df is not None and not df.empty and "running_equity" in df.columns:
        y_vals += df["running_equity"].tolist()
    ymin, ymax = min(y_vals), max(y_vals)
    span = max(ymax - ymin, capital * 0.01)

    fig.update_layout(
        **PLOT_BASE,
        height=240,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9),
                   tickformat="%d/%m\n%H:%M"),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                   tickprefix="$", tickfont=dict(size=9),
                   range=[ymin - span * 0.3, ymax + span * 0.3]),
        hovermode="x unified",
    )
    return fig


def trades_table(df: pd.DataFrame, color: str) -> None:
    """Render a styled paper trades table."""
    if df.empty:
        st.markdown(f'<div style="color:{MUTED};text-align:center;padding:2rem;'
                    f'background:{CARD};border-radius:10px;border:1px solid {BORDER}">'
                    f'Aucun trade fermé pour cette stratégie.</div>',
                    unsafe_allow_html=True)
        return

    disp = df.copy()
    if "timestamp_open" in disp.columns:
        disp["Ouvert"] = pd.to_datetime(disp["timestamp_open"], errors="coerce") \
                           .dt.strftime("%d/%m %H:%M")
    if "timestamp_close" in disp.columns:
        disp["Fermé"] = pd.to_datetime(disp["timestamp_close"], errors="coerce") \
                          .dt.strftime("%d/%m %H:%M")

    cols_show = [c for c in [
        "Ouvert", "Fermé", "ticker", "direction",
        "entry_price_with_slip", "exit_price",
        "stop_loss", "take_profit",
        "r_multiple", "pnl_usd",
        "exit_reason", "hold_bars", "slippage_pct",
    ] if c in disp.columns]

    rename = {
        "ticker":                "Ticker",
        "direction":             "Dir",
        "entry_price_with_slip": "Entrée $",
        "exit_price":            "Sortie $",
        "stop_loss":             "SL $",
        "take_profit":           "TP $",
        "r_multiple":            "R",
        "pnl_usd":               "P&L $",
        "exit_reason":           "Raison",
        "hold_bars":             "Bars",
        "slippage_pct":          "Slip %",
    }

    tbl = disp[cols_show].rename(columns=rename).sort_values(
        "Fermé" if "Fermé" in disp[cols_show].rename(columns=rename).columns else cols_show[0],
        ascending=False,
    ).head(50)

    def _color_r(val):
        if not isinstance(val, (int, float)):
            return ""
        return f"color:{GREEN};font-weight:600" if val > 0 else f"color:{RED};font-weight:600"

    fmt: dict = {}
    if "Entrée $"  in tbl.columns: fmt["Entrée $"]  = "${:.4f}"
    if "Sortie $"  in tbl.columns: fmt["Sortie $"]  = "${:.4f}"
    if "SL $"      in tbl.columns: fmt["SL $"]      = "${:.4f}"
    if "TP $"      in tbl.columns: fmt["TP $"]      = "${:.4f}"
    if "R"         in tbl.columns: fmt["R"]         = "{:+.2f}R"
    if "P&L $"     in tbl.columns: fmt["P&L $"]     = "${:+.2f}"
    if "Slip %"    in tbl.columns: fmt["Slip %"]    = "{:.3%}"

    styled = (
        tbl.style
        .map(_color_r, subset=[c for c in ["R", "P&L $"] if c in tbl.columns])
        .format(fmt)
        .set_properties(**{"background-color": CARD, "color": TEXT,
                           "border-color": BORDER, "font-size": "0.8rem"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", ACCENT), ("color", TEXT),
                ("font-size", "0.70rem"), ("font-weight", "600"),
                ("text-transform", "uppercase"), ("letter-spacing", ".05em"),
                ("padding", "7px 9px"), ("border-bottom", f"1px solid {BORDER}"),
            ]},
            {"selector": "td", "props": [
                ("padding", "6px 9px"), ("border-bottom", f"1px solid {BORDER}11"),
            ]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=340)


def pnl_bar(df: pd.DataFrame, color: str) -> go.Figure:
    """P&L bar chart per trade."""
    if df.empty:
        return go.Figure()
    df_s = df.sort_values("timestamp_close" if "timestamp_close" in df.columns else df.columns[0])
    labels = (df_s.get("ticker", pd.Series(["?"]*len(df_s))).astype(str) + " " +
              df_s.get("direction", pd.Series([""]*len(df_s))).astype(str))
    colors = [GREEN if v > 0 else RED for v in df_s["pnl_usd"]]
    fig = go.Figure(go.Bar(
        x=labels, y=df_s["pnl_usd"],
        marker_color=colors,
        text=df_s["pnl_usd"].apply(lambda x: f"${x:+.1f}"),
        textposition="auto", textfont=dict(size=9, color="white"),
        hovertemplate="<b>%{x}</b><br>P&L : $%{y:+,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_BASE, height=220,
        xaxis=dict(showgrid=False, zeroline=False, tickangle=-40, tickfont=dict(size=8)),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=True,
                   zerolinecolor=BORDER, tickprefix="$", tickfont=dict(size=9)),
        bargap=0.3,
    )
    return fig


def exit_pie(df: pd.DataFrame) -> go.Figure:
    tp  = int((df["exit_reason"] == "TP").sum())      if not df.empty else 0
    sl  = int((df["exit_reason"] == "SL").sum())      if not df.empty else 0
    to  = int((df["exit_reason"] == "TIMEOUT").sum()) if not df.empty else 0
    eod = int((df["exit_reason"] == "EOD").sum())     if not df.empty else 0
    fig = go.Figure(go.Pie(
        labels=["Take Profit", "Stop Loss", "Timeout", "EOD"],
        values=[tp, sl, to, eod],
        hole=0.55,
        marker=dict(colors=[GREEN, RED, GOLD, PURPLE],
                    line=dict(color=BG, width=2)),
        textinfo="percent", textfont=dict(size=10, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value} trades<extra></extra>",
    ))
    total = tp + sl + to + eod
    wr = tp / total * 100 if total > 0 else 0
    fig.update_layout(
        **{**PLOT_BASE, "showlegend": True},
        height=220,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f"{wr:.0f}%\nWR", x=0.5, y=0.5,
                          font=dict(size=14, color=TEXT, family="Inter"),
                          showarrow=False)],
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1.4rem">'
        f'<div style="background:linear-gradient(135deg,{BLUE},{PURPLE});border-radius:10px;'
        f'width:36px;height:36px;display:flex;align-items:center;justify-content:center">'
        f'{svg("<polyline points=\'23 6 13.5 15.5 8.5 10.5 1 18\'/>", "white", 17)}</div>'
        f'<span style="font-size:1.1rem;font-weight:700;">AlgoTrad v2</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    n_days = st.slider("Historique (jours)", 1, 30, 7)
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if st.button("Actualiser"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:.7rem;color:{MUTED};text-align:center">'
        f'v2.0 · {datetime.date.today()}</div>',
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
paper = load_paper_trades()
journal = load_journal()
adaptive = load_adaptive()

# Filter by date
cutoff = (datetime.datetime.now() - datetime.timedelta(days=n_days)).strftime("%Y-%m-%d")
if not paper.empty and "timestamp_open" in paper.columns:
    paper = paper[paper["timestamp_open"] >= cutoff]

# Per-strategy subsets
def strat_df(st_type: str) -> pd.DataFrame:
    if paper.empty or "strategy_type" not in paper.columns:
        return pd.DataFrame()
    return paper[paper["strategy_type"] == st_type].copy()

df_swing  = strat_df("swing")
df_dt     = strat_df("day_trading")
df_scalp  = strat_df("scalping_hfq")

m_swing  = strategy_metrics(df_swing,  CAPITAL_PER_STRATEGY)
m_dt     = strategy_metrics(df_dt,     CAPITAL_PER_STRATEGY)
m_scalp  = strategy_metrics(df_scalp,  CAPITAL_PER_STRATEGY)

# ── Page header ────────────────────────────────────────────────────────────────
total_pnl = m_swing["pnl_usd"] + m_dt["pnl_usd"] + m_scalp["pnl_usd"]
total_cap = CAPITAL_PER_STRATEGY * 3 + total_pnl
n_total   = m_swing["total"] + m_dt["total"] + m_scalp["total"]

st.markdown(
    f'<div style="display:flex;align-items:center;gap:.8rem;margin-bottom:1rem">'
    f'<div style="background:linear-gradient(135deg,{BLUE},{PURPLE});border-radius:10px;'
    f'width:40px;height:40px;display:flex;align-items:center;justify-content:center">'
    f'{svg("<polyline points=\'23 6 13.5 15.5 8.5 10.5 1 18\'/>", "white", 20)}</div>'
    f'<span style="font-size:1.5rem;font-weight:700">AlgoTrad Dashboard</span>'
    f'<span style="font-size:.78rem;color:{MUTED};margin-left:auto">'
    f'Capital total : <b style="color:{TEXT}">${CAPITAL_PER_STRATEGY*3:,.0f}</b> '
    f'&nbsp;·&nbsp; {n_total} trades fermés &nbsp;·&nbsp; {n_days}j</span></div>',
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
t_over, t_swing, t_dt, t_scalp, t_analysis = st.tabs([
    "📊 Overview",
    "🔵 Swing Trading",
    "🟡 Day Trading",
    "🟢 Scalping HFQ",
    "🔍 Analyse",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t_over:
    # ── Global KPIs ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    sign = "+" if total_pnl >= 0 else ""
    pnl_cls = "positive" if total_pnl >= 0 else "negative"

    with c1:
        st.markdown(mc(ICO_WALLET, "Portfolio total",
            f"${total_cap:,.2f}",
            f"{sign}${total_pnl:,.2f} ({sign}{total_pnl/CAPITAL_PER_STRATEGY/3*100:.2f}%)",
            pnl_cls), unsafe_allow_html=True)
    with c2:
        n_w = m_swing["wins"] + m_dt["wins"] + m_scalp["wins"]
        wr  = n_w / n_total * 100 if n_total > 0 else 0
        wr_cls = "positive" if wr >= 55 else "negative" if wr < 40 else "neutral"
        st.markdown(mc(ICO_TARGET, "Win Rate global",
            f"{wr:.1f}%", f"{n_w}W / {n_total-n_w}L sur {n_total} trades",
            wr_cls), unsafe_allow_html=True)
    with c3:
        all_r = paper["r_multiple"].mean() if not paper.empty else 0
        r_cls = "positive" if all_r > 0 else "negative"
        st.markdown(mc(ICO_TREND, "R-multiple moyen",
            f"{all_r:+.2f}R", "Espérance par trade", r_cls), unsafe_allow_html=True)
    with c4:
        dd_total = min(m_swing["max_dd"], m_dt["max_dd"], m_scalp["max_dd"])
        st.markdown(mc(ICO_DOWN, "Max Drawdown",
            f"${dd_total:,.2f}", f"{dd_total/CAPITAL_PER_STRATEGY*100:.2f}% par stratégie",
            "negative" if dd_total < 0 else "neutral"), unsafe_allow_html=True)
    with c5:
        st.markdown(mc(ICO_CLOCK, "Trades fermés", str(n_total),
            f"Swing {m_swing['total']} · DT {m_dt['total']} · Scalp {m_scalp['total']}",
            "neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3-strategy equity comparison ───────────────────────────────────────────
    sec(ICO_TREND, "Equity curves — 3 stratégies")

    strat_rows = [
        ("swing",        df_swing,  m_swing,  BLUE,   "Swing"),
        ("day_trading",  df_dt,     m_dt,     GOLD,   "Day Trading"),
        ("scalping_hfq", df_scalp,  m_scalp,  GREEN,  "Scalping HFQ"),
    ]

    fig_cmp = go.Figure()
    # Baseline
    fig_cmp.add_hline(y=CAPITAL_PER_STRATEGY, line_dash="dot", line_color=MUTED,
                      annotation_text="Capital initial",
                      annotation_font_color=MUTED)

    for st_type, df_s, m, col, lbl in strat_rows:
        if df_s.empty or "running_equity" not in df_s.columns:
            xs = [datetime.datetime.now()]
            ys = [CAPITAL_PER_STRATEGY]
        else:
            df_ss = df_s.sort_values("timestamp_close")
            xs = [datetime.datetime.now() - datetime.timedelta(hours=2)] + \
                 pd.to_datetime(df_ss["timestamp_close"], errors="coerce").tolist()
            ys = [CAPITAL_PER_STRATEGY] + df_ss["running_equity"].tolist()
        fig_cmp.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=lbl, line=dict(color=col, width=2),
            hovertemplate=f"<b>{lbl}</b><br>%{{x|%d/%m %H:%M}}<br>$%{{y:,.2f}}<extra></extra>",
        ))

    fig_cmp.update_layout(
        **{**PLOT_BASE, "showlegend": True},
        height=300,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9),
                   tickformat="%d/%m\n%H:%M"),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                   tickprefix="$", tickfont=dict(size=9)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cmp, use_container_width=True, key="cmp_equity")

    # ── Strategy summary cards ─────────────────────────────────────────────────
    sec(ICO_CHART, "Résumé par stratégie")
    cols = st.columns(3)

    for i, (st_type, df_s, m, col, lbl) in enumerate(strat_rows):
        sign_m = "+" if m["pnl_usd"] >= 0 else ""
        pnl_c  = "positive" if m["pnl_usd"] >= 0 else "negative"
        wr_c   = "positive" if m["win_rate"] >= 55 else "negative" if m["win_rate"] < 40 else "neutral"
        with cols[i]:
            st.markdown(
                f'<div style="background:{CARD};border:1px solid {col}44;border-radius:12px;'
                f'padding:1rem;border-top:3px solid {col}">'
                f'<div style="font-size:.78rem;font-weight:700;color:{col};margin-bottom:.6rem">{lbl}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">'
                f'<div><div style="font-size:.65rem;color:{MUTED}">Capital</div>'
                f'<div style="font-size:1rem;font-weight:700">${m["capital"]:,.2f}</div></div>'
                f'<div><div style="font-size:.65rem;color:{MUTED}">ROI</div>'
                f'<div style="font-size:1rem;font-weight:700;color:{GREEN if m["roi"]>=0 else RED}">{sign_m}{m["roi"]:.2f}%</div></div>'
                f'<div><div style="font-size:.65rem;color:{MUTED}">Win Rate</div>'
                f'<div style="font-size:1rem;font-weight:700;color:{GREEN if m["win_rate"]>=55 else RED if m["win_rate"]<40 else MUTED}">{m["win_rate"]:.1f}%</div></div>'
                f'<div><div style="font-size:.65rem;color:{MUTED}">R moyen</div>'
                f'<div style="font-size:1rem;font-weight:700;color:{GREEN if m["avg_r"]>0 else RED}">{m["avg_r"]:+.2f}R</div></div>'
                f'<div><div style="font-size:.65rem;color:{MUTED}">Trades</div>'
                f'<div style="font-size:1rem;font-weight:700">{m["total"]}</div></div>'
                f'<div><div style="font-size:.65rem;color:{MUTED}">Max DD</div>'
                f'<div style="font-size:1rem;font-weight:700;color:{RED}">${m["max_dd"]:,.2f}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SWING TRADING
# ══════════════════════════════════════════════════════════════════════════════
with t_swing:
    m = m_swing
    sign = "+" if m["pnl_usd"] >= 0 else ""

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(mc(ICO_WALLET, "Capital Swing",
            f"${m['capital']:,.2f}",
            f"{sign}${m['pnl_usd']:,.2f} ({sign}{m['roi']:.2f}%)",
            "positive" if m["pnl_usd"] >= 0 else "negative"), unsafe_allow_html=True)
    with c2:
        wr_c = "positive" if m["win_rate"] >= 55 else "negative" if m["win_rate"] < 40 else "neutral"
        st.markdown(mc(ICO_TARGET, "Win Rate",
            f"{m['win_rate']:.1f}%", f"{m['wins']}W / {m['losses']}L",
            wr_c), unsafe_allow_html=True)
    with c3:
        r_c = "positive" if m["avg_r"] > 0 else "negative"
        st.markdown(mc(ICO_TREND, "R-multiple moyen",
            f"{m['avg_r']:+.2f}R",
            f"TP:{m['tp_count']} SL:{m['sl_count']}",
            r_c), unsafe_allow_html=True)
    with c4:
        st.markdown(mc(ICO_DOWN, "Max Drawdown",
            f"${m['max_dd']:,.2f}",
            f"{m['max_dd']/CAPITAL_PER_STRATEGY*100:.2f}% du capital",
            "negative" if m["max_dd"] < 0 else "neutral"), unsafe_allow_html=True)

    col_eq, col_pie = st.columns([3, 1])
    with col_eq:
        sec(ICO_TREND, "Equity curve — Swing Trading")
        st.plotly_chart(equity_fig(df_swing, CAPITAL_PER_STRATEGY, BLUE), key="swing_equity",
                        use_container_width=True)
    with col_pie:
        sec(ICO_TARGET, "Répartition sorties")
        st.plotly_chart(exit_pie(df_swing), use_container_width=True, key="swing_pie")

    col_bar, _ = st.columns([3, 1])
    with col_bar:
        sec(ICO_CHART, "P&L par trade")
        st.plotly_chart(pnl_bar(df_swing, BLUE), use_container_width=True, key="swing_bar")

    sec(ICO_LIST, "Trades fermés — Swing")
    trades_table(df_swing, BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DAY TRADING
# ══════════════════════════════════════════════════════════════════════════════
with t_dt:
    m = m_dt
    sign = "+" if m["pnl_usd"] >= 0 else ""

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(mc(ICO_WALLET, "Capital Day Trading",
            f"${m['capital']:,.2f}",
            f"{sign}${m['pnl_usd']:,.2f} ({sign}{m['roi']:.2f}%)",
            "positive" if m["pnl_usd"] >= 0 else "negative"), unsafe_allow_html=True)
    with c2:
        wr_c = "positive" if m["win_rate"] >= 55 else "negative" if m["win_rate"] < 40 else "neutral"
        st.markdown(mc(ICO_TARGET, "Win Rate",
            f"{m['win_rate']:.1f}%", f"{m['wins']}W / {m['losses']}L",
            wr_c), unsafe_allow_html=True)
    with c3:
        r_c = "positive" if m["avg_r"] > 0 else "negative"
        st.markdown(mc(ICO_TREND, "R-multiple moyen",
            f"{m['avg_r']:+.2f}R",
            f"EOD fermés: {m['eod_count']}",
            r_c), unsafe_allow_html=True)
    with c4:
        st.markdown(mc(ICO_CLOCK, "Trades",
            str(m["total"]),
            f"TP:{m['tp_count']} · SL:{m['sl_count']} · EOD:{m['eod_count']}",
            "neutral"), unsafe_allow_html=True)

    # Session breakdown
    if not df_dt.empty and "session" in df_dt.columns:
        sec(ICO_CHART, "Répartition par session")
        sess_stats = df_dt.groupby("session").agg(
            trades=("pnl_usd", "count"),
            pnl=("pnl_usd", "sum"),
            wins=("r_multiple", lambda x: (x > 0).sum()),
        ).reset_index()

        fig_sess = go.Figure()
        colors_sess = [BLUE if s == "london" else GOLD for s in sess_stats["session"]]
        fig_sess.add_trace(go.Bar(
            x=sess_stats["session"], y=sess_stats["pnl"],
            marker_color=colors_sess,
            text=sess_stats["pnl"].apply(lambda x: f"${x:+.1f}"),
            textposition="auto", textfont=dict(size=10, color="white"),
        ))
        fig_sess.update_layout(**PLOT_BASE, height=180,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=BORDER, tickprefix="$"))
        st.plotly_chart(fig_sess, use_container_width=True, key="dt_sess")

    col_eq, col_pie = st.columns([3, 1])
    with col_eq:
        sec(ICO_TREND, "Equity curve — Day Trading")
        st.plotly_chart(equity_fig(df_dt, CAPITAL_PER_STRATEGY, GOLD), key="dt_equity",
                        use_container_width=True)
    with col_pie:
        sec(ICO_TARGET, "Répartition sorties")
        st.plotly_chart(exit_pie(df_dt), use_container_width=True, key="dt_pie")

    col_bar, _ = st.columns([3, 1])
    with col_bar:
        sec(ICO_CHART, "P&L par trade")
        st.plotly_chart(pnl_bar(df_dt, GOLD), use_container_width=True, key="dt_bar")

    sec(ICO_LIST, "Trades fermés — Day Trading")
    trades_table(df_dt, GOLD)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCALPING HFQ
# ══════════════════════════════════════════════════════════════════════════════
with t_scalp:
    m = m_scalp
    sign = "+" if m["pnl_usd"] >= 0 else ""

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(mc(ICO_WALLET, "Capital Scalp",
            f"${m['capital']:,.2f}",
            f"{sign}${m['pnl_usd']:,.2f} ({sign}{m['roi']:.2f}%)",
            "positive" if m["pnl_usd"] >= 0 else "negative"), unsafe_allow_html=True)
    with c2:
        wr_c = "positive" if m["win_rate"] >= 55 else "negative" if m["win_rate"] < 40 else "neutral"
        st.markdown(mc(ICO_TARGET, "Win Rate",
            f"{m['win_rate']:.1f}%", f"{m['wins']}W / {m['losses']}L",
            wr_c), unsafe_allow_html=True)
    with c3:
        r_c = "positive" if m["avg_r"] > 0 else "negative"
        st.markdown(mc(ICO_TREND, "R-multiple moyen",
            f"{m['avg_r']:+.2f}R", "Cible: +2.0R", r_c), unsafe_allow_html=True)
    with c4:
        st.markdown(mc(ICO_CLOCK, "Timeouts",
            str(m["to_count"]),
            f"{m['to_count']/m['total']*100:.0f}% des trades" if m["total"] > 0 else "—",
            "negative" if m["to_count"] > m["total"] * 0.4 else "neutral"), unsafe_allow_html=True)
    with c5:
        st.markdown(mc(ICO_CHART, "Trades aujourd'hui",
            str(len(df_scalp[df_scalp.get("timestamp_open", pd.Series(dtype=str)) >= datetime.date.today().isoformat()])) if not df_scalp.empty else "0",
            "SPY (S&P 500)", "neutral"), unsafe_allow_html=True)

    # Setup type breakdown
    if not df_scalp.empty and "setup_type" in df_scalp.columns:
        sec(ICO_CHART, "P&L par setup")
        setup_stats = df_scalp.groupby("setup_type").agg(
            trades=("pnl_usd", "count"),
            pnl=("pnl_usd", "sum"),
            avg_r=("r_multiple", "mean"),
        ).reset_index()
        colors_setup = [GREEN if p > 0 else RED for p in setup_stats["pnl"]]
        fig_setup = go.Figure(go.Bar(
            x=setup_stats["setup_type"], y=setup_stats["pnl"],
            marker_color=colors_setup,
            text=setup_stats["pnl"].apply(lambda x: f"${x:+.1f}"),
            textposition="auto", textfont=dict(size=10, color="white"),
        ))
        fig_setup.update_layout(**PLOT_BASE, height=180,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=BORDER, tickprefix="$"))
        st.plotly_chart(fig_setup, use_container_width=True, key="scalp_setup")

    col_eq, col_pie = st.columns([3, 1])
    with col_eq:
        sec(ICO_TREND, "Equity curve — Scalping HFQ")
        st.plotly_chart(equity_fig(df_scalp, CAPITAL_PER_STRATEGY, GREEN), key="scalp_equity",
                        use_container_width=True)
    with col_pie:
        sec(ICO_TARGET, "Répartition sorties")
        st.plotly_chart(exit_pie(df_scalp), use_container_width=True, key="scalp_pie")

    col_bar, _ = st.columns([3, 1])
    with col_bar:
        sec(ICO_CHART, "P&L par trade")
        st.plotly_chart(pnl_bar(df_scalp, GREEN), use_container_width=True, key="scalp_bar")

    sec(ICO_LIST, "Trades fermés — Scalping HFQ")
    trades_table(df_scalp, GREEN)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
with t_analysis:
    sec(ICO_SHIELD, "Paramètres adaptatifs actifs")

    if not adaptive or all(k.startswith("_") for k in adaptive):
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;'
            f'padding:1.5rem;color:{MUTED};text-align:center">'
            f'Aucun ajustement actif — paramètres par défaut<br>'
            f'<span style="font-size:.75rem">Les ajustements s\'activent après 10 trades fermés</span>'
            f'</div>', unsafe_allow_html=True,
        )
    else:
        import datetime as _dt
        expires_at = adaptive.get("_expires_at", 0)
        generated  = adaptive.get("_generated_at", 0)
        exp_str = _dt.datetime.fromtimestamp(expires_at).strftime("%d/%m/%Y %H:%M") if expires_at else "—"
        gen_str = _dt.datetime.fromtimestamp(generated).strftime("%d/%m/%Y %H:%M") if generated else "—"

        st.markdown(
            f'<div style="background:{CARD};border:1px solid {GOLD}44;border-radius:10px;'
            f'padding:.7rem 1rem;margin-bottom:.8rem;font-size:.78rem">'
            f'<span style="color:{GOLD}">⚡ Ajustements actifs</span> &nbsp;·&nbsp; '
            f'Généré: <b>{gen_str}</b> &nbsp;·&nbsp; '
            f'Expire: <b style="color:{RED}">{exp_str}</b></div>',
            unsafe_allow_html=True,
        )

        a_cols = st.columns(3)
        for i, (st_key, lbl, col) in enumerate([
            ("swing",        "Swing Trading",  BLUE),
            ("day_trading",  "Day Trading",    GOLD),
            ("scalping_hfq", "Scalping HFQ",  GREEN),
        ]):
            adj = adaptive.get(st_key, {})
            with a_cols[i]:
                html_parts = [
                    f'<div style="background:{CARD};border:1px solid {col}33;border-radius:10px;'
                    f'padding:.9rem;border-top:3px solid {col}">'
                    f'<div style="font-size:.75rem;font-weight:700;color:{col};margin-bottom:.5rem">{lbl}</div>'
                ]
                if not adj:
                    html_parts.append(f'<div style="color:{MUTED};font-size:.75rem">Aucun ajustement</div>')
                else:
                    if adj.get("min_confluence_delta", 0) != 0:
                        html_parts.append(
                            f'<div style="font-size:.72rem;margin-bottom:.3rem">'
                            f'<span style="color:{MUTED}">min_confluence</span> '
                            f'<span style="color:{GOLD};font-weight:600">+{adj["min_confluence_delta"]}</span></div>'
                        )
                    for list_key, list_lbl, list_col in [
                        ("blocked_setups",      "Setups bloqués",    RED),
                        ("blocked_sessions",    "Sessions bloquées", RED),
                        ("blacklisted_tickers", "Tickers bannis",    RED),
                    ]:
                        items = adj.get(list_key, [])
                        if items:
                            tags = "".join(f'<span class="tag">{x}</span>' for x in items)
                            html_parts.append(
                                f'<div style="font-size:.70rem;color:{MUTED};margin:.2rem 0">{list_lbl}:</div>{tags}'
                            )
                    if adj.get("breakout_vol_mult_delta", 0) != 0:
                        html_parts.append(
                            f'<div style="font-size:.72rem;margin-top:.3rem">'
                            f'<span style="color:{MUTED}">vol_mult</span> '
                            f'<span style="color:{GOLD};font-weight:600">+{adj["breakout_vol_mult_delta"]:.2f}</span></div>'
                        )
                html_parts.append('</div>')
                st.markdown("".join(html_parts), unsafe_allow_html=True)

    # ── Pattern analysis ────────────────────────────────────────────────────────
    sec(ICO_ALERT, "Patterns de perte détectés")

    if paper.empty:
        st.markdown(
            f'<div style="color:{MUTED};text-align:center;padding:1.5rem;'
            f'background:{CARD};border-radius:10px;border:1px solid {BORDER}">'
            f'Pas encore de données — en attente de trades fermés</div>',
            unsafe_allow_html=True,
        )
    else:
        # Compute per-strategy, per-setup loss rates
        findings_html = []
        for st_key, df_s, col in [
            ("swing",        df_swing, BLUE),
            ("day_trading",  df_dt,    GOLD),
            ("scalping_hfq", df_scalp, GREEN),
        ]:
            lbl = STRATEGY_LABEL[st_key]
            if df_s.empty or "setup_type" not in df_s.columns:
                continue
            for setup, sub in df_s.groupby("setup_type"):
                n = len(sub)
                if n < 3:
                    continue
                loss_rate = (sub["r_multiple"] <= 0).sum() / n
                if loss_rate > 0.6:
                    sev_col  = RED if loss_rate > 0.75 else GOLD
                    sev_label= "CRITIQUE" if loss_rate > 0.75 else "ATTENTION"
                    findings_html.append(
                        f'<div style="background:{CARD};border:1px solid {sev_col}33;'
                        f'border-radius:9px;padding:.7rem 1rem;margin-bottom:.5rem;'
                        f'border-left:3px solid {sev_col}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="font-size:.78rem;font-weight:600">{lbl} → <code>{setup}</code></span>'
                        f'<span style="font-size:.7rem;color:{sev_col};font-weight:700">{sev_label}</span></div>'
                        f'<div style="font-size:.73rem;color:{MUTED};margin-top:.3rem">'
                        f'Perte {loss_rate:.0%} sur {n} trades &nbsp;·&nbsp; '
                        f'R moyen: {sub["r_multiple"].mean():+.2f} &nbsp;·&nbsp; '
                        f'P&L total: ${sub["pnl_usd"].sum():+.2f}</div></div>'
                    )

        if findings_html:
            st.markdown("\n".join(findings_html), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="background:{CARD};border:1px solid {GREEN}33;border-radius:10px;'
                f'padding:1rem;text-align:center">'
                f'<span style="color:{GREEN};font-weight:600">✓ Aucun pattern critique détecté</span>'
                f'<div style="font-size:.75rem;color:{MUTED};margin-top:.3rem">'
                f'Tous les setups sous le seuil de 60% de pertes</div></div>',
                unsafe_allow_html=True,
            )

    # ── Global performance table ────────────────────────────────────────────────
    sec(ICO_LIST, "Tous les trades fermés")

    if paper.empty:
        st.info("Aucun trade — lancez `python main.py --paper`")
    else:
        st.dataframe(
            paper[[c for c in [
                "strategy_type", "ticker", "direction",
                "timestamp_open", "timestamp_close",
                "entry_price_with_slip", "exit_price",
                "r_multiple", "pnl_usd", "exit_reason",
                "hold_bars", "slippage_pct",
            ] if c in paper.columns]].sort_values(
                "timestamp_close" if "timestamp_close" in paper.columns else paper.columns[0],
                ascending=False,
            ).head(100),
            use_container_width=True,
            height=400,
        )

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
