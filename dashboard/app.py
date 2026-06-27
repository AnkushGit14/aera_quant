"""
dashboard/app.py
----------------
Multi-Asset Futures Spread Monitor & Signal Dashboard
Built by: Ankush Jaiswal

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

# ── Path Setup ────────────────────────────────────────────────────────────────
# Ensure project root is importable regardless of CWD
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.fetcher import fetch_all, fetch_vix, get_asset_metadata, get_sectors
from analysis.indicators import (
    compute_rsi,
    compute_garch_vol,
    classify_regime,
    compute_bollinger_bands,
)
from analysis.spreads import get_all_spreads
from analysis.signals import get_signals_for_all
from analysis.backtester import run_spread_backtest

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Futures Signal Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #0A0E1A 0%, #0d1b3e 50%, #0A0E1A 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,212,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.header-title {
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-title span { color: #00D4FF; }
.header-sub {
    font-size: 13px;
    color: rgba(226,232,240,0.55);
    margin-top: 6px;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(0, 212, 255, 0.4); }
.metric-label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(226,232,240,0.5);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #fff;
}
.metric-value.green  { color: #10B981; }
.metric-value.red    { color: #EF4444; }
.metric-value.yellow { color: #F59E0B; }
.metric-value.cyan   { color: #00D4FF; }

/* ── Signal Pills ── */
.pill-buy  { background:#065f46; color:#6ee7b7; border:1px solid #059669;
             padding:3px 12px; border-radius:999px; font-weight:600; font-size:12px; }
.pill-sell { background:#7f1d1d; color:#fca5a5; border:1px solid #dc2626;
             padding:3px 12px; border-radius:999px; font-weight:600; font-size:12px; }
.pill-hold { background:#44351a; color:#fcd34d; border:1px solid #d97706;
             padding:3px 12px; border-radius:999px; font-weight:600; font-size:12px; }

/* ── Section Headers ── */
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #E2E8F0;
    border-left: 3px solid #00D4FF;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 11px;
    color: rgba(226,232,240,0.3);
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

/* ── Streamlit tweaks ── */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
div[data-testid="metric-container"] { background: rgba(17,24,39,0.5);
                                       border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <p class="header-title">AeraQuant <span>Analytics</span></p>
  <p class="header-sub">
    Real-time signal generation · GARCH(1,1) Volatility · RSI Momentum · Cross-Asset Spread Z-Score<br>
    Assets: Crude Oil (NYMEX) · Gold (COMEX) · S&P 500 (CME) · EUR/USD (Forex) &nbsp;|&nbsp;
    Data: Yahoo Finance via yfinance &nbsp;|&nbsp; Built by Ankush Jaiswal
  </p>
</div>
""", unsafe_allow_html=True)

# ── Data Loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)  # cache 5 mins
def load_all_data():
    prices      = fetch_all(period="5y")   # 5y → ~1,260 obs: GARCH-stable
    vix_series  = fetch_vix(period="5y")
    rsi_dict    = {a: compute_rsi(p)       for a, p in prices.items()}
    garch_dict  = {a: compute_garch_vol(p) for a, p in prices.items()}
    hist_vol    = {a: garch_dict[a]["hist_vol"] for a in prices}
    regime_dict = {a: classify_regime(hist_vol[a]) for a in prices}
    bb_dict     = {a: compute_bollinger_bands(p)   for a, p in prices.items()}
    spreads     = get_all_spreads(prices)
    signals_df  = get_signals_for_all(prices, rsi_dict, regime_dict, spreads)
    return prices, vix_series, rsi_dict, garch_dict, hist_vol, regime_dict, bb_dict, spreads, signals_df

with st.spinner("⏳ Fetching 5Y live market data (12 assets + VIX)..."):
    try:
        prices, vix_series, rsi_dict, garch_dict, hist_vol, regime_dict, bb_dict, spreads, signals_df = load_all_data()
    except Exception as e:
        st.error(f"❌ Data fetch failed: {e}")
        st.stop()

if not prices:
    st.error("❌ No data fetched. Check your internet connection.")
    st.stop()

meta    = get_asset_metadata()
sectors = get_sectors()

# ── VIX Macro Regime Banner ───────────────────────────────────────────────────
if not vix_series.empty:
    vix_now  = float(vix_series.iloc[-1])
    vix_prev = float(vix_series.iloc[-2]) if len(vix_series) > 1 else vix_now
    vix_chg  = vix_now - vix_prev

    if vix_now < 15:
        vix_label = "CALM"
        vix_color = "#10B981"   # green
        vix_bg    = "rgba(16,185,129,0.08)"
        vix_border= "rgba(16,185,129,0.35)"
        vix_icon  = "🟢"
        vix_note  = "Risk-ON · Low fear · Signals active"
    elif vix_now < 25:
        vix_label = "CAUTION"
        vix_color = "#F59E0B"   # orange
        vix_bg    = "rgba(245,158,11,0.08)"
        vix_border= "rgba(245,158,11,0.35)"
        vix_icon  = "🟡"
        vix_note  = "Moderate fear · Monitor closely"
    else:
        vix_label = "FEAR / RISK-OFF"
        vix_color = "#EF4444"   # red
        vix_bg    = "rgba(239,68,68,0.08)"
        vix_border= "rgba(239,68,68,0.35)"
        vix_icon  = "🔴"
        vix_note  = "High fear · BUY signals suppressed globally"

    vix_arrow = "▲" if vix_chg >= 0 else "▼"
    st.markdown(f"""
    <div style="background:{vix_bg};border:1px solid {vix_border};border-radius:12px;
                padding:14px 24px;margin-bottom:20px;display:flex;
                align-items:center;justify-content:space-between;">
        <div>
            <span style="font-size:12px;font-weight:600;color:{vix_color};
                         text-transform:uppercase;letter-spacing:1px;">VIX · CBOE FEAR INDEX</span>
            <span style="font-size:24px;font-weight:700;color:{vix_color};
                         margin-left:16px;">{vix_now:.1f}</span>
            <span style="font-size:13px;color:{vix_color};margin-left:8px;">
                {vix_arrow} {abs(vix_chg):.2f}</span>
        </div>
        <div style="text-align:right;">
            <span style="font-size:15px;font-weight:700;color:{vix_color};">
                {vix_icon} {vix_label}</span>
            <div style="font-size:11px;color:rgba(226,232,240,0.5);margin-top:2px;">{vix_note}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── KPI Strip — Compact Grid ──────────────────────────────────────────────────
st.markdown('<p class="section-title">Live Market Snapshot — 12 Assets · 5Y Data</p>',
            unsafe_allow_html=True)

SECTOR_ICONS = {
    "Energy": "⚡", "Metals": "🥇", "Agriculture": "🌾",
    "Equity": "📈", "Rates": "🏦", "FX": "💱"
}

# Flatten all available assets into a single list
all_assets = []
for sector_name, asset_list in sectors.items():
    icon = SECTOR_ICONS.get(sector_name, "")
    for asset in asset_list:
        if asset in prices:
            all_assets.append((asset, icon))

# Render in a compact grid (6 columns per row)
num_cols = 6
for i in range(0, len(all_assets), num_cols):
    cols = st.columns(num_cols)
    chunk = all_assets[i:i + num_cols]
    for col, (asset, icon) in zip(cols, chunk):
        price_series = prices[asset]
        price_now    = float(price_series.iloc[-1])
        price_prev   = float(price_series.iloc[-2])
        chg          = (price_now - price_prev) / price_prev * 100
        color_cls    = "green" if chg >= 0 else "red"
        arrow        = "▲" if chg >= 0 else "▼"
        ticker       = meta.get(asset, {}).get("ticker", "")
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:10px 8px; margin-bottom:12px;">
                <div class="metric-label" style="font-size:9px;">{icon} {asset}</div>
                <div class="metric-value" style="font-size:15px;">{price_now:,.2f}</div>
                <div class="metric-value {color_cls}" style="font-size:11px;margin-top:2px;">
                    {arrow} {abs(chg):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Tabs Configuration ────────────────────────────────────────────────────────
tab_signals, tab_backtest, tab_docs = st.tabs([
    "Signals Desk",
    "Backtesting Engine",
    "Methodology"
])

# ==============================================================================
# TAB 1: COINTEGRATED SIGNALS DESK
# ==============================================================================
with tab_signals:
    st.markdown('<p class="section-title">Live Signal Summary</p>', unsafe_allow_html=True)

    # Style the dataframe
    def style_signal(val):
        if val == "BUY":  return "background-color:#065f46;color:#6ee7b7;font-weight:700;border-radius:6px"
        if val == "SELL": return "background-color:#7f1d1d;color:#fca5a5;font-weight:700;border-radius:6px"
        return "background-color:#44351a;color:#fcd34d;font-weight:700;border-radius:6px"

    def style_regime(val):
        if val == "HIGH":   return "color:#EF4444;font-weight:600"
        if val == "LOW":    return "color:#10B981;font-weight:600"
        return "color:#F59E0B;font-weight:600"

    def style_change(val):
        try:
            v = float(val)
            return "color:#10B981" if v >= 0 else "color:#EF4444"
        except:
            return ""

    styled_table = (
        signals_df.style
        .map(style_signal,  subset=["Signal"])
        .map(style_regime,  subset=["Regime"])
        .map(style_change,  subset=["Change %"])
        .format({"Price": "{:,.2f}", "RSI": "{:.1f}", "Z-Score": "{:.2f}", "Change %": "{:+.2f}"})
    )
    st.dataframe(styled_table, width='stretch', hide_index=True, height=200)

    # Export
    csv = signals_df.to_csv(index=False)
    st.download_button("⬇️ Export Signals CSV", data=csv,
                       file_name="aeraquant_signals.csv", mime="text/csv")

    # ── Per-Asset Deep Dive ───────────────────────────────────────────────────
    st.markdown('<p class="section-title">Asset Deep Dive — 5Y History</p>', unsafe_allow_html=True)

    # Sector-grouped label for selectbox options
    sector_grouped_assets = []
    for s, alist in sectors.items():
        for a in alist:
            if a in prices:
                sector_grouped_assets.append(f"{SECTOR_ICONS.get(s,'')} {a}")

    selected_label = st.selectbox("Select Asset for Deep Dive", sector_grouped_assets, label_visibility="collapsed")
    selected = selected_label.split(" ", 1)[1] if " " in selected_label else selected_label

    price_s = prices[selected].tail(1260)   # 5Y
    rsi_s   = rsi_dict[selected].tail(1260)
    vol_s   = hist_vol[selected].tail(1260)
    bb_s    = bb_dict[selected].tail(1260)
    reg_s   = regime_dict[selected].tail(1260)

    # ── Row 1: Price + Bollinger Bands | RSI ─────────────────────────────────
    c1, c2 = st.columns(2)

    DARK_BG   = "#0A0E1A"
    GRID_CLR  = "rgba(255,255,255,0.05)"
    CYAN      = "#00D4FF"
    ORANGE    = "#F59E0B"
    PURPLE    = "#8B5CF6"
    GREEN     = "#10B981"
    RED       = "#EF4444"

    layout_base = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(family="Inter", color="#E2E8F0", size=11),
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
    )

    with c1:
        fig1 = go.Figure()
        # Bollinger Band fill
        fig1.add_trace(go.Scatter(
            x=list(bb_s.index) + list(bb_s.index[::-1]),
            y=list(bb_s["Upper"]) + list(bb_s["Lower"][::-1]),
            fill="toself", fillcolor="rgba(0,212,255,0.05)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="BB Band"
        ))
        fig1.add_trace(go.Scatter(x=bb_s.index, y=bb_s["Upper"],
            line=dict(color="rgba(0,212,255,0.3)", dash="dot", width=1),
            name="BB Upper", showlegend=False))
        fig1.add_trace(go.Scatter(x=bb_s.index, y=bb_s["Lower"],
            line=dict(color="rgba(0,212,255,0.3)", dash="dot", width=1),
            name="BB Lower", showlegend=False))
        fig1.add_trace(go.Scatter(x=price_s.index, y=price_s.values,
            line=dict(color=CYAN, width=2), name="Price"))
        fig1.add_trace(go.Scatter(x=bb_s.index, y=bb_s["Mid"],
            line=dict(color="rgba(226,232,240,0.3)", dash="dash", width=1),
            name="20-MA"))
        fig1.update_layout(**layout_base, height=300, title=f"{selected} · Price + Bollinger Bands (1Y)")
        st.plotly_chart(fig1, width='stretch')

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s.values,
            line=dict(color=ORANGE, width=2), name="RSI(14)"))
        fig2.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.07, line_width=0)
        fig2.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.07, line_width=0)
        fig2.add_hline(y=70, line_dash="dash", line_color=RED,   line_width=1,
                       annotation_text="Overbought 70", annotation_font_color=RED)
        fig2.add_hline(y=30, line_dash="dash", line_color=GREEN, line_width=1,
                       annotation_text="Oversold 30", annotation_font_color=GREEN)
        fig2.update_layout(**layout_base, height=300, title="RSI (14-Period)")
        fig2.update_yaxes(range=[0, 100])
        st.plotly_chart(fig2, width='stretch')

    # ── Row 2: GARCH Historical Vol | 5-Day Forecast ─────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=vol_s.index, y=vol_s.values,
            line=dict(color=PURPLE, width=2), name="GARCH Vol", fill="tozeroy",
            fillcolor="rgba(139,92,246,0.08)"))
        fig3.update_layout(**layout_base, height=300, title="GARCH(1,1) Conditional Volatility")
        st.plotly_chart(fig3, width='stretch')

    with c4:
        fc_vals = garch_dict[selected]["forecast"]
        days    = [f"Day {i+1}" for i in range(len(fc_vals))]
        bar_colors = [RED if v > np.mean(fc_vals) else GREEN for v in fc_vals]
        fig4 = go.Figure(go.Bar(
            x=days, y=fc_vals,
            marker_color=bar_colors,
            marker_line_color="rgba(0,0,0,0)",
            text=[f"{v:.2f}%" for v in fc_vals],
            textposition="outside",
            textfont=dict(color="#E2E8F0", size=10),
        ))
        fig4.update_layout(**layout_base, height=300, title="5-Day Volatility Forecast (GARCH)")
        st.plotly_chart(fig4, width='stretch')

    # ── Regime Timeline ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Volatility Regime Timeline</p>', unsafe_allow_html=True)

    regime_color = {"HIGH": RED, "NORMAL": ORANGE, "LOW": GREEN}
    reg_tail = regime_dict[selected].tail(1260)

    fig_reg = go.Figure()
    for regime_label, color in regime_color.items():
        mask = reg_tail == regime_label
        x_vals = reg_tail[mask].index
        y_vals = vol_s.reindex(x_vals)
        fig_reg.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.8),
            name=regime_label
        ))
    fig_reg.add_trace(go.Scatter(x=vol_s.index, y=vol_s.values,
        line=dict(color="rgba(226,232,240,0.2)", width=1),
        name="Vol", showlegend=False))
    fig_reg.update_layout(**layout_base, height=220,
        title=f"{selected} · Regime Classification (HIGH=Red, NORMAL=Orange, LOW=Green)")
    st.plotly_chart(fig_reg, width='stretch')

    # ── Cross-Asset Spread Z-Scores ───────────────────────────────────────────
    st.markdown('<p class="section-title">Cross-Asset Spread Analysis (Z-Score)</p>',
                unsafe_allow_html=True)

    if spreads:
        spread_cols = st.columns(min(3, len(spreads)))
        for idx, (pair_name, spread_df) in enumerate(spreads.items()):
            col_target = spread_cols[idx % min(3, len(spreads))]
            with col_target:
                zs = spread_df["ZScore"].tail(252)
                current_z = float(zs.iloc[-1]) if not zs.empty else 0.0
                z_color   = RED if current_z > 2 else (GREEN if current_z < -2 else ORANGE)

                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=zs.index, y=zs.values,
                    line=dict(color=CYAN, width=1.5), fill="tozeroy",
                    fillcolor="rgba(0,212,255,0.05)", name="Z-Score"))
                fig5.add_hline(y= 2,  line_dash="dash", line_color=RED,   line_width=1,
                               annotation_text="+2σ SELL", annotation_font_color=RED)
                fig5.add_hline(y=-2,  line_dash="dash", line_color=GREEN, line_width=1,
                               annotation_text="-2σ BUY",  annotation_font_color=GREEN)
                fig5.add_hline(y= 0,  line_dash="dot",  line_color="rgba(255,255,255,0.15)", line_width=1)
                fig5.update_layout(**layout_base, height=240, title=f"{pair_name}")

                fig5.add_annotation(
                    x=zs.index[-1], y=current_z,
                    text=f"  Z={current_z:.2f}",
                    showarrow=False, font=dict(color=z_color, size=12, family="Inter"),
                    xanchor="left"
                )
                st.plotly_chart(fig5, width='stretch')
    else:
        st.info("Not enough assets loaded to compute spread pairs.")

    # ── VIX Historical Chart (5Y) ───────────────────────────────────────────
    st.markdown('<p class="section-title">VIX — 5Y History</p>', unsafe_allow_html=True)

    if not vix_series.empty:
        vix_tail = vix_series.tail(1260)
        fig_vix = go.Figure()

        # Background bands
        fig_vix.add_hrect(y0=0,  y1=15, fillcolor="rgba(16,185,129,0.05)", line_width=0)
        fig_vix.add_hrect(y0=15, y1=25, fillcolor="rgba(245,158,11,0.05)", line_width=0)
        fig_vix.add_hrect(y0=25, y1=100, fillcolor="rgba(239,68,68,0.05)", line_width=0)

        fig_vix.add_trace(go.Scatter(
            x=vix_tail.index, y=vix_tail.values,
            line=dict(color="#00D4FF", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.04)",
            name="VIX"
        ))

        # Threshold lines
        fig_vix.add_hline(y=15, line_dash="dash", line_color=GREEN, line_width=1,
                          annotation_text="15 — Risk-ON", annotation_font_color=GREEN)
        fig_vix.add_hline(y=25, line_dash="dash", line_color=RED,   line_width=1,
                          annotation_text="25 — Fear/Risk-OFF", annotation_font_color=RED)

        fig_vix.update_layout(**layout_base, height=260,
            title="VIX (CBOE Volatility Index) — 5Y · Green=Calm | Orange=Caution | Red=Fear")
        st.plotly_chart(fig_vix, width='stretch')


# ==============================================================================
# TAB 2: BACKTESTING & STRATEGY ENGINE
# ==============================================================================
with tab_backtest:
    st.markdown('<p class="section-title">Strategy Backtester</p>', unsafe_allow_html=True)

    if not spreads:
        st.info("No cointegrated spreads available to backtest.")
    else:
        # Layout parameters
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            selected_pair = st.selectbox("Select Spread Pair to Cointegrate", list(spreads.keys()))
        with bc2:
            entry_z = st.slider("Strategy Entry Z-Score Threshold (σ)", 1.0, 3.0, 2.0, 0.1)
        with bc3:
            exit_z = st.slider("Strategy Target Exit Z-Score (σ)", -1.0, 1.0, 0.0, 0.1)

        # Run strategy
        backtest_result = run_spread_backtest(spreads[selected_pair], entry_z, exit_z)

        if not backtest_result:
            st.error("Backtest failed due to missing metrics.")
        else:
            # Performance indicators
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            
            # Color coding
            ret_color = "green" if backtest_result["total_return"] >= 0 else "red"
            win_color = "green" if backtest_result["win_rate"] >= 50 else "yellow"
            sharpe_val = backtest_result["sharpe"]
            sh_color = "cyan" if sharpe_val >= 2.0 else ("green" if sharpe_val >= 1.0 else "yellow")

            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Strategy Return</div>
                    <div class="metric-value {ret_color}">{backtest_result['total_return']:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Annualized Return</div>
                    <div class="metric-value {ret_color}">{backtest_result['ann_return']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {sh_color}">{backtest_result['sharpe']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value red">{backtest_result['max_dd']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with mc5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Win Rate / Trades</div>
                    <div class="metric-value {win_color}">{backtest_result['win_rate']:.1f}% <span style="font-size:12px;color:rgba(226,232,240,0.5)">({backtest_result['total_trades']} trades)</span></div>
                </div>
                """, unsafe_allow_html=True)

            # Plots
            bdf = backtest_result["df"]
            
            pc1, pc2 = st.columns(2)
            
            with pc1:
                # Equity Curve
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=bdf.index, y=bdf["Equity_Curve"],
                    line=dict(color="#00D4FF", width=2),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
                    name="Z-Score Strategy"
                ))
                
                # Benchmark Buy and Hold of the spread
                bench_equity = bdf["Ratio"] / bdf["Ratio"].iloc[0]
                fig_equity.add_trace(go.Scatter(
                    x=bdf.index, y=bench_equity,
                    line=dict(color="rgba(226,232,240,0.3)", width=1, dash="dash"),
                    name="Spread Ratio Benchmark"
                ))
                
                fig_equity.update_layout(**layout_base, height=350, title=f"Strategy Equity Curve (Growth of 1.0) vs Spread Benchmark")
                st.plotly_chart(fig_equity, width='stretch')

            with pc2:
                # Position State Timeline
                fig_pos = go.Figure()
                fig_pos.add_trace(go.Scatter(
                    x=bdf.index, y=bdf["Position"],
                    line=dict(color="#8B5CF6", width=1.5),
                    fill="tozeroy", fillcolor="rgba(139,92,246,0.06)",
                    name="Position State"
                ))
                fig_pos.update_layout(**layout_base, height=350, title="Strategy Execution State Timeline (+1=Long Spread, -1=Short Spread, 0=Cash)")
                fig_pos.update_yaxes(tickvals=[-1, 0, 1], ticktext=["SHORT", "CASH", "LONG"])
                st.plotly_chart(fig_pos, width='stretch')

            # Show Execution Logs (Last 10 trades/signals)
            st.markdown('<p class="section-title">Execution Log — Recent Trades</p>', unsafe_allow_html=True)
            log_cols = ["Ratio", "ZScore", "Position", "Strategy_Return", "Equity_Curve"]
            log_df = bdf[log_cols].tail(15).copy()
            log_df["Position"] = log_df["Position"].map({1: "LONG", -1: "SHORT", 0: "CASH"})
            log_df["Strategy_Return"] = log_df["Strategy_Return"] * 100
            
            styled_log = (
                log_df.style
                .map(lambda v: "color:#10B981" if v in ["LONG"] else ("color:#EF4444" if v in ["SHORT"] else ""), subset=["Position"])
                .format({"Ratio": "{:.4f}", "ZScore": "{:.2f}", "Strategy_Return": "{:+.2f}%", "Equity_Curve": "{:.4f}"})
            )
            st.dataframe(styled_log, width='stretch', height=240)


# ==============================================================================
# TAB 3: QUANTITATIVE METHODOLOGY & DOCUMENTATION
# ==============================================================================
with tab_docs:
    st.markdown('<p class="section-title">Strategy Methodology</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Cointegration & Spread Trading
    Pairs trading is an market-neutral investment strategy that exploits deviations from a long-term equilibrium. When two assets are cointegrated, their price ratio $S_t = P_{A,t} / P_{B,t}$ forms a stationary series, which means it tends to revert to a constant mean over time.
    
    The system measures this stretch using the rolling Z-score:
    
    $$Z_t = \\frac{S_t - \\mu_{t, n}}{\\sigma_{t, n}}$$
    
    Where:
    - $\\mu_{t, n}$ is the $n$-period rolling average of the price ratio.
    - $\\sigma_{t, n}$ is the $n$-period rolling standard deviation of the price ratio.
    
    **Execution Protocol:**
    - **Long Spread (Z < −Entry Threshold):** Buy Asset A, Short Asset B (expect the ratio to rise).
    - **Short Spread (Z > +Entry Threshold):** Sell Asset A, Buy Asset B (expect the ratio to fall).
    - **Mean Reversion Target (Z = Exit Threshold):** Close all positions as the ratio reverts back to its historical mean.

    ---

    ### 2. GARCH(1,1) Volatility Modeling
    To measure conditional volatility and identify volatility regimes, we fit a Generalized Autoregressive Conditional Heteroskedasticity model (GARCH) with Student-t innovations to accommodate the fat tails of daily returns:
    
    $$\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$
    
    Where:
    - $\\sigma_t^2$ is the conditional variance for day $t$.
    - $\\omega$ is the constant baseline variance.
    - $\\alpha$ measures the impact of yesterday's shock (ARCH term).
    - $\\beta$ measures the persistence of yesterday's volatility (GARCH term).
    
    We feed the model with 5 years of daily observations (~1,260 data points) to ensure parameter stability.

    ---

    ### 3. VIX Global Risk Overlay
    The VIX Index reflects the market's expectation of 30-day implied volatility derived from S&P 500 options prices. We utilize it as a macro regime filter:
    - **VIX < 15 (Calm):** Risk-on. Cointegration strategies operate at normal size.
    - **VIX 15 - 25 (Caution):** Increased macro uncertainty. 
    - **VIX > 25 (Fear):** Global risk-off. Cointegration patterns are highly susceptible to structural correlation break-downs (spreads can diverge indefinitely). All **BUY** signals are globally overridden to **HOLD** to avoid catching falling knives during market crashes.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AeraQuant: Derivatives Volatility & Spread Analytics &nbsp;|&nbsp;
    Built by <strong>Ankush Jaiswal</strong> &nbsp;|&nbsp;
    Stack: Python · yfinance · ARCH · Streamlit · Plotly &nbsp;|&nbsp;
    ⚠️ Educational purposes only — not financial advice
</div>
""", unsafe_allow_html=True)

