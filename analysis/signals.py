"""
Signal Engine
Generates BUY/SELL/HOLD signals based on RSI, Z-Score, and volatility regimes.
"""

import pandas as pd
import numpy as np
from analysis.spreads import get_zscore_for_asset


# ── Signal Thresholds ─────────────────────────────────────────────────────────
RSI_OVERSOLD  = 35
RSI_OVERBOUGHT = 65
ZSCORE_BUY    = -1.5
ZSCORE_SELL   = +1.5


# ── Single-Asset Signal ───────────────────────────────────────────────────────
def generate_signal(rsi_val: float, regime: str, zscore: float) -> dict:
    """
    Combines RSI, regime, and Z-score into a final signal.

    Returns dict:
      signal     — "BUY" | "SELL" | "HOLD"
      confidence — int 0-100
      reasons    — list of str describing conditions
    """
    buy_flags  = []
    sell_flags = []

    # RSI
    if rsi_val < RSI_OVERSOLD:
        buy_flags.append(f"RSI oversold ({rsi_val:.1f} < {RSI_OVERSOLD})")
    elif rsi_val > RSI_OVERBOUGHT:
        sell_flags.append(f"RSI overbought ({rsi_val:.1f} > {RSI_OVERBOUGHT})")

    # Z-Score spread
    if zscore < ZSCORE_BUY:
        buy_flags.append(f"Spread compressed (Z={zscore:.2f})")
    elif zscore > ZSCORE_SELL:
        sell_flags.append(f"Spread extended (Z={zscore:.2f})")

    # Regime
    if regime == "LOW":
        buy_flags.append("Low-vol regime (favours mean reversion)")
    elif regime == "HIGH":
        sell_flags.append("High-vol regime (risk-off)")

    # Final decision: need at least 2 buy flags for BUY (confluence)
    if len(buy_flags) >= 2 and regime != "HIGH":
        signal     = "BUY"
        confidence = min(100, 40 + len(buy_flags) * 20)
        reasons    = buy_flags
    elif len(sell_flags) >= 2:
        signal     = "SELL"
        confidence = min(100, 40 + len(sell_flags) * 20)
        reasons    = sell_flags
    else:
        signal     = "HOLD"
        confidence = 50
        reasons    = buy_flags + sell_flags if (buy_flags or sell_flags) else ["Neutral conditions"]

    return {
        "signal":     signal,
        "confidence": confidence,
        "reasons":    " | ".join(reasons),
    }


# ── All Assets Signal Table ───────────────────────────────────────────────────
def get_signals_for_all(prices: dict, rsi_dict: dict,
                         regime_dict: dict, spreads: dict) -> pd.DataFrame:
    """
    Generates the full signal summary DataFrame with one row per asset.

    Columns: Asset, Price, Change%, RSI, Regime, ZScore, Signal, Confidence, Reason
    """
    rows = []

    for asset, price_series in prices.items():
        if price_series.empty or len(price_series) < 2:
            continue

        price_now  = float(price_series.iloc[-1])
        price_prev = float(price_series.iloc[-2])
        change_pct = ((price_now - price_prev) / price_prev * 100) if price_prev != 0 else 0.0

        rsi_series    = rsi_dict.get(asset, pd.Series(dtype=float))
        regime_series = regime_dict.get(asset, pd.Series(dtype=str))

        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
        regime  = str(regime_series.iloc[-1]) if not regime_series.empty else "NORMAL"
        zscore  = get_zscore_for_asset(asset, spreads)

        sig = generate_signal(rsi_val, regime, zscore)

        rows.append({
            "Asset":      asset,
            "Price":      round(price_now, 2),
            "Change %":   round(change_pct, 2),
            "RSI":        round(rsi_val, 1),
            "Regime":     regime,
            "Z-Score":    round(zscore, 2),
            "Signal":     sig["signal"],
            "Confidence": f"{sig['confidence']}%",
            "Reason":     sig["reasons"],
        })

    return pd.DataFrame(rows)
