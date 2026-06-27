"""
Cross-asset spread analysis.
Calculates rolling Z-scores for predefined asset pairs to identify mean-reversion opportunities.
"""

import pandas as pd
import numpy as np


# ── Rolling Z-Score ───────────────────────────────────────────────────────────
def compute_rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-score = (value - rolling_mean) / rolling_std
    Measures how many std devs current value is from its recent average.
    """
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).rename("ZScore")


# ── Cross-Asset Spread ────────────────────────────────────────────────────────
def compute_cross_asset_spread(s1: pd.Series, s2: pd.Series,
                                name1: str, name2: str,
                                window: int = 20) -> pd.DataFrame:
    """
    Computes the price ratio between two assets, then its rolling Z-score.

    Example: Gold/Crude ratio — if unusually high, Gold is expensive relative
    to Crude; a mean-reversion trade would be to sell Gold / buy Crude.

    Returns DataFrame with columns:
      name1, name2, Ratio, ZScore
    """
    aligned = pd.concat([s1, s2], axis=1).dropna()
    aligned.columns = [name1, name2]

    # Ratio spread
    aligned["Ratio"]  = aligned[name1] / aligned[name2]
    aligned["ZScore"] = compute_rolling_zscore(aligned["Ratio"], window=window)

    return aligned


# ── All Spreads ───────────────────────────────────────────────────────────────
def get_all_spreads(prices: dict, window: int = 20) -> dict:
    """
    Compute spread DataFrames for all meaningful cross-asset pairs.

    Returns dict: {pair_name: pd.DataFrame}
    """
    spreads = {}
    available = set(prices.keys())

    # 1. Gold / Crude Oil
    if "Gold" in available and "Crude Oil" in available:
        spreads["Gold / Crude Oil"] = compute_cross_asset_spread(
            prices["Gold"], prices["Crude Oil"], "Gold", "Crude", window=window
        )

    # 2. S&P 500 / Crude Oil
    if "S&P 500" in available and "Crude Oil" in available:
        spreads["S&P 500 / Crude Oil"] = compute_cross_asset_spread(
            prices["S&P 500"], prices["Crude Oil"], "ES", "Crude", window=window
        )

    # 3. Gold / S&P 500
    if "Gold" in available and "S&P 500" in available:
        spreads["Gold / S&P 500"] = compute_cross_asset_spread(
            prices["Gold"], prices["S&P 500"], "Gold", "ES", window=window
        )

    # 4. Gold / Silver (Precious Metals Ratio)
    if "Gold" in available and "Silver" in available:
        spreads["Gold / Silver"] = compute_cross_asset_spread(
            prices["Gold"], prices["Silver"], "Gold", "Silver", window=window
        )

    # 5. S&P 500 / Nasdaq 100 (Index Spread)
    if "S&P 500" in available and "Nasdaq 100" in available:
        spreads["S&P 500 / Nasdaq 100"] = compute_cross_asset_spread(
            prices["S&P 500"], prices["Nasdaq 100"], "ES", "NQ", window=window
        )

    # 6. EUR/USD / GBP/USD (FX Cointegration)
    if "EUR/USD" in available and "GBP/USD" in available:
        spreads["EUR/USD / GBP/USD"] = compute_cross_asset_spread(
            prices["EUR/USD"], prices["GBP/USD"], "EURUSD", "GBPUSD", window=window
        )

    return spreads


# ── Lookup Helper ─────────────────────────────────────────────────────────────
def get_zscore_for_asset(asset_name: str, spreads: dict) -> float:
    """
    Returns the latest Z-score for the most relevant spread pair
    containing the given asset. Used by the signal engine.
    """
    asset_map = {
        "Crude Oil":   "Crude",
        "Gold":        "Gold",
        "S&P 500":     "ES",
        "EUR/USD":     "EURUSD",
        "Silver":      "Silver",
        "Nasdaq 100":  "NQ",
        "GBP/USD":     "GBPUSD",
    }
    mapped = asset_map.get(asset_name, "")
    if not mapped:
        return 0.0

    for pair_name, df in spreads.items():
        if mapped in pair_name.replace("/", "").replace(" ", ""):
            zscore_series = df["ZScore"].dropna()
            if not zscore_series.empty:
                return float(zscore_series.iloc[-1])
    return 0.0  # fallback: neutral

