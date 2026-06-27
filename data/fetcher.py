"""
data/fetcher.py
---------------
Fetches live OHLCV data for 12 futures assets and the VIX.
Default history is 5 years to ensure sufficient data points for GARCH(1,1) estimation.
"""

import yfinance as yf
import pandas as pd
import time

# Asset Registry
ASSETS = {
    # ── ENERGY ──────────────────────────────────────────────────
    "Crude Oil":   {"ticker": "CL=F",     "label": "CL",  "exchange": "NYMEX",  "sector": "Energy"},
    "Natural Gas": {"ticker": "NG=F",     "label": "NG",  "exchange": "NYMEX",  "sector": "Energy"},
    # ── METALS ──────────────────────────────────────────────────
    "Gold":        {"ticker": "GC=F",     "label": "GC",  "exchange": "COMEX",  "sector": "Metals"},
    "Silver":      {"ticker": "SI=F",     "label": "SI",  "exchange": "COMEX",  "sector": "Metals"},
    "Copper":      {"ticker": "HG=F",     "label": "HG",  "exchange": "COMEX",  "sector": "Metals"},
    # ── AGRICULTURE ─────────────────────────────────────────────
    "Corn":        {"ticker": "ZC=F",     "label": "ZC",  "exchange": "CBOT",   "sector": "Agriculture"},
    "Wheat":       {"ticker": "ZW=F",     "label": "ZW",  "exchange": "CBOT",   "sector": "Agriculture"},
    # ── EQUITY INDICES ───────────────────────────────────────────
    "S&P 500":     {"ticker": "ES=F",     "label": "ES",  "exchange": "CME",    "sector": "Equity"},
    "Nasdaq 100":  {"ticker": "NQ=F",     "label": "NQ",  "exchange": "CME",    "sector": "Equity"},
    # ── INTEREST RATES ───────────────────────────────────────────
    "10Y T-Note":  {"ticker": "ZN=F",     "label": "ZN",  "exchange": "CBOT",   "sector": "Rates"},
    # ── FX ──────────────────────────────────────────────────────
    "EUR/USD":     {"ticker": "EURUSD=X", "label": "EU",  "exchange": "Forex",  "sector": "FX"},
    "GBP/USD":     {"ticker": "GBPUSD=X", "label": "GBP", "exchange": "Forex",  "sector": "FX"},
}

# VIX fetched separately
VIX_TICKER = "^VIX"


# Core Fetcher
def fetch_price_series(ticker: str, period: str = "5y", interval: str = "1d",
                        retries: int = 3) -> pd.Series:
    """
    Download closing price series for a given ticker.
    Retries up to `retries` times on failure with exponential backoff.
    """
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            close = df["Close"]
            if isinstance(getattr(close, "columns", None), pd.MultiIndex):
                close = close.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()

            return close.dropna().rename(ticker)

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"[WARN] Failed to fetch {ticker} after {retries} attempts: {e}")
                return pd.Series(dtype=float, name=ticker)


def fetch_all(period: str = "5y") -> dict:
    """
    Fetch price series for all 12 assets.
    Returns dict: {asset_name: pd.Series}
    """
    result = {}
    for name, meta in ASSETS.items():
        series = fetch_price_series(meta["ticker"], period=period)
        if not series.empty:
            result[name] = series
        else:
            print(f"[WARN] Skipping {name} — data unavailable")
    return result


def fetch_vix(period: str = "5y") -> pd.Series:
    """
    Fetch VIX (CBOE Volatility Index) as a macro regime overlay.
    VIX < 15 = calm, 15-25 = caution, > 25 = fear/risk-off.
    """
    return fetch_price_series(VIX_TICKER, period=period)


def get_asset_metadata() -> dict:
    """Returns ticker/exchange/sector info per asset for dashboard display."""
    return {
        name: {
            "ticker":   meta["ticker"],
            "exchange": meta["exchange"],
            "sector":   meta["sector"],
        }
        for name, meta in ASSETS.items()
    }


def get_sectors() -> dict:
    """Returns dict mapping sector → list of asset names."""
    sectors: dict = {}
    for name, meta in ASSETS.items():
        s = meta["sector"]
        sectors.setdefault(s, []).append(name)
    return sectors
