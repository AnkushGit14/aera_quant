"""
Technical indicators and volatility modeling.
Includes RSI, GARCH(1,1), Regime Classification, and Bollinger Bands.
"""

import numpy as np
import pandas as pd
from arch import arch_model


# ── RSI ───────────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using exponential weighted moving averages.
    RSI > 70 → overbought | RSI < 30 → oversold
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return rsi.rename("RSI")


# ── GARCH Volatility ──────────────────────────────────────────────────────────
def compute_garch_vol(series: pd.Series, forecast_horizon: int = 5) -> dict:
    """
    Fit GARCH(1,1) with Student-t distribution to log returns.
    Returns:
      hist_vol  — pd.Series of conditional volatility (annualised %)
      forecast  — np.array of 5-day ahead vol forecast
      params    — model parameter estimates
    """
    # Log returns in % (ARCH library expects this scale)
    log_ret = (np.log(series / series.shift(1)).dropna() * 100).squeeze()

    if len(log_ret) < 60:
        # Not enough data — return zeros gracefully
        hist_vol = pd.Series(np.zeros(len(series)), index=series.index, name="GARCH_Vol")
        return {"hist_vol": hist_vol, "forecast": np.zeros(forecast_horizon), "params": {}}

    try:
        model = arch_model(log_ret, vol="Garch", p=1, q=1, dist="t", rescale=False)
        res   = model.fit(disp="off", show_warning=False)

        hist_vol  = np.sqrt(res.conditional_volatility).rename("GARCH_Vol")
        fc        = res.forecast(horizon=forecast_horizon, reindex=False)
        fcast_vol = np.sqrt(fc.variance.values[-1])  # shape: (forecast_horizon,)

        return {"hist_vol": hist_vol, "forecast": fcast_vol, "params": res.params}

    except Exception as e:
        print(f"[WARN] GARCH fit failed: {e}")
        hist_vol = pd.Series(np.zeros(len(log_ret)), index=log_ret.index, name="GARCH_Vol")
        return {"hist_vol": hist_vol, "forecast": np.zeros(forecast_horizon), "params": {}}


# ── Regime Classifier ─────────────────────────────────────────────────────────
def classify_regime(hist_vol: pd.Series) -> pd.Series:
    """
    Classifies each date into LOW / NORMAL / HIGH volatility regime
    based on 30th and 70th percentile thresholds of the full history.
    """
    p30 = hist_vol.quantile(0.30)
    p70 = hist_vol.quantile(0.70)

    def _label(v: float) -> str:
        if v <= p30:   return "LOW"
        elif v <= p70: return "NORMAL"
        else:          return "HIGH"

    return hist_vol.apply(_label).rename("Regime")


# ── Bollinger Bands ───────────────────────────────────────────────────────────
def compute_bollinger_bands(series: pd.Series, window: int = 20,
                             num_std: float = 2.0) -> pd.DataFrame:
    """
    Standard Bollinger Bands.
    Returns DataFrame with columns: Mid, Upper, Lower, %B
    %B = (Price - Lower) / (Upper - Lower)  —  0 = lower band, 1 = upper band
    """
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame({
        "Mid":   mid,
        "Upper": upper,
        "Lower": lower,
        "%B":    pct_b,
    }, index=series.index)
