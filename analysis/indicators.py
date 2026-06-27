"""
Technical indicators and volatility modeling.
Includes RSI, GARCH(1,1), Regime Classification, and Bollinger Bands.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn import hmm


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


# ── Regime Classifier (Hidden Markov Model) ───────────────────────────────────
def classify_regime(price_series: pd.Series, n_states: int = 3) -> pd.Series:
    """
    Classifies each date into LOW / NORMAL / HIGH volatility regime
    using an Unsupervised Hidden Markov Model (GaussianHMM) on log returns.
    """
    # Calculate log returns
    rets = np.log(price_series / price_series.shift(1)).dropna().values.reshape(-1, 1)
    
    if len(rets) < 100:
        return pd.Series("NORMAL", index=price_series.index, name="Regime")
        
    try:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(rets)
        hidden_states = model.predict(rets)
        
        # Sort states by variance to identify LOW, NORMAL, HIGH
        variances = np.array([np.diag(model.covars_[i]) for i in range(n_states)]).squeeze()
        
        if n_states == 3:
            sorted_idx = np.argsort(variances)
            state_map = {sorted_idx[0]: "LOW", sorted_idx[1]: "NORMAL", sorted_idx[2]: "HIGH"}
        else:
            state_map = {i: f"STATE_{i}" for i in range(n_states)}
            
        # Pad the first element (which was dropped due to NaN return)
        regimes = [state_map[hidden_states[0]]] + [state_map[s] for s in hidden_states]
        return pd.Series(regimes, index=price_series.index, name="Regime")
    except Exception as e:
        print(f"[WARN] HMM fit failed: {e}")
        return pd.Series("NORMAL", index=price_series.index, name="Regime")

# ── Hurst Exponent ────────────────────────────────────────────────────────────
def compute_hurst(series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate the Hurst Exponent of a time series to detect mean-reversion.
    H < 0.5: Mean-reverting (Good for Pairs Trading)
    H = 0.5: Random Walk
    H > 0.5: Trending
    """
    if len(series) < max_lag * 2:
        return 0.5
        
    lags = range(2, max_lag)
    # Calculate variance of differences at various lags
    tau = [np.sqrt(np.std(np.subtract(series.values[lag:], series.values[:-lag]))) for lag in lags]
    
    # Fit line to log-log plot
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Hurst exponent is the slope * 2
    return poly[0] * 2.0


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
