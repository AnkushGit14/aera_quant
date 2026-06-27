"""
Historical Backtesting Engine for Spread Mean-Reversion.
"""

import pandas as pd
import numpy as np


def run_spread_backtest(spread_df: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.0) -> dict:
    """
    Backtests a mean-reversion strategy on a spread ratio.
    
    Rules:
    - Long Spread (Buy A, Sell B) if Z-Score < -entry_z
    - Short Spread (Sell A, Buy B) if Z-Score > entry_z
    - Exit position when Z-Score crosses/reaches exit_z (or changes sign)
    
    Returns a dict with performance metrics and the equity curve series.
    """
    df = spread_df.copy()
    if df.empty or "ZScore" not in df.columns or "Ratio" not in df.columns:
        return {}

    # Calculate log returns of the spread ratio itself
    df["Spread_Return"] = np.log(df["Ratio"] / df["Ratio"].shift(1))
    df.dropna(inplace=True)

    position = 0  # +1: Long, -1: Short, 0: Cash
    positions = []
    trade_signals = [] # for counting trades

    for i in range(len(df)):
        z = df["ZScore"].iloc[i]
        
        # Check exits
        if position == 1 and z >= -exit_z:
            position = 0
        elif position == -1 and z <= exit_z:
            position = 0
            
        # Check entries
        if position == 0:
            if z < -entry_z:
                position = 1
                trade_signals.append(1)
            elif z > entry_z:
                position = -1
                trade_signals.append(-1)
                
        positions.append(position)

    df["Position"] = positions
    # Shift position by 1 day to prevent look-ahead bias (returns are earned the day after taking position)
    df["Strategy_Return"] = df["Position"].shift(1) * df["Spread_Return"]
    df["Strategy_Return"] = df["Strategy_Return"].fillna(0)

    # Cumulative Returns (Equity Curve)
    df["Equity_Curve"] = np.exp(df["Strategy_Return"].cumsum())

    # Calculate Metrics
    returns = df["Strategy_Return"]
    total_return = (df["Equity_Curve"].iloc[-1] - 1.0) * 100 if len(df) > 0 else 0.0
    
    # Annualized Return & Volatility
    n_days = len(df)
    years = n_days / 252.0
    ann_return = (df["Equity_Curve"].iloc[-1] ** (1.0 / years) - 1.0) * 100 if years > 0 and df["Equity_Curve"].iloc[-1] > 0 else 0.0
    ann_vol = returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (assuming risk-free rate is 0)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0

    # Drawdown
    cum_max = df["Equity_Curve"].cummax()
    drawdown = (df["Equity_Curve"] - cum_max) / cum_max * 100
    max_dd = drawdown.min()

    # Win rate of trades
    # Group returns by continuous position runs to define individual trades
    df["Trade_ID"] = (df["Position"] != df["Position"].shift(1)).cumsum()
    trades = df.groupby("Trade_ID").agg(
        Total_Return=("Strategy_Return", "sum"),
        Pos_Type=("Position", "first")
    )
    # Filter only actual trading periods (exclude cash runs)
    active_trades = trades[trades["Pos_Type"] != 0]
    total_trades = len(active_trades)
    wins = len(active_trades[active_trades["Total_Return"] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    return {
        "df": df,
        "total_return": round(total_return, 2),
        "ann_return": round(ann_return, 2),
        "ann_vol": round(ann_vol, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2)
    }
