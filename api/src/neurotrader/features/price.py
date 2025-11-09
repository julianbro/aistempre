"""
Price-based features.
"""

import pandas as pd
import numpy as np


def compute_log_returns(
    df: pd.DataFrame,
    periods: int = 1,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute log returns.
    
    Args:
        df: Input DataFrame
        periods: Number of periods for return calculation
        column: Price column
        
    Returns:
        DataFrame with log return column
    """
    log_returns = np.log(df[column] / df[column].shift(periods))
    return pd.DataFrame({f"log_return_{periods}": log_returns}, index=df.index)


def compute_returns(
    df: pd.DataFrame,
    periods: int = 1,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute simple returns.
    
    Args:
        df: Input DataFrame
        periods: Number of periods for return calculation
        column: Price column
        
    Returns:
        DataFrame with return column
    """
    returns = df[column].pct_change(periods)
    return pd.DataFrame({f"return_{periods}": returns}, index=df.index)


def compute_cumulative_returns(
    df: pd.DataFrame,
    window: int = 20,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute cumulative returns over a window.
    
    Args:
        df: Input DataFrame
        window: Window size
        column: Price column
        
    Returns:
        DataFrame with cumulative return column
    """
    cum_return = (df[column] / df[column].shift(window)) - 1
    return pd.DataFrame({f"cum_return_{window}": cum_return}, index=df.index)


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Volume Weighted Average Price.
    
    Args:
        df: Input DataFrame with OHLCV data
        
    Returns:
        DataFrame with VWAP column
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    
    return pd.DataFrame({"vwap": vwap}, index=df.index)


def compute_price_zscore(
    df: pd.DataFrame,
    window: int = 100,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute z-scored price.
    
    Args:
        df: Input DataFrame
        window: Rolling window for mean/std calculation
        column: Price column
        
    Returns:
        DataFrame with z-score column
    """
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    
    zscore = (df[column] - rolling_mean) / rolling_std
    
    return pd.DataFrame({f"price_zscore_{window}": zscore}, index=df.index)


def compute_rolling_mean_gap(
    df: pd.DataFrame,
    window: int = 20,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute gap between price and rolling mean.
    
    Args:
        df: Input DataFrame
        window: Rolling window
        column: Price column
        
    Returns:
        DataFrame with gap column
    """
    rolling_mean = df[column].rolling(window=window).mean()
    gap = (df[column] - rolling_mean) / rolling_mean
    
    return pd.DataFrame({f"mean_gap_{window}": gap}, index=df.index)


def compute_price_momentum(
    df: pd.DataFrame,
    window: int = 10,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute price momentum.
    
    Args:
        df: Input DataFrame
        window: Momentum window
        column: Price column
        
    Returns:
        DataFrame with momentum column
    """
    momentum = df[column] - df[column].shift(window)
    return pd.DataFrame({f"momentum_{window}": momentum}, index=df.index)


def compute_price_acceleration(
    df: pd.DataFrame,
    window: int = 10,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute price acceleration (second derivative).
    
    Args:
        df: Input DataFrame
        window: Window for acceleration
        column: Price column
        
    Returns:
        DataFrame with acceleration column
    """
    velocity = df[column].diff()
    acceleration = velocity.diff(window)
    
    return pd.DataFrame({f"acceleration_{window}": acceleration}, index=df.index)
