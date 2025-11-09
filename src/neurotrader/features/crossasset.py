"""
Cross-asset features (requires multiple symbols).
"""

import pandas as pd
import numpy as np


def compute_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    window: int = 60,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute rolling correlation between two assets.
    
    Args:
        df1: First asset DataFrame
        df2: Second asset DataFrame
        window: Rolling window
        column: Price column
        
    Returns:
        DataFrame with correlation
    """
    returns1 = df1[column].pct_change()
    returns2 = df2[column].pct_change()
    
    correlation = returns1.rolling(window=window).corr(returns2)
    
    return pd.DataFrame({f"correlation_{window}": correlation}, index=df1.index)


def compute_spread(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute price spread between two assets.
    
    Args:
        df1: First asset DataFrame
        df2: Second asset DataFrame
        column: Price column
        
    Returns:
        DataFrame with spread features
    """
    spread = df1[column] - df2[column]
    spread_pct = (df1[column] - df2[column]) / df2[column]
    
    return pd.DataFrame(
        {
            "spread": spread,
            "spread_pct": spread_pct,
        },
        index=df1.index,
    )


def compute_ratio(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    window: int = 20,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute price ratio and its z-score.
    
    Args:
        df1: Numerator asset DataFrame
        df2: Denominator asset DataFrame
        window: Window for z-score
        column: Price column
        
    Returns:
        DataFrame with ratio features
    """
    ratio = df1[column] / df2[column]
    
    ratio_mean = ratio.rolling(window=window).mean()
    ratio_std = ratio.rolling(window=window).std()
    ratio_zscore = (ratio - ratio_mean) / ratio_std
    
    return pd.DataFrame(
        {
            "ratio": ratio,
            "ratio_zscore": ratio_zscore,
        },
        index=df1.index,
    )
