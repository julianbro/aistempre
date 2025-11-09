"""
Threshold utilities for trend classification.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def optimize_epsilon_threshold(
    df: pd.DataFrame,
    horizon: str = "30m",
    min_epsilon: float = 5.0,
    max_epsilon: float = 50.0,
    step: float = 5.0,
    target_flat_ratio: float = 0.33,
) -> float:
    """
    Find optimal epsilon threshold to achieve target FLAT class ratio.
    
    Args:
        df: Input DataFrame with price data
        horizon: Time horizon
        min_epsilon: Minimum epsilon (bps)
        max_epsilon: Maximum epsilon (bps)
        step: Step size for search
        target_flat_ratio: Target ratio of FLAT samples
        
    Returns:
        Optimal epsilon value
    """
    from neurotrader.labels.targets import compute_trend_targets
    
    best_epsilon = min_epsilon
    best_diff = float("inf")
    
    for epsilon in np.arange(min_epsilon, max_epsilon + step, step):
        labels = compute_trend_targets(df, horizon=horizon, epsilon_bps=epsilon)
        flat_ratio = (labels == 1).sum() / len(labels)
        
        diff = abs(flat_ratio - target_flat_ratio)
        if diff < best_diff:
            best_diff = diff
            best_epsilon = epsilon
    
    return best_epsilon


def get_dynamic_thresholds(
    df: pd.DataFrame,
    window: int = 100,
    percentile: float = 0.25,
    column: str = "close",
) -> pd.Series:
    """
    Compute dynamic epsilon thresholds based on volatility.
    
    Args:
        df: Input DataFrame
        window: Rolling window for volatility
        percentile: Percentile of volatility to use as epsilon
        column: Price column
        
    Returns:
        Series with dynamic epsilon thresholds (in bps)
    """
    # Compute realized volatility
    returns = np.log(df[column] / df[column].shift(1))
    volatility = returns.rolling(window=window).std()
    
    # Convert to bps
    epsilon_bps = volatility * 10000 * percentile
    
    return epsilon_bps


def stratify_by_threshold(
    returns_bps: pd.Series,
    epsilon_bps: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Stratify returns into UP/DOWN/FLAT groups.
    
    Args:
        returns_bps: Returns in basis points
        epsilon_bps: Epsilon threshold
        
    Returns:
        Tuple of (up_mask, flat_mask, down_mask)
    """
    up_mask = returns_bps > epsilon_bps
    down_mask = returns_bps < -epsilon_bps
    flat_mask = ~(up_mask | down_mask)
    
    return up_mask, flat_mask, down_mask


def analyze_threshold_distribution(
    df: pd.DataFrame,
    horizon: str = "30m",
    epsilon_values: list[float] = None,
    column: str = "close",
) -> pd.DataFrame:
    """
    Analyze class distribution for different epsilon values.
    
    Args:
        df: Input DataFrame
        horizon: Time horizon
        epsilon_values: List of epsilon values to test
        column: Price column
        
    Returns:
        DataFrame with epsilon and class distributions
    """
    from neurotrader.labels.targets import compute_trend_targets
    
    if epsilon_values is None:
        epsilon_values = [5, 10, 15, 20, 25, 30, 40, 50]
    
    results = []
    
    for epsilon in epsilon_values:
        labels = compute_trend_targets(
            df, horizon=horizon, epsilon_bps=epsilon, column=column
        )
        
        counts = labels.value_counts().sort_index()
        total = len(labels.dropna())
        
        results.append({
            "epsilon_bps": epsilon,
            "down_count": counts.get(0, 0),
            "flat_count": counts.get(1, 0),
            "up_count": counts.get(2, 0),
            "down_ratio": counts.get(0, 0) / total,
            "flat_ratio": counts.get(1, 0) / total,
            "up_ratio": counts.get(2, 0) / total,
        })
    
    return pd.DataFrame(results)
