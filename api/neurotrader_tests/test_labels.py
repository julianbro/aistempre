"""
Test label generation.
"""

import pytest
import pandas as pd
import numpy as np

from neurotrader.labels.targets import (
    compute_next_return_target,
    compute_trend_targets,
    compute_multi_horizon_targets,
)


def create_test_price_data(n_days=100):
    """Create test price data."""
    dates = pd.date_range(
        start="2020-01-01",
        periods=n_days * 24,  # Hourly data
        freq="H",
        tz="UTC",
    )
    
    # Random walk
    returns = np.random.normal(0.0001, 0.01, len(dates))
    log_prices = np.log(100.0) + np.cumsum(returns)
    prices = np.exp(log_prices)
    
    df = pd.DataFrame({"close": prices}, index=dates)
    
    return df


def test_next_return_target():
    """Test next return target computation."""
    df = create_test_price_data()
    
    # Log returns
    log_returns = compute_next_return_target(df, horizon=1, log_return=True)
    
    assert len(log_returns) == len(df)
    assert not np.all(np.isnan(log_returns[:-1]))  # All except last should be valid
    
    # Simple returns
    simple_returns = compute_next_return_target(df, horizon=1, log_return=False)
    
    assert len(simple_returns) == len(df)


def test_trend_targets():
    """Test trend classification targets."""
    df = create_test_price_data()
    
    # Short-term trend
    short_trend = compute_trend_targets(df, horizon="4h", epsilon_bps=10.0)
    
    assert len(short_trend) == len(df)
    
    # Should have 3 classes
    unique_labels = short_trend.dropna().unique()
    assert len(unique_labels) <= 3
    assert all(label in [0, 1, 2] for label in unique_labels)
    
    # Check class distribution
    counts = short_trend.value_counts()
    assert counts.sum() > 0


def test_multi_horizon_targets():
    """Test multi-horizon target generation."""
    df = create_test_price_data()
    
    targets = compute_multi_horizon_targets(
        df,
        next_horizon=1,
        short_horizon="4h",
        long_horizon="1d",
        epsilon_bps=10.0,
    )
    
    assert "next_return" in targets.columns
    assert "short_trend" in targets.columns
    assert "long_trend" in targets.columns
    
    assert len(targets) == len(df)
    
    # Check that we have both regression and classification targets
    assert targets["next_return"].dtype == np.float64
    assert targets["short_trend"].dtype in [np.int64, np.float64]
    assert targets["long_trend"].dtype in [np.int64, np.float64]


def test_epsilon_threshold_effect():
    """Test that epsilon threshold affects class distribution."""
    df = create_test_price_data()
    
    # Small epsilon -> more UP/DOWN
    trend_small = compute_trend_targets(df, horizon="4h", epsilon_bps=5.0)
    flat_ratio_small = (trend_small == 1).sum() / len(trend_small)
    
    # Large epsilon -> more FLAT
    trend_large = compute_trend_targets(df, horizon="4h", epsilon_bps=50.0)
    flat_ratio_large = (trend_large == 1).sum() / len(trend_large)
    
    # Larger epsilon should result in more FLAT labels
    assert flat_ratio_large > flat_ratio_small
