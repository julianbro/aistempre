"""
Test purged walk-forward splitter.
"""

import numpy as np
import pandas as pd
from neurotrader.data.splitter import PurgedWalkForwardSplitter, TimeSeriesSplitter


def create_test_dataframe(n_days=365):
    """Create test DataFrame with daily data."""
    dates = pd.date_range(
        start="2020-01-01",
        periods=n_days,
        freq="D",
        tz="UTC",
    )

    df = pd.DataFrame(
        {"value": np.random.randn(n_days)},
        index=dates,
    )

    return df


def test_purged_walk_forward_splitter():
    """Test purged walk-forward splitter."""
    df = create_test_dataframe(n_days=365)

    splitter = PurgedWalkForwardSplitter(
        n_splits=5,
        purge_days=7,
        train_ratio=0.7,
    )

    splits = list(splitter.split(df))

    # Should have 5 splits
    assert len(splits) <= 5

    for train_idx, val_idx in splits:
        # Train and val should not overlap
        assert len(set(train_idx) & set(val_idx)) == 0

        # Train should come before val
        assert train_idx.max() < val_idx.min()

        # Check purge gap
        train_end_date = df.index[train_idx[-1]]
        val_start_date = df.index[val_idx[0]]
        gap = (val_start_date - train_end_date).days

        assert gap >= 7


def test_time_series_splitter():
    """Test time series splitter."""
    df = create_test_dataframe(n_days=365)

    splitter = TimeSeriesSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_days=7,
    )

    train_idx, val_idx, test_idx = splitter.split(df)

    # All indices should be unique
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == len(set(all_idx))

    # Train comes before val comes before test
    assert train_idx.max() < val_idx.min()
    assert val_idx.max() < test_idx.min()

    # Check approximate ratios
    total = len(df)
    train_ratio = len(train_idx) / total
    val_ratio = len(val_idx) / total
    test_ratio = len(test_idx) / total

    # Allow some slack due to purging
    assert 0.5 < train_ratio < 0.8
    assert 0.05 < val_ratio < 0.25
    assert 0.05 < test_ratio < 0.25
