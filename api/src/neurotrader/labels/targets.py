"""
Target label generation for prediction tasks.
"""

import numpy as np
import pandas as pd


def compute_next_return_target(
    df: pd.DataFrame,
    horizon: int = 1,
    column: str = "close",
    log_return: bool = True,
) -> pd.Series:
    """
    Compute next-period return target.

    Args:
        df: Input DataFrame with price data
        horizon: Number of periods ahead
        column: Price column
        log_return: Use log returns (True) or simple returns (False)

    Returns:
        Series with return targets
    """
    if log_return:
        target = np.log(df[column].shift(-horizon) / df[column])
    else:
        target = (df[column].shift(-horizon) - df[column]) / df[column]

    return target


def compute_trend_targets(
    df: pd.DataFrame,
    horizon: str = "30m",
    epsilon_bps: float = 10.0,
    column: str = "close",
) -> pd.Series:
    """
    Compute trend classification targets (UP/DOWN/FLAT).

    Args:
        df: Input DataFrame with price data and DatetimeIndex
        horizon: Time horizon (e.g., '30m', '2h', '1d')
        epsilon_bps: Epsilon band in basis points for FLAT class
        column: Price column

    Returns:
        Series with trend labels (0=DOWN, 1=FLAT, 2=UP)
    """
    from neurotrader.utils.time import parse_timeframe

    # Parse horizon to timedelta
    horizon_delta = parse_timeframe(horizon)

    # Find future prices
    future_prices = []
    for idx, timestamp in enumerate(df.index):
        target_time = timestamp + horizon_delta

        # Find closest future timestamp
        future_idx = df.index.searchsorted(target_time)

        if future_idx < len(df):
            future_prices.append(df[column].iloc[future_idx])
        else:
            future_prices.append(np.nan)

    future_prices = pd.Series(future_prices, index=df.index)

    # Compute returns in basis points
    returns_bps = ((future_prices - df[column]) / df[column]) * 10000

    # Classify trends
    labels = pd.Series(1, index=df.index)  # Default to FLAT
    labels[returns_bps > epsilon_bps] = 2  # UP
    labels[returns_bps < -epsilon_bps] = 0  # DOWN

    return labels


def compute_multi_horizon_targets(
    df: pd.DataFrame,
    next_horizon: int = 1,
    short_horizon: str = "30m",
    long_horizon: str = "1w",
    epsilon_bps: float = 10.0,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute all target labels for multi-task learning.

    Args:
        df: Input DataFrame with price data
        next_horizon: Horizon for next-price prediction
        short_horizon: Horizon for short-term trend
        long_horizon: Horizon for long-term trend
        epsilon_bps: Epsilon band in basis points
        column: Price column

    Returns:
        DataFrame with all target columns
    """
    targets = pd.DataFrame(index=df.index)

    # Next-price target (log return)
    targets["next_return"] = compute_next_return_target(
        df, horizon=next_horizon, column=column, log_return=True
    )

    # Short-term trend
    targets["short_trend"] = compute_trend_targets(
        df, horizon=short_horizon, epsilon_bps=epsilon_bps, column=column
    )

    # Long-term trend
    targets["long_trend"] = compute_trend_targets(
        df, horizon=long_horizon, epsilon_bps=epsilon_bps, column=column
    )

    return targets


def get_trend_class_weights(
    labels: pd.Series,
    method: str = "balanced",
) -> np.ndarray:
    """
    Compute class weights for trend classification.

    Args:
        labels: Trend labels (0=DOWN, 1=FLAT, 2=UP)
        method: Weighting method ('balanced' or 'effective')

    Returns:
        Array of class weights [weight_down, weight_flat, weight_up]
    """
    from sklearn.utils.class_weight import compute_class_weight

    # Remove NaN values
    labels_clean = labels.dropna()

    if method == "balanced":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2]),
            y=labels_clean.values,
        )
    else:
        # Custom effective number of samples weighting
        class_counts = labels_clean.value_counts().sort_index()
        total = len(labels_clean)
        weights = total / (3 * class_counts.values)

    return weights


def apply_label_smoothing(
    labels: pd.Series,
    alpha: float = 0.1,
    n_classes: int = 3,
) -> pd.DataFrame:
    """
    Apply label smoothing to classification labels.

    Args:
        labels: Class labels
        alpha: Smoothing parameter
        n_classes: Number of classes

    Returns:
        DataFrame with smoothed one-hot encoded labels
    """
    # One-hot encode
    onehot = pd.get_dummies(labels, prefix="class")

    # Ensure all classes are present
    for i in range(n_classes):
        col = f"class_{i}"
        if col not in onehot.columns:
            onehot[col] = 0

    # Apply label smoothing: y_smooth = (1-alpha)*y + alpha/K
    smoothed = onehot * (1 - alpha) + alpha / n_classes

    return smoothed
