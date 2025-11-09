"""
Time handling utilities for financial data.
"""

from datetime import datetime

import pandas as pd


def to_utc(dt: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """
    Convert various datetime formats to UTC pandas Timestamp.

    Args:
        dt: Datetime string, datetime object, or pandas Timestamp

    Returns:
        UTC-localized pandas Timestamp
    """
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    elif isinstance(dt, datetime):
        dt = pd.Timestamp(dt)

    if dt.tz is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")

    return dt


def parse_timeframe(timeframe: str) -> pd.Timedelta:
    """
    Parse timeframe string to pandas Timedelta.

    Args:
        timeframe: Timeframe string (e.g., '1m', '15m', '4h', '1d', '1w')

    Returns:
        Pandas Timedelta object
    """
    # Map common abbreviations
    mapping = {
        "m": "min",
        "h": "h",
        "d": "D",
        "w": "W",
    }

    # Extract number and unit
    num = ""
    unit = ""
    for char in timeframe:
        if char.isdigit():
            num += char
        else:
            unit += char

    num = int(num) if num else 1
    unit = mapping.get(unit.lower(), unit)

    return pd.Timedelta(f"{num}{unit}")


def resample_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    agg_dict: dict = None,
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        timeframe: Target timeframe (e.g., '15m', '4h', '1d')
        agg_dict: Custom aggregation dictionary (default OHLCV aggregation)

    Returns:
        Resampled DataFrame
    """
    if agg_dict is None:
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Parse timeframe
    freq = parse_timeframe(timeframe)

    # Resample with left-closed intervals (standard for OHLC)
    resampled = df.resample(freq, closed="left", label="left").agg(agg_dict)

    return resampled.dropna()


def align_timeframes(
    dfs: dict[str, pd.DataFrame],
    fill_method: str = "ffill",
    limit: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Align multiple timeframe DataFrames to common timestamps.

    Args:
        dfs: Dictionary of {timeframe: DataFrame}
        fill_method: Method to fill missing values ('ffill', 'bfill', None)
        limit: Maximum number of consecutive fills

    Returns:
        Dictionary of aligned DataFrames
    """
    # Find common date range
    start = max(df.index[0] for df in dfs.values())
    end = min(df.index[-1] for df in dfs.values())

    aligned = {}
    for tf, df in dfs.items():
        # Filter to common range
        df_filtered = df.loc[start:end]

        # Fill missing values
        if fill_method == "ffill":
            df_filtered = df_filtered.fillna(method="ffill", limit=limit)
        elif fill_method == "bfill":
            df_filtered = df_filtered.fillna(method="bfill", limit=limit)

        aligned[tf] = df_filtered

    return aligned
