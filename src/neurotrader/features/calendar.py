"""
Calendar-based features.
"""

import pandas as pd
import numpy as np


def compute_hour_of_day(df: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
    """
    Compute hour of day features.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        cyclical: Use cyclical encoding (sin/cos)
        
    Returns:
        DataFrame with hour features
    """
    hour = df.index.hour
    
    if cyclical:
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        return pd.DataFrame(
            {
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
            },
            index=df.index,
        )
    else:
        return pd.DataFrame({"hour": hour}, index=df.index)


def compute_day_of_week(df: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
    """
    Compute day of week features.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        cyclical: Use cyclical encoding (sin/cos)
        
    Returns:
        DataFrame with day of week features
    """
    day = df.index.dayofweek
    
    if cyclical:
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        return pd.DataFrame(
            {
                "dayofweek_sin": day_sin,
                "dayofweek_cos": day_cos,
            },
            index=df.index,
        )
    else:
        return pd.DataFrame({"dayofweek": day}, index=df.index)


def compute_month_of_year(df: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
    """
    Compute month of year features.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        cyclical: Use cyclical encoding (sin/cos)
        
    Returns:
        DataFrame with month features
    """
    month = df.index.month
    
    if cyclical:
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        return pd.DataFrame(
            {
                "month_sin": month_sin,
                "month_cos": month_cos,
            },
            index=df.index,
        )
    else:
        return pd.DataFrame({"month": month}, index=df.index)


def compute_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekend indicator.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with weekend indicator
    """
    is_weekend = (df.index.dayofweek >= 5).astype(int)
    return pd.DataFrame({"is_weekend": is_weekend}, index=df.index)


def compute_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading session flags (Asian, European, US).
    
    Args:
        df: Input DataFrame with DatetimeIndex (UTC)
        
    Returns:
        DataFrame with session flags
    """
    hour = df.index.hour
    
    # Session hours in UTC
    # Asian: 00:00 - 09:00 UTC
    # European: 07:00 - 16:00 UTC
    # US: 13:00 - 21:00 UTC
    
    is_asian = ((hour >= 0) & (hour < 9)).astype(int)
    is_european = ((hour >= 7) & (hour < 16)).astype(int)
    is_us = ((hour >= 13) & (hour < 21)).astype(int)
    
    return pd.DataFrame(
        {
            "session_asian": is_asian,
            "session_european": is_european,
            "session_us": is_us,
        },
        index=df.index,
    )


def compute_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute day of month feature.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with day of month
    """
    day = df.index.day
    return pd.DataFrame({"day_of_month": day}, index=df.index)


def compute_is_month_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month start indicator.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with month start indicator
    """
    is_start = df.index.is_month_start.astype(int)
    return pd.DataFrame({"is_month_start": is_start}, index=df.index)


def compute_is_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month end indicator.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with month end indicator
    """
    is_end = df.index.is_month_end.astype(int)
    return pd.DataFrame({"is_month_end": is_end}, index=df.index)
