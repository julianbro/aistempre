"""
Volatility-based features.
"""

import numpy as np
import pandas as pd


def compute_realized_volatility(
    df: pd.DataFrame,
    window: int = 20,
    column: str = "close",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute realized volatility.

    Args:
        df: Input DataFrame
        window: Rolling window
        column: Price column
        annualize: Whether to annualize volatility

    Returns:
        DataFrame with realized volatility column
    """
    returns = np.log(df[column] / df[column].shift(1))
    rv = returns.rolling(window=window).std()

    if annualize:
        # Assuming daily data, annualize with sqrt(252)
        rv = rv * np.sqrt(252)

    return pd.DataFrame({f"realized_vol_{window}": rv}, index=df.index)


def compute_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute Parkinson volatility (uses high-low range).

    Args:
        df: Input DataFrame with OHLC data
        window: Rolling window
        annualize: Whether to annualize volatility

    Returns:
        DataFrame with Parkinson volatility column
    """
    hl_ratio = np.log(df["high"] / df["low"])
    parkinson = np.sqrt((hl_ratio**2).rolling(window=window).mean() / (4 * np.log(2)))

    if annualize:
        parkinson = parkinson * np.sqrt(252)

    return pd.DataFrame({f"parkinson_vol_{window}": parkinson}, index=df.index)


def compute_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute Garman-Klass volatility.

    Args:
        df: Input DataFrame with OHLC data
        window: Rolling window
        annualize: Whether to annualize volatility

    Returns:
        DataFrame with Garman-Klass volatility column
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    gk = np.sqrt(
        (0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)).rolling(window=window).mean()
    )

    if annualize:
        gk = gk * np.sqrt(252)

    return pd.DataFrame({f"gk_vol_{window}": gk}, index=df.index)


def compute_volatility_ratio(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute ratio of short-term to long-term volatility.

    Args:
        df: Input DataFrame
        short_window: Short window for volatility
        long_window: Long window for volatility
        column: Price column

    Returns:
        DataFrame with volatility ratio column
    """
    returns = np.log(df[column] / df[column].shift(1))

    short_vol = returns.rolling(window=short_window).std()
    long_vol = returns.rolling(window=long_window).std()

    vol_ratio = short_vol / long_vol

    return pd.DataFrame({"volatility_ratio": vol_ratio}, index=df.index)


def compute_volatility_zscore(
    df: pd.DataFrame,
    window: int = 20,
    zscore_window: int = 100,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute z-scored volatility.

    Args:
        df: Input DataFrame
        window: Window for volatility calculation
        zscore_window: Window for z-score calculation
        column: Price column

    Returns:
        DataFrame with volatility z-score column
    """
    returns = np.log(df[column] / df[column].shift(1))
    vol = returns.rolling(window=window).std()

    vol_mean = vol.rolling(window=zscore_window).mean()
    vol_std = vol.rolling(window=zscore_window).std()

    vol_zscore = (vol - vol_mean) / vol_std

    return pd.DataFrame({f"vol_zscore_{window}": vol_zscore}, index=df.index)


def compute_atr_percent(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Compute ATR as percentage of price.

    Args:
        df: Input DataFrame with OHLC data
        period: ATR period

    Returns:
        DataFrame with ATR percentage column
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    atr_pct = (atr / close) * 100

    return pd.DataFrame({f"atr_pct_{period}": atr_pct}, index=df.index)
