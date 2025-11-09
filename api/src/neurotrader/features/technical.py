"""
Technical indicator features.
"""

import pandas as pd


def compute_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    """
    Compute Relative Strength Index.

    Args:
        df: Input DataFrame with OHLCV data
        period: RSI period
        column: Column to compute RSI on

    Returns:
        DataFrame with RSI column
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return pd.DataFrame({f"rsi_{period}": rsi}, index=df.index)


def compute_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """
    Compute Exponential Moving Average.

    Args:
        df: Input DataFrame
        period: EMA period
        column: Column to compute EMA on

    Returns:
        DataFrame with EMA column
    """
    ema = df[column].ewm(span=period, adjust=False).mean()
    return pd.DataFrame({f"ema_{period}": ema}, index=df.index)


def compute_sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """
    Compute Simple Moving Average.

    Args:
        df: Input DataFrame
        period: SMA period
        column: Column to compute SMA on

    Returns:
        DataFrame with SMA column
    """
    sma = df[column].rolling(window=period).mean()
    return pd.DataFrame({f"sma_{period}": sma}, index=df.index)


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute MACD indicator.

    Args:
        df: Input DataFrame
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        column: Column to compute MACD on

    Returns:
        DataFrame with MACD, signal, and histogram columns
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        },
        index=df.index,
    )


def compute_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Args:
        df: Input DataFrame
        period: Moving average period
        std_dev: Number of standard deviations
        column: Column to compute bands on

    Returns:
        DataFrame with upper, middle, and lower bands
    """
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()

    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bandwidth = (upper - lower) / sma

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
        },
        index=df.index,
    )


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range.

    Args:
        df: Input DataFrame with OHLC data
        period: ATR period

    Returns:
        DataFrame with ATR column
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return pd.DataFrame({f"atr_{period}": atr}, index=df.index)


def compute_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator.

    Args:
        df: Input DataFrame with OHLC data
        k_period: %K period
        d_period: %D period (smoothing)

    Returns:
        DataFrame with %K and %D columns
    """
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()

    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()

    return pd.DataFrame(
        {
            "stoch_k": k,
            "stoch_d": d,
        },
        index=df.index,
    )


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average Directional Index.

    Args:
        df: Input DataFrame with OHLC data
        period: ADX period

    Returns:
        DataFrame with ADX column
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up = high - high.shift()
    down = low.shift() - low

    pos_dm = up.where((up > down) & (up > 0), 0)
    neg_dm = down.where((down > up) & (down > 0), 0)

    # Smoothed indicators
    atr = tr.rolling(window=period).mean()
    pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

    # ADX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = dx.rolling(window=period).mean()

    return pd.DataFrame({f"adx_{period}": adx}, index=df.index)
