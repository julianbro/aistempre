"""
Generate example/toy financial data for testing.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_price: float = 10000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with random walk.

    Args:
        start_date: Start date
        end_date: End date
        initial_price: Initial price
        volatility: Price volatility
        trend: Drift term
        freq: Frequency (pandas freq string)

    Returns:
        DataFrame with OHLCV data
    """
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq, tz="UTC")
    n = len(timestamps)

    # Generate random walk for close prices
    returns = np.random.normal(trend, volatility, n)
    log_prices = np.log(initial_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)

    # Generate OHLC from close
    high_offset = np.abs(np.random.normal(0, volatility / 2, n))
    low_offset = np.abs(np.random.normal(0, volatility / 2, n))
    open_offset = np.random.normal(0, volatility / 4, n)

    open_prices = close_prices * (1 + open_offset)
    high_prices = np.maximum(close_prices, open_prices) * (1 + high_offset)
    low_prices = np.minimum(close_prices, open_prices) * (1 - low_offset)

    # Generate volume (log-normal distribution)
    volumes = np.exp(np.random.normal(10, 1, n))

    # Create DataFrame
    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
        index=timestamps,
    )

    return df


def main():
    """Generate example data for different timeframes."""
    output_dir = Path("data/example")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating example data...")

    # Generate base 1-minute data
    print("Generating 1m data...")
    df_1m = generate_synthetic_ohlcv(
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_price=10000.0,
        volatility=0.002,
        freq="1min",
    )

    # Resample to other timeframes
    def resample_ohlcv(df, freq):
        return (
            df.resample(freq)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    print("Resampling to other timeframes...")
    df_15m = resample_ohlcv(df_1m, "15min")
    df_4h = resample_ohlcv(df_1m, "4H")
    df_1d = resample_ohlcv(df_1m, "1D")
    df_1w = resample_ohlcv(df_1m, "1W")

    # Save to CSV
    print("Saving to CSV...")
    df_1m.to_csv(output_dir / "BTCUSDT_1m.csv")
    df_15m.to_csv(output_dir / "BTCUSDT_15m.csv")
    df_4h.to_csv(output_dir / "BTCUSDT_4h.csv")
    df_1d.to_csv(output_dir / "BTCUSDT_1d.csv")
    df_1w.to_csv(output_dir / "BTCUSDT_1w.csv")

    print(f"\nExample data saved to {output_dir}/")
    print(f"1m data: {len(df_1m)} rows")
    print(f"15m data: {len(df_15m)} rows")
    print(f"4h data: {len(df_4h)} rows")
    print(f"1d data: {len(df_1d)} rows")
    print(f"1w data: {len(df_1w)} rows")

    print("\nExample usage:")
    print("  neurotrader-train --config-name train.yaml")
    print(f"  # Edit configs/data.yaml to point to {output_dir}/")


if __name__ == "__main__":
    main()
