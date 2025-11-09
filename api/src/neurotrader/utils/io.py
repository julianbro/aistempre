"""
I/O utilities for loading and saving data.
"""

import pickle
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load CSV file with common defaults for financial data.

    Args:
        path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame
    """
    defaults = {
        "parse_dates": True,
        "index_col": 0,
    }
    defaults.update(kwargs)

    return pd.read_csv(path, **defaults)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs):
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame to save
        path: Output path
        **kwargs: Additional arguments passed to df.to_csv
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)


def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load Parquet file.

    Args:
        path: Path to Parquet file
        **kwargs: Additional arguments passed to pd.read_parquet

    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs):
    """
    Save DataFrame to Parquet.

    Args:
        df: DataFrame to save
        path: Output path
        **kwargs: Additional arguments passed to df.to_parquet
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)


def load_pickle(path: str | Path):
    """
    Load pickled object.

    Args:
        path: Path to pickle file

    Returns:
        Unpickled object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str | Path):
    """
    Save object to pickle.

    Args:
        obj: Object to pickle
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
