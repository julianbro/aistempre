"""
Microstructure features (requires tick/L2 data).
"""

import pandas as pd
import numpy as np


def compute_bid_ask_spread(
    df: pd.DataFrame,
    bid_col: str = "bid",
    ask_col: str = "ask",
) -> pd.DataFrame:
    """
    Compute bid-ask spread.
    
    Args:
        df: Input DataFrame with bid/ask columns
        bid_col: Bid price column name
        ask_col: Ask price column name
        
    Returns:
        DataFrame with spread features
    """
    spread = df[ask_col] - df[bid_col]
    mid_price = (df[ask_col] + df[bid_col]) / 2
    spread_bps = (spread / mid_price) * 10000  # basis points
    
    return pd.DataFrame(
        {
            "spread": spread,
            "spread_bps": spread_bps,
            "mid_price": mid_price,
        },
        index=df.index,
    )


def compute_order_imbalance(
    df: pd.DataFrame,
    bid_size_col: str = "bid_size",
    ask_size_col: str = "ask_size",
) -> pd.DataFrame:
    """
    Compute order book imbalance.
    
    Args:
        df: Input DataFrame with bid/ask size columns
        bid_size_col: Bid size column name
        ask_size_col: Ask size column name
        
    Returns:
        DataFrame with order imbalance
    """
    imbalance = (df[bid_size_col] - df[ask_size_col]) / (
        df[bid_size_col] + df[ask_size_col]
    )
    
    return pd.DataFrame({"order_imbalance": imbalance}, index=df.index)


def compute_trade_sign_imbalance(
    df: pd.DataFrame,
    window: int = 100,
    trade_sign_col: str = "trade_sign",
) -> pd.DataFrame:
    """
    Compute trade sign imbalance (buy vs sell pressure).
    
    Args:
        df: Input DataFrame with trade sign column (+1 buy, -1 sell)
        window: Rolling window
        trade_sign_col: Trade sign column name
        
    Returns:
        DataFrame with trade imbalance
    """
    imbalance = df[trade_sign_col].rolling(window=window).sum() / window
    
    return pd.DataFrame({f"trade_imbalance_{window}": imbalance}, index=df.index)


def compute_effective_spread(
    df: pd.DataFrame,
    price_col: str = "price",
    mid_col: str = "mid_price",
) -> pd.DataFrame:
    """
    Compute effective spread (price impact).
    
    Args:
        df: Input DataFrame with trade price and mid price
        price_col: Trade price column
        mid_col: Mid price column
        
    Returns:
        DataFrame with effective spread
    """
    effective_spread = 2 * abs(df[price_col] - df[mid_col])
    effective_spread_bps = (effective_spread / df[mid_col]) * 10000
    
    return pd.DataFrame(
        {
            "effective_spread": effective_spread,
            "effective_spread_bps": effective_spread_bps,
        },
        index=df.index,
    )
