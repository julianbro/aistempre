"""
neurotrader: Multi-input, multi-horizon, probabilistic Transformer for financial time-series.

This package provides tools for training and serving deep learning models
for financial prediction with calibrated probabilities and prediction intervals.
"""

__version__ = "0.1.0"
__author__ = "neurotrader contributors"
__license__ = "MIT"

from neurotrader.utils.seed import set_seed

__all__ = ["set_seed", "__version__"]
