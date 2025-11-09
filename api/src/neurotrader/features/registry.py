"""
Feature registry for plug-in feature engineering.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd


class FeatureRegistry:
    """
    Registry for feature engineering functions.

    Allows dynamic feature creation based on configuration.
    """

    def __init__(self):
        """Initialize feature registry."""
        self._features: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable,
        params: dict[str, Any] | None = None,
        lookback: int = 0,
        output_cols: list[str] | None = None,
    ):
        """
        Register a feature function.

        Args:
            name: Feature name
            func: Feature computation function
            params: Parameters for the function
            lookback: Required lookback period
            output_cols: Names of output columns
        """
        self._features[name] = {
            "func": func,
            "params": params or {},
            "lookback": lookback,
            "output_cols": output_cols or [name],
        }

    def compute(
        self,
        name: str,
        df: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute a feature.

        Args:
            name: Feature name
            df: Input DataFrame
            **kwargs: Additional arguments override registered params

        Returns:
            DataFrame with computed feature columns
        """
        if name not in self._features:
            raise ValueError(f"Feature '{name}' not registered")

        feature = self._features[name]
        func = feature["func"]
        params = {**feature["params"], **kwargs}

        return func(df, **params)

    def compute_all(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute multiple features.

        Args:
            df: Input DataFrame
            feature_names: List of feature names (None for all)

        Returns:
            DataFrame with all computed features
        """
        if feature_names is None:
            feature_names = list(self._features.keys())

        result = df.copy()

        for name in feature_names:
            feature_df = self.compute(name, df)
            result = pd.concat([result, feature_df], axis=1)

        return result

    def get_total_lookback(
        self,
        feature_names: list[str] | None = None,
    ) -> int:
        """
        Get maximum lookback required for features.

        Args:
            feature_names: List of feature names (None for all)

        Returns:
            Maximum lookback period
        """
        if feature_names is None:
            feature_names = list(self._features.keys())

        lookbacks = [
            self._features[name]["lookback"] for name in feature_names if name in self._features
        ]

        return max(lookbacks) if lookbacks else 0

    def list_features(self) -> list[str]:
        """List all registered features."""
        return list(self._features.keys())


# Global registry instance
_global_registry = FeatureRegistry()


def get_registry() -> FeatureRegistry:
    """Get the global feature registry."""
    return _global_registry
