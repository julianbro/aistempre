"""
Conformal prediction for distribution-free prediction intervals.
"""

import numpy as np
from typing import Tuple, Optional


class ConformalPredictor:
    """
    Conformal prediction for regression.
    
    Provides distribution-free prediction intervals with guaranteed coverage.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage level (1-alpha is target coverage)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None
    
    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Calibrate on validation set.
        
        Args:
            predictions: Point predictions [n_samples]
            targets: True targets [n_samples]
        """
        # Compute absolute residuals as conformity scores
        self.calibration_scores = np.abs(targets - predictions)
        
        # Compute quantile for desired coverage
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, q_level)
    
    def predict(
        self,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.
        
        Args:
            predictions: Point predictions [n_samples]
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if self.quantile is None:
            raise ValueError("Must calibrate before prediction")
        
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        return lower, upper
    
    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Evaluate empirical coverage.
        
        Args:
            predictions: Point predictions
            targets: True targets
            
        Returns:
            Empirical coverage rate
        """
        lower, upper = self.predict(predictions)
        covered = (targets >= lower) & (targets <= upper)
        return covered.mean()


class AdaptiveConformalPredictor:
    """
    Adaptive conformal prediction with time-varying coverage.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 100,
    ):
        """
        Initialize adaptive conformal predictor.
        
        Args:
            alpha: Miscoverage level
            window_size: Window size for adaptation
        """
        self.alpha = alpha
        self.window_size = window_size
        self.calibration_scores = []
    
    def update(
        self,
        prediction: float,
        target: float,
    ):
        """
        Update with new observation.
        
        Args:
            prediction: Point prediction
            target: True target
        """
        score = abs(target - prediction)
        self.calibration_scores.append(score)
        
        # Keep only recent scores
        if len(self.calibration_scores) > self.window_size:
            self.calibration_scores.pop(0)
    
    def get_interval_width(self) -> float:
        """
        Get current prediction interval width.
        
        Returns:
            Interval half-width
        """
        if len(self.calibration_scores) < 10:
            # Not enough data, use conservative estimate
            return float("inf")
        
        n = len(self.calibration_scores)
        q_level = (n + 1) * (1 - self.alpha) / n
        quantile = np.quantile(self.calibration_scores, q_level)
        
        return quantile
    
    def predict(
        self,
        prediction: float,
    ) -> Tuple[float, float]:
        """
        Generate prediction interval.
        
        Args:
            prediction: Point prediction
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        width = self.get_interval_width()
        return prediction - width, prediction + width


class QuantileConformalPredictor:
    """
    Conformal prediction for quantile regression.
    """
    
    def __init__(
        self,
        quantiles: list[float] = None,
        alpha: float = 0.1,
    ):
        """
        Initialize quantile conformal predictor.
        
        Args:
            quantiles: Quantiles used in regression
            alpha: Miscoverage level
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.quantiles = quantiles
        self.alpha = alpha
        self.lower_correction = 0.0
        self.upper_correction = 0.0
    
    def calibrate(
        self,
        quantile_predictions: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Calibrate quantile predictions.
        
        Args:
            quantile_predictions: Quantile predictions [n_samples, n_quantiles]
            targets: True targets [n_samples]
        """
        # Assume first and last quantiles are bounds
        lower_preds = quantile_predictions[:, 0]
        upper_preds = quantile_predictions[:, -1]
        
        # Compute conformity scores
        lower_scores = targets - lower_preds  # positive if too low
        upper_scores = upper_preds - targets  # positive if too high
        
        # Find corrections needed
        n = len(targets)
        q_level = (n + 1) * (1 - self.alpha / 2) / n
        
        self.lower_correction = np.quantile(lower_scores, q_level)
        self.upper_correction = np.quantile(upper_scores, q_level)
    
    def predict(
        self,
        quantile_predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate conformal prediction intervals.
        
        Args:
            quantile_predictions: Quantile predictions [n_samples, n_quantiles]
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower_preds = quantile_predictions[:, 0]
        upper_preds = quantile_predictions[:, -1]
        
        # Apply corrections
        lower = lower_preds - self.lower_correction
        upper = upper_preds + self.upper_correction
        
        return lower, upper


def compute_interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """
    Compute metrics for prediction intervals.
    
    Args:
        lower: Lower bounds
        upper: Upper bounds
        targets: True targets
        
    Returns:
        Dictionary with metrics
    """
    # Coverage
    covered = (targets >= lower) & (targets <= upper)
    coverage = covered.mean()
    
    # Average interval width
    width = (upper - lower).mean()
    
    # Interval score (lower is better)
    interval_score = (upper - lower) + (2 / 0.1) * (
        (lower - targets) * (targets < lower) +
        (targets - upper) * (targets > upper)
    )
    avg_interval_score = interval_score.mean()
    
    return {
        "coverage": coverage,
        "average_width": width,
        "interval_score": avg_interval_score,
    }
