"""
Probability calibration methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.

    Scales logits by a learned temperature parameter to calibrate probabilities.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaling.

        Args:
            temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw logits [batch, n_classes]

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ):
        """
        Fit temperature parameter on validation set.

        Args:
            logits: Validation logits [n_samples, n_classes]
            labels: Validation labels [n_samples]
            max_iter: Maximum optimization iterations
            lr: Learning rate
        """
        # Create NLL criterion
        nll_criterion = nn.CrossEntropyLoss()

        # Optimize temperature
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

    def get_calibrated_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get calibrated probabilities.

        Args:
            logits: Raw logits

        Returns:
            Calibrated probabilities
        """
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=-1)


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration.

    Uses sklearn's isotonic regression to calibrate probabilities.
    """

    def __init__(self):
        """Initialize isotonic calibration."""

        self.calibrators = []  # One per class

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_classes: int,
    ):
        """
        Fit isotonic regression calibrators.

        Args:
            probs: Predicted probabilities [n_samples, n_classes]
            labels: True labels [n_samples]
            n_classes: Number of classes
        """
        from sklearn.isotonic import IsotonicRegression

        self.calibrators = []

        # Fit one calibrator per class
        for c in range(n_classes):
            calibrator = IsotonicRegression(out_of_bounds="clip")

            # Binary labels for this class
            y_binary = (labels == c).astype(int)

            # Fit on probabilities for this class
            calibrator.fit(probs[:, c], y_binary)

            self.calibrators.append(calibrator)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            probs: Predicted probabilities [n_samples, n_classes]

        Returns:
            Calibrated probabilities
        """
        calibrated = np.zeros_like(probs)

        for c, calibrator in enumerate(self.calibrators):
            calibrated[:, c] = calibrator.transform(probs[:, c])

        # Normalize to sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities [n_samples, n_classes]
        labels: True labels [n_samples]
        n_bins: Number of bins for calibration

    Returns:
        ECE value
    """
    # Get predicted class and confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    # Bin confidences
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins[:-1]) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = accuracies[mask].mean()
            bin_weight = mask.sum() / len(confidences)

            ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return ece


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Brier score.

    Args:
        probs: Predicted probabilities [n_samples, n_classes]
        labels: True labels [n_samples]

    Returns:
        Brier score
    """
    # One-hot encode labels
    n_classes = probs.shape[1]
    labels_onehot = np.eye(n_classes)[labels]

    # Brier score
    brier = np.mean((probs - labels_onehot) ** 2)

    return brier


class RegressionCalibration:
    """
    Calibration for regression prediction intervals.
    """

    def __init__(self):
        """Initialize regression calibration."""
        self.scale_factor = 1.0

    def fit(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        coverage: float = 0.9,
    ):
        """
        Fit calibration scale factor.

        Args:
            predictions: Point predictions [n_samples]
            uncertainties: Predicted standard deviations [n_samples]
            targets: True targets [n_samples]
            coverage: Target coverage level
        """
        # Compute z-scores
        z_scores = np.abs((targets - predictions) / uncertainties)

        # Find scale factor to achieve desired coverage
        target_z = np.percentile(z_scores, coverage * 100)
        expected_z = 1.645  # For 90% coverage in standard normal

        self.scale_factor = target_z / expected_z

    def calibrate(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """
        Apply calibration to uncertainties.

        Args:
            predictions: Point predictions
            uncertainties: Predicted standard deviations

        Returns:
            Calibrated uncertainties
        """
        return uncertainties * self.scale_factor
