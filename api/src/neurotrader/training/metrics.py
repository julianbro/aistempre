"""
Custom metrics for financial time-series prediction.
"""

import numpy as np
import torch
from torchmetrics import Metric


class DirectionalAccuracy(Metric):
    """Directional accuracy metric (fraction of correct direction predictions)."""

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metric state."""
        # Check if predicted and actual have same sign
        correct = ((preds * targets) > 0).float().sum()

        self.correct += correct
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        """Compute final metric."""
        return self.correct / self.total


class sMAPE(Metric):
    """Symmetric Mean Absolute Percentage Error."""

    def __init__(self):
        super().__init__()
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metric state."""
        error = torch.abs(preds - targets) / (torch.abs(preds) + torch.abs(targets) + 1e-8)

        self.sum_error += error.sum()
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        """Compute final metric."""
        return 200.0 * self.sum_error / self.total


class MCC(Metric):
    """Matthews Correlation Coefficient for multi-class classification."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.add_state(
            "confusion_matrix",
            default=torch.zeros(n_classes, n_classes),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix."""
        for t, p in zip(targets, preds):
            self.confusion_matrix[t.long(), p.long()] += 1

    def compute(self) -> torch.Tensor:
        """Compute MCC."""
        C = self.confusion_matrix
        t_sum = C.sum(dim=1)
        p_sum = C.sum(dim=0)
        n_correct = torch.diag(C).sum()
        n_samples = C.sum()

        cov_ytyp = n_correct * n_samples - (t_sum * p_sum).sum()
        cov_ypyp = n_samples**2 - (p_sum**2).sum()
        cov_ytyt = n_samples**2 - (t_sum**2).sum()

        mcc = cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp + 1e-8)

        return mcc


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year

    if len(excess_returns) == 0 or excess_returns.std() == 0:
        return 0.0

    sharpe = excess_returns.mean() / excess_returns.std()
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)

    return sharpe_annualized


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sortino ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year

    # Downside deviation
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float("inf")

    downside_std = np.sqrt((downside_returns**2).mean())

    if downside_std == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_std
    sortino_annualized = sortino * np.sqrt(periods_per_year)

    return sortino_annualized


def compute_max_drawdown(
    cumulative_returns: np.ndarray,
) -> float:
    """
    Compute maximum drawdown.

    Args:
        cumulative_returns: Array of cumulative returns

    Returns:
        Maximum drawdown (positive value)
    """
    cumulative_wealth = 1.0 + cumulative_returns
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max

    return abs(drawdown.min())


def compute_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Calmar ratio (return / max drawdown).

    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    cumulative_returns = np.cumsum(returns)
    max_dd = compute_max_drawdown(cumulative_returns)

    if max_dd == 0:
        return float("inf")

    annual_return = returns.mean() * periods_per_year

    return annual_return / max_dd
