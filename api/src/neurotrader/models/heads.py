"""
Prediction heads for regression and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLHead(nn.Module):
    """Gaussian Negative Log-Likelihood regression head."""

    def __init__(self, d_model: int, min_var: float = 1e-6):
        """
        Initialize Gaussian NLL head.

        Args:
            d_model: Input dimension
            min_var: Minimum variance (for stability)
        """
        super().__init__()
        self.min_var = min_var

        self.mu_proj = nn.Linear(d_model, 1)
        self.logvar_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Dictionary with 'mu' and 'var'
        """
        mu = self.mu_proj(x).squeeze(-1)
        logvar = self.logvar_proj(x).squeeze(-1)
        var = torch.exp(logvar) + self.min_var

        return {"mu": mu, "var": var}


class StudentTNLLHead(nn.Module):
    """Student-t Negative Log-Likelihood regression head."""

    def __init__(self, d_model: int, min_scale: float = 1e-6):
        """
        Initialize Student-t NLL head.

        Args:
            d_model: Input dimension
            min_scale: Minimum scale (for stability)
        """
        super().__init__()
        self.min_scale = min_scale

        self.mu_proj = nn.Linear(d_model, 1)
        self.logscale_proj = nn.Linear(d_model, 1)
        self.df_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Dictionary with 'mu', 'scale', and 'df'
        """
        mu = self.mu_proj(x).squeeze(-1)
        logscale = self.logscale_proj(x).squeeze(-1)
        scale = torch.exp(logscale) + self.min_scale
        df = F.softplus(self.df_proj(x)).squeeze(-1) + 2.0  # df > 2

        return {"mu": mu, "scale": scale, "df": df}


class QuantileHead(nn.Module):
    """Quantile regression head."""

    def __init__(self, d_model: int, quantiles: list[float] = None):
        """
        Initialize quantile head.

        Args:
            d_model: Input dimension
            quantiles: List of quantiles to predict
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        self.proj = nn.Linear(d_model, self.n_quantiles)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Dictionary with quantile predictions
        """
        preds = self.proj(x)  # [batch, n_quantiles]

        # Return as dictionary
        result = {f"q{int(q * 100)}": preds[:, i] for i, q in enumerate(self.quantiles)}
        result["quantiles"] = preds

        return result


class DeterministicHead(nn.Module):
    """Simple deterministic regression head."""

    def __init__(self, d_model: int):
        """
        Initialize deterministic head.

        Args:
            d_model: Input dimension
        """
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Dictionary with 'pred'
        """
        pred = self.proj(x).squeeze(-1)
        return {"pred": pred}


class ClassificationHead(nn.Module):
    """Classification head with softmax."""

    def __init__(self, d_model: int, n_classes: int = 3, dropout: float = 0.1):
        """
        Initialize classification head.

        Args:
            d_model: Input dimension
            n_classes: Number of classes
            dropout: Dropout probability
        """
        super().__init__()
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Dictionary with 'logits' and 'probs'
        """
        x = self.dropout(x)
        logits = self.proj(x)
        probs = F.softmax(logits, dim=-1)

        return {"logits": logits, "probs": probs}


class MultiTaskHead(nn.Module):
    """Multi-task head combining regression and classification."""

    def __init__(
        self,
        d_model: int,
        regression_type: str = "gaussian_nll",
        n_classes_short: int = 3,
        n_classes_long: int = 3,
        quantiles: list[float] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-task head.

        Args:
            d_model: Input dimension
            regression_type: Type of regression head
            n_classes_short: Number of classes for short-term trend
            n_classes_long: Number of classes for long-term trend
            quantiles: Quantiles for quantile regression
            dropout: Dropout probability
        """
        super().__init__()
        self.regression_type = regression_type

        # Regression head
        if regression_type == "gaussian_nll":
            self.regression_head = GaussianNLLHead(d_model)
        elif regression_type == "student_t":
            self.regression_head = StudentTNLLHead(d_model)
        elif regression_type == "quantile":
            self.regression_head = QuantileHead(d_model, quantiles)
        elif regression_type == "deterministic":
            self.regression_head = DeterministicHead(d_model)
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")

        # Classification heads
        self.short_trend_head = ClassificationHead(d_model, n_classes_short, dropout)
        self.long_trend_head = ClassificationHead(d_model, n_classes_long, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Dictionary with outputs for each task
        """
        outputs = {
            "regression": self.regression_head(x),
            "short_trend": self.short_trend_head(x),
            "long_trend": self.long_trend_head(x),
        }

        return outputs
