"""
Loss factory for flexible loss function selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class QuantileLoss(nn.Module):
    """Quantile loss for quantile regression."""
    
    def __init__(self, quantiles: List[float] = None):
        """
        Initialize quantile loss.
        
        Args:
            quantiles: List of quantiles
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = torch.tensor(quantiles)
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            preds: Predictions [batch, n_quantiles]
            targets: Targets [batch]
            
        Returns:
            Loss value
        """
        targets = targets.unsqueeze(-1)  # [batch, 1]
        quantiles = self.quantiles.to(preds.device).unsqueeze(0)  # [1, n_quantiles]
        
        errors = targets - preds  # [batch, n_quantiles]
        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for classification with class imbalance."""
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.
        
        Args:
            gamma: Focusing parameter
            alpha: Class weights
            reduction: Reduction method
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Logits [batch, n_classes]
            targets: Targets [batch]
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal_loss = alpha[targets] * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood loss."""
    
    def __init__(self, min_var: float = 1e-6):
        """
        Initialize Gaussian NLL loss.
        
        Args:
            min_var: Minimum variance
        """
        super().__init__()
        self.min_var = min_var
    
    def forward(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL.
        
        Args:
            mu: Mean predictions [batch]
            var: Variance predictions [batch]
            targets: Targets [batch]
            
        Returns:
            Loss value
        """
        var = torch.clamp(var, min=self.min_var)
        # Include normalization constant (log(2*pi)) so the loss is a proper
        # negative log-likelihood and generally positive for reasonable variances.
        two_pi = torch.tensor(2.0 * 3.141592653589793, device=var.device, dtype=var.dtype)
        loss = 0.5 * (torch.log(two_pi * var) + (targets - mu) ** 2 / var)
        return loss.mean()


class StudentTNLLLoss(nn.Module):
    """Student-t Negative Log-Likelihood loss."""
    
    def __init__(self, min_scale: float = 1e-6):
        """
        Initialize Student-t NLL loss.
        
        Args:
            min_scale: Minimum scale
        """
        super().__init__()
        self.min_scale = min_scale
    
    def forward(
        self,
        mu: torch.Tensor,
        scale: torch.Tensor,
        df: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Student-t NLL.
        
        Args:
            mu: Location predictions [batch]
            scale: Scale predictions [batch]
            df: Degrees of freedom [batch]
            targets: Targets [batch]
            
        Returns:
            Loss value
        """
        scale = torch.clamp(scale, min=self.min_scale)
        
        # Student-t NLL
        z = (targets - mu) / scale
        log_prob = (
            torch.lgamma((df + 1) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * torch.log(torch.tensor(3.14159265359) * df)
            - torch.log(scale)
            - ((df + 1) / 2) * torch.log(1 + z ** 2 / df)
        )
        
        return -log_prob.mean()


class LossFactory:
    """
    Factory for creating loss functions based on configuration.
    """
    
    @staticmethod
    def create_regression_loss(
        loss_type: str,
        **kwargs,
    ) -> nn.Module:
        """
        Create regression loss function.
        
        Args:
            loss_type: Type of loss (mse, mae, huber, quantile, gaussian_nll, student_t_nll)
            **kwargs: Additional arguments for loss
            
        Returns:
            Loss function
        """
        if loss_type == "mse":
            return nn.MSELoss()
        
        elif loss_type == "mae":
            return nn.L1Loss()
        
        elif loss_type == "huber":
            delta = kwargs.get("huber_delta", 1.0)
            return nn.HuberLoss(delta=delta)
        
        elif loss_type == "quantile":
            quantiles = kwargs.get("quantiles", [0.1, 0.5, 0.9])
            return QuantileLoss(quantiles=quantiles)
        
        elif loss_type == "gaussian_nll":
            min_var = kwargs.get("gaussian_min_var", 1e-6)
            return GaussianNLLLoss(min_var=min_var)
        
        elif loss_type == "student_t_nll":
            min_scale = kwargs.get("student_t_min_scale", 1e-6)
            return StudentTNLLLoss(min_scale=min_scale)
        
        else:
            raise ValueError(f"Unknown regression loss type: {loss_type}")
    
    @staticmethod
    def create_classification_loss(
        loss_type: str,
        **kwargs,
    ) -> nn.Module:
        """
        Create classification loss function.
        
        Args:
            loss_type: Type of loss (cross_entropy, focal)
            **kwargs: Additional arguments for loss
            
        Returns:
            Loss function
        """
        if loss_type == "cross_entropy":
            label_smoothing = kwargs.get("label_smoothing", 0.0)
            class_weights = kwargs.get("class_weights", None)
            
            if class_weights is not None:
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        
        elif loss_type == "focal":
            gamma = kwargs.get("focal_gamma", 2.0)
            alpha = kwargs.get("focal_alpha", None)
            
            if alpha is not None:
                alpha = torch.tensor(alpha, dtype=torch.float32)
            
            return FocalLoss(gamma=gamma, alpha=alpha)
        
        else:
            raise ValueError(f"Unknown classification loss type: {loss_type}")
    
    @staticmethod
    def create_multitask_loss(
        regression_type: str,
        classification_type: str,
        regression_weight: float = 0.5,
        short_trend_weight: float = 0.25,
        long_trend_weight: float = 0.25,
        **kwargs,
    ):
        """
        Create multi-task loss function.
        
        Args:
            regression_type: Type of regression loss
            classification_type: Type of classification loss
            regression_weight: Weight for regression loss
            short_trend_weight: Weight for short-term trend loss
            long_trend_weight: Weight for long-term trend loss
            **kwargs: Additional arguments for losses
            
        Returns:
            MultiTaskLoss instance
        """
        return MultiTaskLoss(
            regression_type=regression_type,
            classification_type=classification_type,
            regression_weight=regression_weight,
            short_trend_weight=short_trend_weight,
            long_trend_weight=long_trend_weight,
            **kwargs,
        )


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining regression and classification."""
    
    def __init__(
        self,
        regression_type: str,
        classification_type: str,
        regression_weight: float = 0.5,
        short_trend_weight: float = 0.25,
        long_trend_weight: float = 0.25,
        **kwargs,
    ):
        """
        Initialize multi-task loss.
        
        Args:
            regression_type: Type of regression loss
            classification_type: Type of classification loss
            regression_weight: Weight for regression loss
            short_trend_weight: Weight for short-term trend loss
            long_trend_weight: Weight for long-term trend loss
            **kwargs: Additional arguments for losses
        """
        super().__init__()
        
        self.regression_weight = regression_weight
        self.short_trend_weight = short_trend_weight
        self.long_trend_weight = long_trend_weight
        
        self.regression_loss = LossFactory.create_regression_loss(
            regression_type, **kwargs
        )
        self.short_trend_loss = LossFactory.create_classification_loss(
            classification_type, **kwargs
        )
        self.long_trend_loss = LossFactory.create_classification_loss(
            classification_type, **kwargs
        )
        
        self.regression_type = regression_type
    
    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary with predictions for each task
            targets: Dictionary with targets for each task
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Regression loss
        if self.regression_type in ["gaussian_nll"]:
            reg_loss = self.regression_loss(
                predictions["regression"]["mu"],
                predictions["regression"]["var"],
                targets["next_return"],
            )
        elif self.regression_type in ["student_t_nll"]:
            reg_loss = self.regression_loss(
                predictions["regression"]["mu"],
                predictions["regression"]["scale"],
                predictions["regression"]["df"],
                targets["next_return"],
            )
        elif self.regression_type in ["quantile"]:
            reg_loss = self.regression_loss(
                predictions["regression"]["quantiles"],
                targets["next_return"],
            )
        else:  # deterministic or simple losses
            pred = predictions["regression"].get("pred", predictions["regression"].get("mu"))
            reg_loss = self.regression_loss(pred, targets["next_return"])
        
        losses["regression"] = reg_loss
        
        # Short-term trend loss
        short_loss = self.short_trend_loss(
            predictions["short_trend"]["logits"],
            targets["short_trend"].long(),
        )
        losses["short_trend"] = short_loss
        
        # Long-term trend loss
        long_loss = self.long_trend_loss(
            predictions["long_trend"]["logits"],
            targets["long_trend"].long(),
        )
        losses["long_trend"] = long_loss
        
        # Total weighted loss
        total_loss = (
            self.regression_weight * reg_loss
            + self.short_trend_weight * short_loss
            + self.long_trend_weight * long_loss
        )
        losses["total"] = total_loss
        
        return losses
