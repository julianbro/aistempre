"""
Test loss factory functionality.
"""

import torch

from neurotrader.losses.factory import (
    FocalLoss,
    GaussianNLLLoss,
    LossFactory,
    MultiTaskLoss,
    QuantileLoss,
)


def test_gaussian_nll_loss():
    """Test Gaussian NLL loss."""
    loss_fn = GaussianNLLLoss()

    mu = torch.tensor([1.0, 2.0, 3.0])
    var = torch.tensor([0.1, 0.2, 0.3])
    targets = torch.tensor([1.1, 1.9, 3.2])

    loss = loss_fn(mu, var, targets)

    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_quantile_loss():
    """Test quantile loss."""
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    preds = torch.randn(10, 3)
    targets = torch.randn(10)

    loss = loss_fn(preds, targets)

    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_focal_loss():
    """Test focal loss."""
    loss_fn = FocalLoss(gamma=2.0)

    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))

    loss = loss_fn(logits, targets)

    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_loss_factory_regression():
    """Test loss factory for regression losses."""
    # MSE
    mse_loss = LossFactory.create_regression_loss("mse")
    assert isinstance(mse_loss, torch.nn.MSELoss)

    # Huber
    huber_loss = LossFactory.create_regression_loss("huber", huber_delta=1.0)
    assert isinstance(huber_loss, torch.nn.HuberLoss)

    # Gaussian NLL
    gauss_loss = LossFactory.create_regression_loss("gaussian_nll")
    assert isinstance(gauss_loss, GaussianNLLLoss)

    # Quantile
    quantile_loss = LossFactory.create_regression_loss("quantile")
    assert isinstance(quantile_loss, QuantileLoss)


def test_loss_factory_classification():
    """Test loss factory for classification losses."""
    # Cross-entropy
    ce_loss = LossFactory.create_classification_loss("cross_entropy")
    assert isinstance(ce_loss, torch.nn.CrossEntropyLoss)

    # Focal
    focal_loss = LossFactory.create_classification_loss("focal", focal_gamma=2.0)
    assert isinstance(focal_loss, FocalLoss)


def test_multitask_loss():
    """Test multi-task loss."""
    loss_fn = MultiTaskLoss(
        regression_type="gaussian_nll",
        classification_type="cross_entropy",
        regression_weight=0.5,
        short_trend_weight=0.25,
        long_trend_weight=0.25,
    )

    # Create dummy predictions
    predictions = {
        "regression": {"mu": torch.randn(10), "var": torch.ones(10) * 0.1},
        "short_trend": {"logits": torch.randn(10, 3)},
        "long_trend": {"logits": torch.randn(10, 3)},
    }

    # Create dummy targets
    targets = {
        "next_return": torch.randn(10),
        "short_trend": torch.randint(0, 3, (10,)),
        "long_trend": torch.randint(0, 3, (10,)),
    }

    losses = loss_fn(predictions, targets)

    assert "total" in losses
    assert "regression" in losses
    assert "short_trend" in losses
    assert "long_trend" in losses

    assert losses["total"].item() > 0
    assert not torch.isnan(losses["total"])
