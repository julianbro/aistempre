"""Label generation module initialization."""

from neurotrader.labels.targets import (
    compute_next_return_target,
    compute_trend_targets,
)

__all__ = ["compute_next_return_target", "compute_trend_targets"]
