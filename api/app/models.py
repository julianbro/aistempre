"""Data models for the API."""

from typing import Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DatasetInfo(BaseModel):
    """Information about a dataset file."""

    symbol: str
    timeframe: str
    filename: str
    row_count: int
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    columns: list[str]


class DatasetListResponse(BaseModel):
    """Response model for listing datasets."""

    datasets: list[DatasetInfo]
    total_count: int


class ValidationIssue(BaseModel):
    """A single validation issue."""

    severity: str  # "error", "warning", "info"
    message: str
    row_number: Optional[int] = None


class ValidationReport(BaseModel):
    """Validation report for a dataset."""

    filename: str
    is_valid: bool
    total_rows: int
    issues: list[ValidationIssue]
    schema_valid: bool
    timezone_valid: bool
    has_gaps: bool
    gap_count: int = 0
    inferred_dtypes: dict[str, str]
    date_range: Optional[dict[str, str]] = None


class ValidateRequest(BaseModel):
    """Request model for dataset validation."""

    filename: str


# Training Run Models


class RunStatus(str, Enum):
    """Status of a training run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunConfig(BaseModel):
    """Configuration for a training run."""

    # Data configuration
    data_source: str
    timeframes: list[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Model configuration
    variant: Literal["base", "medium", "large"] = "base"
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers_tf: Optional[int] = None
    n_layers_fusion: Optional[int] = None
    dropout: Optional[float] = None

    # Features configuration
    features: Optional[list[str]] = None
    enable_technical_indicators: bool = True
    enable_calendar_features: bool = True

    # Training configuration
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.05

    # Loss configuration
    loss_weights: Optional[dict[str, float]] = None
    regression_loss: str = "gaussian_nll"

    # Horizons
    next_horizon: int = 1
    short_horizon: str = "30m"
    long_horizon: str = "1w"

    # Additional Hydra overrides
    overrides: Optional[dict[str, Any]] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    """Metrics snapshot for a training run."""

    epoch: Optional[int] = None
    step: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_da: Optional[float] = None
    val_f1: Optional[float] = None
    val_rmse: Optional[float] = None
    learning_rate: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    eta_minutes: Optional[float] = None


class CheckpointInfo(BaseModel):
    """Information about a checkpoint."""

    filename: str
    epoch: int
    val_loss: Optional[float] = None
    val_score: Optional[float] = None
    size_mb: Optional[float] = None
    created_at: datetime


class TrainingRun(BaseModel):
    """Training run information."""

    id: str
    status: RunStatus
    config: RunConfig
    metrics: Optional[RunMetrics] = None
    checkpoints: list[CheckpointInfo] = Field(default_factory=list)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class CreateRunRequest(BaseModel):
    """Request to create a new training run."""

    config: RunConfig
    name: Optional[str] = None


class CreateRunResponse(BaseModel):
    """Response after creating a training run."""

    id: str
    status: RunStatus
    created_at: datetime


class RunListResponse(BaseModel):
    """Response for listing training runs."""

    runs: list[TrainingRun]
    total_count: int


class StreamEvent(BaseModel):
    """Event streamed via WebSocket during training."""

    event_type: Literal["metrics", "log", "status", "checkpoint"]
    timestamp: datetime
    data: dict[str, Any]


class ArtifactInfo(BaseModel):
    """Information about an artifact."""

    name: str
    size_bytes: int
    created_at: datetime
    artifact_type: str  # "checkpoint", "config", "scaler", "metrics", "predictions"


class ArtifactListResponse(BaseModel):
    """Response for listing artifacts."""

    artifacts: list[ArtifactInfo]
    total_count: int
