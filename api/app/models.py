"""Data models for the API."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


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
