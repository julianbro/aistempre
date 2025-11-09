"""Dataset API routes."""

from fastapi import APIRouter, HTTPException
from app.models import DatasetListResponse, ValidateRequest, ValidationReport
from app.datasets import discover_datasets, validate_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetListResponse)
def list_datasets():
    """
    List all available datasets with metadata.

    Returns:
        - symbols: List of available trading symbols
        - timeframes: List of available timeframes
        - date_ranges: Date ranges for each dataset
        - row_counts: Number of rows in each dataset
    """
    datasets = discover_datasets()
    return DatasetListResponse(datasets=datasets, total_count=len(datasets))


@router.post("/validate", response_model=ValidationReport)
def validate_dataset_route(request: ValidateRequest):
    """
    Validate a dataset file.

    Checks:
    - Schema: OHLCV columns present
    - Timezone: All timestamps are UTC
    - No NaN gaps across session bounds
    - OHLC consistency (high >= open/close, low <= open/close)

    Returns a validation report with:
    - inferred_dtypes: Data types for each column
    - timeframe_alignment_hints: Detected timeframe
    - missing_data_report: Gaps and missing values
    """
    try:
        report = validate_dataset(request.filename)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
