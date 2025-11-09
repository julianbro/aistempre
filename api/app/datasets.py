"""Dataset discovery and validation utilities."""

import csv
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from app.models import DatasetInfo, ValidationIssue, ValidationReport


def get_data_root() -> Path:
    """Get the data root directory from environment or use default."""
    data_root = os.getenv("DATA_ROOT", "./data")
    return Path(data_root).resolve()


def is_safe_path(base_dir: Path, user_path: str) -> bool:
    """
    Check if the user-provided path is safe (no path traversal).
    
    Args:
        base_dir: The base directory that should contain the file
        user_path: The user-provided path
        
    Returns:
        True if the path is safe, False otherwise
    """
    try:
        # Construct the full path
        full_path = (base_dir / user_path).resolve()
        # Check if it's within base_dir
        full_path.relative_to(base_dir)
        return True
    except (ValueError, OSError):
        return False


def discover_datasets() -> list[DatasetInfo]:
    """Discover all CSV datasets in the data directory."""
    data_root = get_data_root()
    datasets = []

    if not data_root.exists():
        return datasets

    # Walk through all CSV files
    for csv_file in data_root.rglob("*.csv"):
        try:
            # Parse filename (e.g., BTCUSDT_1m.csv)
            filename = csv_file.name
            name_parts = filename.replace(".csv", "").split("_")

            if len(name_parts) >= 2:
                symbol = name_parts[0]
                timeframe = name_parts[1]
            else:
                symbol = filename.replace(".csv", "")
                timeframe = "unknown"

            # Read first and last lines to get date range
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                row_count = len(rows)

                date_start = None
                date_end = None
                columns = list(rows[0].keys()) if rows else []

                if rows and "timestamp" in rows[0]:
                    date_start = rows[0]["timestamp"]
                    date_end = rows[-1]["timestamp"]

            datasets.append(
                DatasetInfo(
                    symbol=symbol,
                    timeframe=timeframe,
                    filename=str(csv_file.relative_to(data_root)),
                    row_count=row_count,
                    date_start=date_start,
                    date_end=date_end,
                    columns=columns,
                )
            )
        except Exception as e:
            # Skip files that can't be parsed
            print(f"Error parsing {csv_file}: {e}")
            continue

    return datasets


def validate_dataset(filename: str) -> ValidationReport:
    """Validate a dataset CSV file."""
    data_root = get_data_root()
    
    # Prevent path traversal attacks
    if not is_safe_path(data_root, filename):
        return ValidationReport(
            filename=filename,
            is_valid=False,
            total_rows=0,
            issues=[
                ValidationIssue(
                    severity="error",
                    message="Invalid file path or access denied"
                )
            ],
            schema_valid=False,
            timezone_valid=False,
            has_gaps=False,
            inferred_dtypes={},
        )
    
    filepath = (data_root / filename).resolve()
    issues = []
    is_valid = True
    schema_valid = True
    timezone_valid = True
    has_gaps = False
    gap_count = 0
    inferred_dtypes = {}
    total_rows = 0
    date_range = None

    if not filepath.exists():
        issues.append(
            ValidationIssue(severity="error", message=f"File not found: {filename}")
        )
        return ValidationReport(
            filename=filename,
            is_valid=False,
            total_rows=0,
            issues=issues,
            schema_valid=False,
            timezone_valid=False,
            has_gaps=False,
            inferred_dtypes={},
        )

    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total_rows = len(rows)

            if not rows:
                issues.append(
                    ValidationIssue(severity="error", message="File is empty")
                )
                is_valid = False
                schema_valid = False
                return ValidationReport(
                    filename=filename,
                    is_valid=is_valid,
                    total_rows=total_rows,
                    issues=issues,
                    schema_valid=schema_valid,
                    timezone_valid=timezone_valid,
                    has_gaps=has_gaps,
                    inferred_dtypes=inferred_dtypes,
                )

            # Check required columns for OHLCV
            required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
            columns = set(rows[0].keys())
            missing_columns = required_columns - columns

            if missing_columns:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Missing required columns: {', '.join(missing_columns)}",
                    )
                )
                is_valid = False
                schema_valid = False

            # Infer data types
            for col in columns:
                if col == "timestamp":
                    inferred_dtypes[col] = "datetime"
                elif col in {"open", "high", "low", "close", "volume"}:
                    inferred_dtypes[col] = "float"
                else:
                    inferred_dtypes[col] = "string"

            # Validate timestamps and check for gaps
            if "timestamp" in columns:
                timestamps = []
                for i, row in enumerate(rows):
                    try:
                        # Try to parse timestamp
                        ts_str = row["timestamp"]
                        # Check if timezone info is present (UTC)
                        if "+00:00" not in ts_str and "Z" not in ts_str:
                            if i == 0:  # Only report once
                                issues.append(
                                    ValidationIssue(
                                        severity="warning",
                                        message="Timestamps may not be in UTC format",
                                        row_number=i + 2,
                                    )
                                )
                                timezone_valid = False

                        # Parse timestamp
                        if "+" in ts_str:
                            ts = datetime.fromisoformat(ts_str)
                        else:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        timestamps.append(ts)
                    except Exception as e:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                message=f"Invalid timestamp format: {e}",
                                row_number=i + 2,
                            )
                        )
                        is_valid = False

                if len(timestamps) >= 2:
                    date_range = {
                        "start": timestamps[0].isoformat(),
                        "end": timestamps[-1].isoformat(),
                    }

                    # Check for gaps (simple check: look for large time differences)
                    # This is a simplified check - a real implementation would need
                    # to know the expected timeframe
                    for i in range(1, len(timestamps)):
                        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
                        # If gap is more than 2x the previous interval, flag it
                        if i >= 2:
                            prev_delta = (
                                timestamps[i - 1] - timestamps[i - 2]
                            ).total_seconds()
                            if delta > prev_delta * 2 and prev_delta > 0:
                                gap_count += 1
                                has_gaps = True
                                if gap_count <= 3:  # Only report first few gaps
                                    issues.append(
                                        ValidationIssue(
                                            severity="warning",
                                            message=f"Potential gap detected: {delta}s between rows",
                                            row_number=i + 2,
                                        )
                                    )

            # Check for NaN or missing values in OHLCV columns
            for i, row in enumerate(rows):
                for col in {"open", "high", "low", "close", "volume"}:
                    if col in row:
                        value = row[col].strip()
                        if not value or value.lower() in {"nan", "null", "none", ""}:
                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    message=f"Missing or NaN value in column '{col}'",
                                    row_number=i + 2,
                                )
                            )
                            is_valid = False

            # Check OHLC consistency
            for i, row in enumerate(rows):
                try:
                    if all(col in row for col in {"open", "high", "low", "close"}):
                        o = float(row["open"])
                        h = float(row["high"])
                        l = float(row["low"])
                        c = float(row["close"])

                        if not (l <= o <= h and l <= c <= h):
                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    message="OHLC values are inconsistent (high should be max, low should be min)",
                                    row_number=i + 2,
                                )
                            )
                            is_valid = False
                except ValueError:
                    # Already caught by NaN check
                    pass

    except Exception as e:
        issues.append(ValidationIssue(severity="error", message=f"Error reading file: {e}"))
        is_valid = False

    return ValidationReport(
        filename=filename,
        is_valid=is_valid,
        total_rows=total_rows,
        issues=issues,
        schema_valid=schema_valid,
        timezone_valid=timezone_valid,
        has_gaps=has_gaps,
        gap_count=gap_count,
        inferred_dtypes=inferred_dtypes,
        date_range=date_range,
    )
