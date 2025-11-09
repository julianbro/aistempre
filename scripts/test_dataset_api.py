#!/usr/bin/env python3
"""
Manual test script for dataset API endpoints.
This script can be run without starting the full server.
"""

import sys
import os

# Add the api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from app.datasets import discover_datasets, validate_dataset

def test_discover_datasets():
    """Test dataset discovery."""
    print("Testing dataset discovery...")
    datasets = discover_datasets()
    
    print(f"\nFound {len(datasets)} datasets:")
    for ds in datasets:
        print(f"  - {ds.symbol} ({ds.timeframe}): {ds.row_count} rows")
        if ds.date_start and ds.date_end:
            print(f"    Date range: {ds.date_start} to {ds.date_end}")
    
    return datasets

def test_validate_dataset(filename):
    """Test dataset validation."""
    print(f"\nValidating {filename}...")
    report = validate_dataset(filename)
    
    print(f"  Valid: {report.is_valid}")
    print(f"  Total rows: {report.total_rows}")
    print(f"  Schema valid: {report.schema_valid}")
    print(f"  Timezone valid: {report.timezone_valid}")
    print(f"  Has gaps: {report.has_gaps} (count: {report.gap_count})")
    
    if report.inferred_dtypes:
        print(f"  Data types: {report.inferred_dtypes}")
    
    if report.issues:
        print(f"  Issues ({len(report.issues)}):")
        for issue in report.issues[:5]:  # Show first 5 issues
            print(f"    [{issue.severity}] {issue.message}")
            if issue.row_number:
                print(f"      at row {issue.row_number}")
    
    return report

def main():
    """Run manual tests."""
    print("=" * 60)
    print("Dataset API Manual Tests")
    print("=" * 60)
    
    # Test discovery
    datasets = test_discover_datasets()
    
    # Test validation for each dataset
    if datasets:
        print("\n" + "=" * 60)
        print("Validation Tests")
        print("=" * 60)
        
        for ds in datasets[:3]:  # Test first 3 datasets
            test_validate_dataset(ds.filename)
    
    # Test path traversal protection
    print("\n" + "=" * 60)
    print("Security Tests")
    print("=" * 60)
    print("\nTesting path traversal protection...")
    
    dangerous_paths = [
        "../../../etc/passwd",
        "../../.env",
        "./data/../../../secret.txt"
    ]
    
    for path in dangerous_paths:
        report = validate_dataset(path)
        if not report.is_valid and any("Invalid file path" in issue.message for issue in report.issues):
            print(f"  ✓ Blocked: {path}")
        else:
            print(f"  ✗ FAILED TO BLOCK: {path}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
