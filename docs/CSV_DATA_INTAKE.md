# CSV Data Intake and Validation

This document describes the CSV data intake and validation features added to the AI Trading Platform.

## Overview

The platform now supports:
1. **Backend API**: Dataset discovery and validation endpoints
2. **Frontend UI**: Data Explorer page for browsing and validating CSV files

## Backend API

### Configuration

Add the data root directory to your `.env` file:

```env
DATA_ROOT=./data
```

### Endpoints

#### GET /datasets

Lists all available CSV datasets with metadata.

**Response:**
```json
{
  "datasets": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "1m",
      "filename": "example/BTCUSDT_1m.csv",
      "row_count": 5,
      "date_start": "2024-01-01T00:00:00+00:00",
      "date_end": "2024-01-01T00:04:00+00:00",
      "columns": ["timestamp", "open", "high", "low", "close", "volume"]
    }
  ],
  "total_count": 1
}
```

**Example:**
```bash
curl http://localhost:8000/datasets
```

#### POST /datasets/validate

Validates a CSV dataset file.

**Request:**
```json
{
  "filename": "example/BTCUSDT_1m.csv"
}
```

**Response:**
```json
{
  "filename": "example/BTCUSDT_1m.csv",
  "is_valid": true,
  "total_rows": 5,
  "issues": [],
  "schema_valid": true,
  "timezone_valid": true,
  "has_gaps": false,
  "gap_count": 0,
  "inferred_dtypes": {
    "timestamp": "datetime",
    "open": "float",
    "high": "float",
    "low": "float",
    "close": "float",
    "volume": "float"
  },
  "date_range": {
    "start": "2024-01-01T00:00:00+00:00",
    "end": "2024-01-01T00:04:00+00:00"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/datasets/validate \
  -H "Content-Type: application/json" \
  -d '{"filename": "example/BTCUSDT_1m.csv"}'
```

### Validation Checks

The validator performs the following checks:

1. **Schema Validation**: Ensures all required OHLCV columns are present (timestamp, open, high, low, close, volume)
2. **Timezone Validation**: Checks that timestamps are in UTC format
3. **Data Gaps**: Detects gaps in the time series
4. **OHLC Consistency**: Validates that high is max and low is min for each row
5. **Missing Values**: Checks for NaN or missing values in OHLCV columns

### CSV Format

CSV files should follow this format:

```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00+00:00,42000.0,42100.0,41900.0,42050.0,150.5
2024-01-01T00:01:00+00:00,42050.0,42150.0,42000.0,42100.0,200.3
```

**Requirements:**
- Filename format: `SYMBOL_TIMEFRAME.csv` (e.g., `BTCUSDT_1m.csv`)
- Required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamps must be in ISO format with UTC timezone
- No missing or NaN values in OHLCV columns
- OHLC values must be consistent (high >= open/close, low <= open/close)

### File Organization

Place CSV files in the data directory:

```
data/
├── example/
│   ├── BTCUSDT_1m.csv
│   ├── BTCUSDT_15m.csv
│   └── ETHUSDT_1d.csv
└── production/
    └── ...
```

## Frontend UI

### Data Explorer Page

Access the Data Explorer at: `http://localhost:3000/data`

**Features:**
- **Dataset List**: Browse all available datasets grouped by symbol
- **Timeframe Filters**: View different timeframes (1m, 15m, 4h, 1d, 1w)
- **Validation**: Click "Validate" to check dataset quality
- **Chart Placeholder**: Ready for TradingView or lightweight-charts integration
- **Indicator Controls**: Toggle EMA, Volume, RSI overlays (placeholder)

### Validation Modal

The validation modal displays:
- Overall validation status (Valid/Invalid)
- Individual check results:
  - Schema validation (OHLCV columns)
  - Timezone validation (UTC)
  - Data gap detection
- Inferred data types for each column
- Detailed list of issues with severity levels (error, warning, info)

## Security

### Path Traversal Protection

The API includes protection against path traversal attacks:
- User-provided filenames are validated before file operations
- Paths are resolved and checked to ensure they're within the configured data root
- Invalid paths return an error response

## Development

### Running the Backend

```bash
cd api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running the Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

### Running Tests

```bash
cd api
pytest tests/test_datasets.py -v
```

### Type Checking

```bash
cd frontend
pnpm type-check
```

### Linting

```bash
cd frontend
pnpm lint
```

## Next Steps

Future enhancements:
1. Integrate TradingView or lightweight-charts for OHLC visualization
2. Add actual indicator calculations (EMA, RSI, etc.)
3. Implement CSV data streaming for large files
4. Add dataset statistics and analytics
5. Support for additional data formats (Parquet, HDF5)
6. Automatic data quality scoring
7. Data preprocessing and cleaning tools
