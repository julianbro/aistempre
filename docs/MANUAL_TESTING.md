# Manual Testing Guide

This guide provides manual testing steps for the CSV Data Intake and Validation features.

## Prerequisites

1. **Backend**: Python 3.11+, FastAPI, pydantic, python-dotenv
2. **Frontend**: Node.js 18+, pnpm 8+

## Setting Up Test Data

Create test CSV files in the `data/example/` directory:

```bash
mkdir -p data/example
```

Example test files have been created:
- `data/example/BTCUSDT_1m.csv`
- `data/example/BTCUSDT_15m.csv`
- `data/example/ETHUSDT_1d.csv`

## Backend Testing

### 1. Start the Backend Server

```bash
cd api
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test GET /datasets Endpoint

```bash
curl http://localhost:8000/datasets | jq
```

**Expected Response:**
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
  "total_count": 3
}
```

**Validation:**
- ✓ Returns list of datasets
- ✓ Includes metadata for each dataset
- ✓ Row counts are accurate

### 3. Test POST /datasets/validate Endpoint

```bash
curl -X POST http://localhost:8000/datasets/validate \
  -H "Content-Type: application/json" \
  -d '{"filename": "example/BTCUSDT_1m.csv"}' | jq
```

**Expected Response:**
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

**Validation:**
- ✓ Returns validation report
- ✓ Schema is validated
- ✓ Timezone is checked
- ✓ Data types are inferred correctly

### 4. Test Path Traversal Protection

Try to access files outside the data directory:

```bash
curl -X POST http://localhost:8000/datasets/validate \
  -H "Content-Type: application/json" \
  -d '{"filename": "../../../etc/passwd"}' | jq
```

**Expected Response:**
```json
{
  "filename": "../../../etc/passwd",
  "is_valid": false,
  "total_rows": 0,
  "issues": [
    {
      "severity": "error",
      "message": "Invalid file path or access denied",
      "row_number": null
    }
  ],
  "schema_valid": false,
  "timezone_valid": false,
  "has_gaps": false,
  "gap_count": 0,
  "inferred_dtypes": {}
}
```

**Validation:**
- ✓ Path traversal attempts are blocked
- ✓ Returns error message
- ✓ Does not access files outside data root

### 5. Test OpenAPI Documentation

Visit: http://localhost:8000/docs

**Validation:**
- ✓ API documentation is accessible
- ✓ Endpoints are documented
- ✓ Can test endpoints from the UI

## Frontend Testing

### 1. Start the Frontend Server

```bash
cd frontend
pnpm install
pnpm dev
```

### 2. Test Home Page

Visit: http://localhost:3000

**Validation:**
- ✓ Page loads successfully
- ✓ "Data Explorer" button is visible
- ✓ Clicking button navigates to /data

### 3. Test Data Explorer Page

Visit: http://localhost:3000/data

**Validation:**
- ✓ Page loads successfully
- ✓ Dataset list appears in sidebar
- ✓ Datasets are grouped by symbol
- ✓ Each dataset shows:
  - Symbol name
  - Timeframe badge
  - Row count
  - Date range
  - Validate button

### 4. Test Dataset Selection

Click on a dataset in the list:

**Validation:**
- ✓ Dataset is highlighted
- ✓ Chart placeholder appears in main panel
- ✓ Indicator controls are visible
- ✓ Dataset statistics panel is shown

### 5. Test Validation Modal

Click the "Validate" button on a dataset:

**Validation:**
- ✓ Modal opens
- ✓ Shows validation status (Valid/Invalid)
- ✓ Displays total rows
- ✓ Shows validation checks:
  - Schema validation status
  - Timezone validation status
  - Gap detection results
- ✓ Lists inferred data types
- ✓ Shows issues (if any) with severity levels
- ✓ Close button works

### 6. Test Error Handling

With the backend stopped, try to use the Data Explorer:

**Validation:**
- ✓ Shows error message when backend is unavailable
- ✓ Error message is user-friendly
- ✓ App doesn't crash

## Build & Lint Tests

### Frontend

```bash
cd frontend

# Type checking
pnpm type-check  # Should pass

# Linting
pnpm lint  # Should pass

# Build
pnpm build  # Should complete successfully
```

**Validation:**
- ✓ No TypeScript errors
- ✓ No ESLint errors
- ✓ Build completes without errors
- ✓ All pages are generated

## Integration Tests

### End-to-End Flow

1. Start both backend and frontend servers
2. Navigate to http://localhost:3000
3. Click "Data Explorer"
4. Select a dataset from the list
5. Click "Validate"
6. Review validation results
7. Close modal
8. Select another dataset

**Validation:**
- ✓ Full flow works without errors
- ✓ Data flows correctly from backend to frontend
- ✓ UI is responsive and user-friendly

## Troubleshooting

### Backend not starting
- Check Python version (3.11+)
- Install dependencies: `pip install -e ".[dev]"`
- Check port 8000 is available

### Frontend not loading datasets
- Check backend is running on port 8000
- Check CORS settings in backend
- Check browser console for errors
- Verify NEXT_PUBLIC_API_URL in .env

### Validation not working
- Check CSV file format
- Verify file is in data directory
- Check file permissions
- Review backend logs for errors

## Success Criteria

All tests should pass with:
- ✅ Backend endpoints return correct data
- ✅ Path traversal protection is working
- ✅ Frontend loads and displays datasets
- ✅ Validation modal shows correct information
- ✅ No TypeScript or linting errors
- ✅ Build completes successfully
