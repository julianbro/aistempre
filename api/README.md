# AI Trading Platform - API

FastAPI backend for the AI Trading Platform with integrated neurotrader package.

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest -q

# Lint
ruff check .

# Type check
mypy app/
```
