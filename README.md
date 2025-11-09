# Neurotrader Monorepo

**Multi-input, multi-horizon, probabilistic Transformer for financial time-series prediction**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Node 20](https://img.shields.io/badge/node-20-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

This is a monorepo containing:

- **Frontend** (`frontend/`): Next.js 16 web application with TypeScript and Tailwind CSS
- **API** (`api/`): FastAPI backend with the core neurotrader ML package

## 📦 Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+
- PNPM (installed automatically via npm)
- Docker & Docker Compose (optional, for containerized deployment)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/julianbro/aistempre.git
cd aistempre

# Start both services with Docker Compose
docker compose up
```

This will start:
- Frontend at http://localhost:3000
- API at http://localhost:8000

### Local Development

#### Install Dependencies

```bash
# Install all dependencies (frontend + root)
pnpm install

# Install Python backend dependencies
cd api
pip install -r requirements-dev.txt
cd ..
```

#### Development Mode

```bash
# Run both frontend and API in development mode
pnpm dev

# Or run them separately:
pnpm dev:frontend   # Frontend only at :3000
pnpm dev:api        # API only at :8000
```

## 🏗️ Project Structure

```
neurotrader/
├── frontend/                 # Next.js web application
│   ├── src/
│   │   └── app/             # App router pages
│   ├── public/              # Static assets
│   ├── package.json
│   └── tsconfig.json
│
├── api/                      # FastAPI backend + neurotrader ML package
│   ├── main.py              # FastAPI application entry point
│   ├── src/neurotrader/     # Core ML package
│   │   ├── models/          # Transformer architectures
│   │   ├── features/        # Feature engineering
│   │   ├── losses/          # Loss functions & calibration
│   │   ├── training/        # Training utilities
│   │   ├── tuning/          # Hyperparameter optimization
│   │   ├── inference/       # Prediction & serving
│   │   └── backtest/        # Backtesting utilities
│   ├── tests/               # Backend tests
│   ├── configs/             # Hydra configuration files
│   ├── requirements.txt     # Production dependencies
│   ├── requirements-dev.txt # Development dependencies
│   └── pyproject.toml       # Python package configuration
│
├── .github/workflows/        # CI/CD pipelines
│   ├── frontend-ci.yml      # Frontend build, lint, test
│   ├── backend-ci.yml       # Backend pytest, ruff, mypy
│   └── docker-build.yml     # Docker image builds
│
├── .devcontainer/           # VSCode devcontainer configuration
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile.frontend      # Frontend Docker image
├── Dockerfile.api           # API Docker image
├── pnpm-workspace.yaml      # PNPM workspace configuration
├── package.json             # Root package with workspace scripts
└── .env.example             # Environment variables template
```

## 🛠️ Development Commands

### Frontend

```bash
pnpm -C frontend dev         # Start dev server
pnpm -C frontend build       # Build for production
pnpm -C frontend lint        # Run ESLint
pnpm -C frontend format      # Format with Prettier
pnpm -C frontend type-check  # TypeScript type checking
pnpm -C frontend test        # Run tests
```

### API

```bash
cd api

# Run FastAPI server
uvicorn main:app --reload

# Linting & Formatting
ruff check .                 # Check code with Ruff
ruff format .                # Format code with Ruff

# Type Checking
mypy src/ main.py --ignore-missing-imports

# Testing
pytest -q                    # Run tests (quiet mode)
pytest -v                    # Run tests (verbose)
pytest --cov=neurotrader     # Run with coverage
```

### Workspace (Root)

```bash
pnpm dev                     # Run both frontend and API
pnpm build                   # Build frontend
pnpm lint                    # Lint both frontend and API
pnpm format                  # Format both codebases
pnpm type-check              # Type check both projects
pnpm test                    # Run all tests
pnpm docker:up               # Start with Docker Compose
pnpm docker:down             # Stop Docker Compose
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:
- `NEXT_PUBLIC_API_URL`: Frontend API endpoint (default: http://localhost:8000)
- `PORT`: API server port (default: 8000)
- `DATA_DIR`: Directory for training data
- `CACHE_DIR`: Directory for cached features

### Code Quality Tools

#### Frontend
- **ESLint**: Linting with Next.js recommended rules
- **Prettier**: Code formatting (100 char line length)
- **TypeScript**: Strict mode enabled
- **Tailwind CSS**: Utility-first CSS framework

#### Backend
- **Ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **pytest**: Testing framework
- **Black-compatible**: Formatting follows Black style

## 🐳 Docker Deployment

### Build and Run

```bash
# Build images
docker compose build

# Start services
docker compose up

# Start in detached mode
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Individual Services

```bash
# Build frontend image
docker build -f Dockerfile.frontend -t neurotrader-frontend .

# Build API image
docker build -f Dockerfile.api -t neurotrader-api .

# Run frontend
docker run -p 3000:3000 neurotrader-frontend

# Run API
docker run -p 8000:8000 neurotrader-api
```

## 🧪 CI/CD

GitHub Actions workflows automatically run on push and PR:

### Frontend CI
- ESLint check
- Prettier format check
- TypeScript type checking
- Unit tests
- Production build

### Backend CI
- Ruff linting
- Ruff format check
- mypy type checking
- pytest test suite

### Docker Build
- Build frontend Docker image
- Build API Docker image
- Cache layers for faster builds

## 📚 API Endpoints

Once running, the API provides:

- `GET /` - API information
- `GET /health` - Health check
- `GET /api/models` - List available models
- `GET /api/status` - System status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## 🔬 ML Features

The neurotrader package includes:

### Model Architecture
- Multi-Scale Transformer with cross-attention fusion
- Support for multiple timeframes (1m, 15m, 4h, 1d, 1w)
- Patch-based embeddings for efficient processing

### Multi-Task Learning
- Regression: Next-price prediction with uncertainty
- Classification: Short/long-term trend prediction
- Calibrated probability outputs

### Feature Engineering
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price features (returns, VWAP, z-score)
- Volatility measures (realized vol, Parkinson, Garman-Klass)
- Calendar features (hour, day, month encoding)

### Training & Optimization
- Purged walk-forward cross-validation
- Hyperparameter tuning (Optuna, Ray Tune, Evolutionary)
- Multiple loss functions (MSE, MAE, Huber, Quantile, NLL)
- Probability calibration (temperature scaling, isotonic regression)

### Evaluation & Backtesting
- Comprehensive metrics (RMSE, MAE, Sharpe, Sortino, etc.)
- Directional accuracy and classification metrics
- Full backtesting framework with risk metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run linting and tests
5. Submit a pull request

Ensure all checks pass:
```bash
pnpm lint && pnpm type-check && pnpm test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Risk Disclaimer

**This software is for research and educational purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Always validate on held-out test sets
- Use proper risk management in live trading

## 🙏 Acknowledgments

- Built with [Next.js](https://nextjs.org/), [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/), and [PyTorch Lightning](https://lightning.ai/)
- Configuration management via [Hydra](https://hydra.cc/)
- Inspired by research in financial ML and transformer architectures

---

**Remember:** Financial markets are complex and unpredictable. This tool is meant to aid research and analysis, not to provide trading signals.
