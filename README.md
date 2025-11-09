# AI Trading Platform

**Multi-input, multi-horizon, probabilistic Transformer for financial time-series prediction**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Node 20](https://img.shields.io/badge/node-20-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏗️ Monorepo Structure

This repository is organized as a monorepo with the following structure:

```
aistempre/
├── frontend/          # Next.js 14 + TypeScript + Tailwind CSS
├── api/              # FastAPI + neurotrader package
├── docs/             # Documentation
├── .github/          # GitHub Actions CI/CD
├── .devcontainer/    # VSCode devcontainer configuration
└── docker-compose.yml # Local development setup
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and **pnpm** 8+
- **Python** 3.11+
- **Docker** and **Docker Compose** (for containerized setup)

### One-Command Local Setup

```bash
# Start both frontend and API
docker compose up
```

This will:
- Start the **frontend** at [http://localhost:3000](http://localhost:3000)
- Start the **API** at [http://localhost:8000](http://localhost:8000)

### Manual Development Setup

#### Frontend

```bash
cd frontend

# Install dependencies
pnpm install

# Run development server
pnpm dev

# Build for production
pnpm build

# Lint
pnpm lint

# Type check
pnpm type-check

# Format code
pnpm format
```

#### API

```bash
cd api

# Install dependencies
pip install -e ".[dev]"

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest -q

# Lint
ruff check .

# Type check
mypy app/
```

## 🎯 Key Features

### Frontend (Next.js)
- **TypeScript** with strict mode
- **Tailwind CSS** for styling
- **ESLint** and **Prettier** for code quality
- Server-side rendering and static generation
- Optimized for production

### Backend (FastAPI)
- **FastAPI** for high-performance API
- **neurotrader** package integration for ML models
- **Ruff** and **mypy** for code quality and type checking
- Async/await support
- OpenAPI documentation at `/docs`

### ML Pipeline (neurotrader)
- **Multi-Scale Transformer**: Separate encoders per timeframe with cross-attention fusion
- **Multi-Task Learning**: Regression and classification heads
- **Calibrated Outputs**: Temperature scaling, isotonic regression, and conformal prediction
- **Comprehensive Feature Engineering**: Technical indicators, volatility measures, calendar features
- **Robust Evaluation**: Purged walk-forward CV, multiple metrics, backtesting
- **Hyperparameter Tuning**: Optuna, Ray Tune PBT, Evolutionary optimization

## 📐 Architecture

### System Overview

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Browser   │─────▶│   Frontend   │─────▶│      API        │
│             │      │   (Next.js)  │      │   (FastAPI)     │
└─────────────┘      └──────────────┘      └─────────────────┘
                                                    │
                                                    ▼
                                            ┌─────────────────┐
                                            │  neurotrader    │
                                            │  ML Pipeline    │
                                            └─────────────────┘
```

### Multi-Scale Transformer

```
Input: {1m: [B, L₁, F₁], 15m: [B, L₂, F₂], ...}
  ↓
Per-Timeframe Encoders:
  Patch Embedding → Positional Encoding → N Transformer Layers
  ↓
Timeframe Embeddings Added
  ↓
Multi-Scale Fusion:
  Cross-Attention across timeframes
  ↓
Pooling to [B, d_model]
  ↓
Multi-Task Heads:
  ├─ Regression Head (Gaussian NLL / Student-t / Quantile)
  ├─ Short-Term Trend Head (3-class softmax)
  └─ Long-Term Trend Head (3-class softass)
```

## 🧪 CI/CD

GitHub Actions automatically:
- ✅ Lints and type-checks frontend code
- ✅ Builds the frontend
- ✅ Runs Python tests with pytest
- ✅ Lints Python code with ruff
- ✅ Type-checks Python code with mypy
- ✅ Builds Docker images for both services

## 📦 Docker

### Build Images

```bash
# Build frontend
docker build -t aistempre-frontend ./frontend

# Build API
docker build -t aistempre-api ./api
```

### Run with Docker Compose

```bash
# Start services
docker compose up

# Start in detached mode
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f
```

## 🔧 Development

### VSCode DevContainer

This project includes a devcontainer configuration for VSCode:

1. Install the **Remote - Containers** extension
2. Open the project in VSCode
3. Press `F1` and select **"Remote-Containers: Reopen in Container"**

The devcontainer will automatically:
- Set up Python and Node.js environments
- Install all dependencies
- Configure linting and formatting

### Code Quality

We enforce strict code quality standards:

**Frontend:**
- ESLint with Next.js recommended rules
- Prettier for consistent formatting
- TypeScript in strict mode

**Backend:**
- Ruff for fast Python linting
- mypy for static type checking
- pytest for testing

## 📚 Documentation

- [Frontend README](./frontend/README.md)
- [API README](./api/README.md)
- [neurotrader Documentation](./docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Risk Disclaimer

**This software is for research and educational purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Beware of overfitting and look-ahead bias
- Always validate on held-out test sets
- Use proper risk management in live trading

## 🙏 Acknowledgments

- Built with [Next.js](https://nextjs.org/), [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/), and [PyTorch Lightning](https://lightning.ai/)
- Configuration management via [Hydra](https://hydra.cc/)
- Inspired by research in financial ML and transformer architectures

---

**Remember:** Financial markets are complex and unpredictable. This tool is meant to aid research and analysis, not to provide trading signals. Always do your own due diligence and never risk more than you can afford to lose.
