# neurotrader

**Multi-input, multi-horizon, probabilistic Transformer for financial time-series prediction**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

`neurotrader` is a comprehensive Python package for training and serving deep learning models for financial time-series prediction. It implements a Multi-Scale Transformer architecture that:

- **Predicts next price movements** using log-returns with calibrated probability distributions
- **Classifies short-term and long-term trends** (UP/DOWN/FLAT) with epsilon bands
- **Outputs calibrated probabilities** and prediction intervals for uncertainty quantification
- **Supports multi-resolution inputs** across different timeframes (1m, 15m, 4h, 1d, 1w)
- **Provides full ML pipeline** from data loading to backtesting

## ✨ Key Features

### 🎯 Multi-Task Learning
- **Regression**: Next-price prediction with Gaussian NLL, Student-t, or Quantile heads
- **Classification**: Short-term and long-term trend prediction with configurable horizons
- **Calibrated Outputs**: Temperature scaling, isotonic regression, and conformal prediction

### 🔧 Flexible Architecture
- **Multi-Scale Transformer**: Separate encoders per timeframe with cross-attention fusion
- **Patching**: Efficient processing of long sequences via patch embeddings
- **Multiple Loss Functions**: MSE, MAE, Huber, Quantile, Gaussian NLL, Student-t NLL, Cross-Entropy, Focal

### 📊 Comprehensive Feature Engineering
- **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands, ATR, ADX, Stochastic
- **Price Features**: Log returns, cumulative returns, VWAP, z-scored price
- **Volatility Measures**: Realized volatility, Parkinson, Garman-Klass
- **Calendar Features**: Hour/day/month encoding, session flags
- **Plugin System**: Easy-to-extend feature registry

### 🛡️ Robust Evaluation
- **Purged Walk-Forward CV**: No data leakage between folds
- **Multiple Metrics**: RMSE, MAE, sMAPE, Directional Accuracy, F1, AUROC, MCC, ECE, Brier
- **Backtesting**: Sharpe ratio, Sortino ratio, Maximum Drawdown, Calmar ratio

### 🔬 Hyperparameter Tuning
- **Optuna**: TPE and CMA-ES samplers with pruning
- **Ray Tune PBT**: Population-based training for scalable optimization
- **Evolutionary**: DEAP/Nevergrad for strategy threshold optimization

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/julianbro/aistempre.git
cd aistempre

# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Requirements
- Python 3.11+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Hydra for configuration management
- See `pyproject.toml` for full dependencies

## 🚀 Quick Start

### 1. Prepare Your Data

Place your OHLCV data in CSV or Parquet format:

```
data/
  BTCUSDT_1m.csv
  BTCUSDT_15m.csv
  BTCUSDT_4h.csv
  BTCUSDT_1d.csv
  BTCUSDT_1w.csv
```

Each file should have columns: `timestamp, open, high, low, close, volume`

### 2. Configure Your Experiment

Edit configuration files in `configs/`:

```yaml
# configs/data.yaml
source:
  type: csv
  path: ./data
timeframes: [1m, 15m, 4h, 1d, 1w]
start_date: "2020-01-01"
end_date: "2023-12-31"

# configs/model.yaml
variant: base  # base | medium | large
base:
  d_model: 256
  n_heads: 8
  n_layers_tf: 2
  patch_len: 16

# configs/train.yaml
trainer:
  max_epochs: 100
  accelerator: auto
optimizer:
  lr: 2.0e-4
  weight_decay: 0.05
```

### 3. Train a Model

```bash
# Train with default config
neurotrader-train

# Train with custom config
neurotrader-train --config-name my_config.yaml

# Override specific parameters
neurotrader-train -o model.variant=large -o trainer.max_epochs=200
```

### 4. Run Predictions

```bash
# Generate predictions
neurotrader-predict \
  --checkpoint checkpoints/best_model.ckpt \
  --input data/test_data.csv \
  --output predictions.parquet
```

### 5. Calibrate Probabilities

```bash
# Calibrate on validation set
neurotrader-calibrate \
  --checkpoint checkpoints/best_model.ckpt \
  --val-data data/val_data.csv \
  --output calibrated_model.ckpt
```

### 6. Backtest Strategy

```bash
# Run backtest
neurotrader-backtest \
  --predictions predictions.parquet \
  --prices data/historical_prices.csv \
  --cash 100000 \
  --output backtest_results.json
```

### 7. Tune Hyperparameters

```bash
# Optuna tuning
neurotrader-tune --backend optuna --n-trials 100

# Ray Tune PBT
neurotrader-tune --backend ray-pbt --time-budget-hours 6

# Evolutionary optimization
neurotrader-tune --backend evolutionary
```

## 📐 Model Architecture

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
  └─ Long-Term Trend Head (3-class softmax)
  ↓
Outputs: {
  regression: {mu, var},
  short_trend: {logits, probs},
  long_trend: {logits, probs}
}
```

### Recommended Sizes

**Base** (default):
- `d_model=256, n_heads=8, n_layers_tf=2, n_layers_fusion=2`
- Suitable for single-symbol training on consumer GPUs

**Medium**:
- `d_model=512, n_heads=8, n_layers_tf=3, n_layers_fusion=3`
- More capacity, requires more data and compute

**Large**:
- `d_model=768, n_heads=12, n_layers_tf=4, n_layers_fusion=4`
- For large datasets and multi-GPU setups

## 🎯 Probability of Correct Output

### Classification
After temperature scaling calibration:
```python
p_correct = max(calibrated_probs)
```

Track calibration quality with ECE and Brier score.

### Regression
For Gaussian/Student-t heads:
```python
# Probability within epsilon band (e.g., 10 bps)
p_within_eps = CDF(+ε/σ) - CDF(-ε/σ)
```

For conformal prediction:
```python
# Empirical coverage of (1-α) prediction intervals
coverage_rate = fraction_of_targets_in_intervals
```

## 🔬 Loss Functions

Easily swap losses via `configs/loss.yaml`:

```yaml
regression:
  loss_type: gaussian_nll  # mse | mae | huber | quantile | gaussian_nll | student_t_nll
classification:
  loss_type: cross_entropy  # cross_entropy | focal
weights:
  regression: 0.5
  short_trend: 0.25
  long_trend: 0.25
thresholds:
  epsilon_bps: 10
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_losses.py
pytest tests/test_splitter.py
pytest tests/test_labels.py

# With coverage
pytest --cov=neurotrader --cov-report=html
```

## 📚 Project Structure

```
neurotrader/
├── configs/               # Hydra configuration files
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   ├── loss.yaml
│   ├── features.yaml
│   └── tune.yaml
├── src/neurotrader/
│   ├── cli.py            # CLI entrypoints
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── labels/           # Target generation
│   ├── models/           # Model architecture
│   ├── losses/           # Loss functions and calibration
│   ├── training/         # Training loop and metrics
│   ├── tuning/           # Hyperparameter optimization
│   ├── inference/        # Prediction and serving
│   ├── backtest/         # Backtesting utilities
│   └── utils/            # Common utilities
├── scripts/              # Training/inference scripts
└── tests/                # Unit tests
```

## ⚠️ Important Guardrails

### Data Leakage Prevention
- ✅ Scalers fit only on training data
- ✅ Features computed using only past information
- ✅ Purged walk-forward cross-validation
- ✅ Strict UTC time alignment
- ✅ No forward-filling across session boundaries

### Evaluation Best Practices
- ✅ Walk-forward validation (no peeking into future)
- ✅ Report out-of-sample metrics per fold
- ✅ Never tune on test set
- ✅ Recalibrate after any retraining

### Risk Disclaimer
⚠️ **This software is for research and educational purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Beware of overfitting and look-ahead bias
- Always validate on held-out test sets
- Use proper risk management in live trading

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Configuration management via [Hydra](https://hydra.cc/)
- Inspired by research in financial ML and transformer architectures

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Remember:** Financial markets are complex and unpredictable. This tool is meant to aid research and analysis, not to provide trading signals. Always do your own due diligence and never risk more than you can afford to lose.
