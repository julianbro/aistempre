# Quick Start Guide

This guide will help you get started with neurotrader quickly.

## Installation

```bash
# Clone and install
git clone https://github.com/julianbro/aistempre.git
cd aistempre
pip install -e .
```

## Generate Example Data

```bash
python scripts/generate_example_data.py
```

This creates synthetic OHLCV data in `data/example/` for all configured timeframes.

## Basic Usage

### 1. Training

```bash
neurotrader-train
```

### 2. Prediction

```bash
neurotrader-predict \
  --checkpoint checkpoints/best.ckpt \
  --input data/test.csv \
  --output predictions.parquet
```

### 3. Calibration

```bash
neurotrader-calibrate \
  --checkpoint checkpoints/best.ckpt \
  --val-data data/val.csv
```

For more details, see the main [README.md](../README.md).
