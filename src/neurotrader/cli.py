"""
CLI entrypoints for neurotrader using Typer.
"""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="neurotrader",
    help="Multi-horizon probabilistic Transformer for financial time-series",
    add_completion=False,
)


@app.command()
def train(
    config_name: str = typer.Option("train.yaml", "--config-name", help="Training config file"),
    config_dir: Path = typer.Option("./configs", "--config-dir", help="Config directory"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="Resume from checkpoint"),
    overrides: Optional[list[str]] = typer.Option(None, "--override", "-o", help="Config overrides"),
):
    """Train a neurotrader model."""
    typer.echo(f"Training with config: {config_name}")
    typer.echo(f"Config directory: {config_dir}")
    
    if checkpoint:
        typer.echo(f"Resuming from checkpoint: {checkpoint}")
    
    # Import here to avoid circular dependencies
    from neurotrader.utils.logging import setup_logger
    from neurotrader.utils.seed import set_seed
    
    logger = setup_logger("train")
    logger.info("Starting training...")
    
    # This will be implemented in scripts/train.py
    typer.echo("Training functionality will be implemented in the training module.")


@app.command()
def predict(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Model checkpoint path"),
    input_data: Path = typer.Option(..., "--input", help="Input CSV/Parquet file"),
    output: Path = typer.Option("predictions.parquet", "--output", help="Output file"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for inference"),
):
    """Run inference with a trained model."""
    typer.echo(f"Loading model from: {checkpoint}")
    typer.echo(f"Input data: {input_data}")
    typer.echo(f"Output: {output}")
    
    from neurotrader.utils.logging import setup_logger
    
    logger = setup_logger("predict")
    logger.info("Starting prediction...")
    
    # This will be implemented in scripts/predict.py
    typer.echo("Prediction functionality will be implemented in the inference module.")


@app.command()
def tune(
    backend: str = typer.Option("optuna", "--backend", help="Tuning backend: optuna|ray-pbt|evolutionary"),
    n_trials: Optional[int] = typer.Option(None, "--n-trials", help="Number of trials (Optuna)"),
    time_budget_hours: Optional[float] = typer.Option(None, "--time-budget-hours", help="Time budget (Ray PBT)"),
    config_dir: Path = typer.Option("./configs", "--config-dir", help="Config directory"),
):
    """Run hyperparameter tuning."""
    typer.echo(f"Tuning with backend: {backend}")
    
    if n_trials:
        typer.echo(f"Number of trials: {n_trials}")
    if time_budget_hours:
        typer.echo(f"Time budget: {time_budget_hours} hours")
    
    from neurotrader.utils.logging import setup_logger
    
    logger = setup_logger("tune")
    logger.info(f"Starting hyperparameter tuning with {backend}...")
    
    # This will be implemented in scripts/tune.py
    typer.echo("Tuning functionality will be implemented in the tuning module.")


@app.command()
def calibrate(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Model checkpoint path"),
    val_data: Path = typer.Option(..., "--val-data", help="Validation data for calibration"),
    output: Path = typer.Option("calibrated_model.ckpt", "--output", help="Output calibrated checkpoint"),
    method: str = typer.Option("temperature", "--method", help="Calibration method: temperature|isotonic|conformal"),
):
    """Calibrate model probabilities."""
    typer.echo(f"Calibrating model: {checkpoint}")
    typer.echo(f"Validation data: {val_data}")
    typer.echo(f"Method: {method}")
    
    from neurotrader.utils.logging import setup_logger
    
    logger = setup_logger("calibrate")
    logger.info("Starting model calibration...")
    
    # This will be implemented in scripts/calibrate.py
    typer.echo("Calibration functionality will be implemented in the losses module.")


@app.command()
def backtest(
    predictions: Path = typer.Option(..., "--predictions", help="Predictions file"),
    prices: Path = typer.Option(..., "--prices", help="Historical prices file"),
    cash: float = typer.Option(100000.0, "--cash", help="Initial cash"),
    output: Path = typer.Option("backtest_results.json", "--output", help="Output results file"),
):
    """Run backtest on predictions."""
    typer.echo(f"Backtesting predictions: {predictions}")
    typer.echo(f"Historical prices: {prices}")
    typer.echo(f"Initial cash: ${cash:,.2f}")
    
    from neurotrader.utils.logging import setup_logger
    
    logger = setup_logger("backtest")
    logger.info("Starting backtest...")
    
    # This will be implemented in scripts/backtest.py
    typer.echo("Backtest functionality will be implemented in the backtest module.")


@app.command()
def export_onnx(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Model checkpoint path"),
    output: Path = typer.Option("model.onnx", "--output", help="Output ONNX file"),
    opset_version: int = typer.Option(14, "--opset-version", help="ONNX opset version"),
):
    """Export model to ONNX format."""
    typer.echo(f"Exporting model: {checkpoint}")
    typer.echo(f"Output: {output}")
    typer.echo(f"ONNX opset version: {opset_version}")
    
    from neurotrader.utils.logging import setup_logger
    
    logger = setup_logger("export")
    logger.info("Starting ONNX export...")
    
    # This will be implemented in scripts/export_onnx.py
    typer.echo("ONNX export functionality will be implemented in the inference module.")


def train_command():
    """Entry point for neurotrader-train CLI command."""
    app(["train"])


def predict_command():
    """Entry point for neurotrader-predict CLI command."""
    app(["predict"])


def tune_command():
    """Entry point for neurotrader-tune CLI command."""
    app(["tune"])


def calibrate_command():
    """Entry point for neurotrader-calibrate CLI command."""
    app(["calibrate"])


def backtest_command():
    """Entry point for neurotrader-backtest CLI command."""
    app(["backtest"])


def export_onnx_command():
    """Entry point for neurotrader-export-onnx CLI command."""
    app(["export-onnx"])


if __name__ == "__main__":
    app()
