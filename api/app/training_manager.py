"""Training run manager for orchestrating training jobs."""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess
import threading

from app.models import (
    TrainingRun,
    RunConfig,
    RunStatus,
    RunMetrics,
    CheckpointInfo,
)


class TrainingManager:
    """Manages training runs and their lifecycle."""

    def __init__(self, runs_dir: str = "./runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, TrainingRun] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_buffers: dict[str, list[str]] = {}
        self._load_existing_runs()

    def _load_existing_runs(self):
        """Load existing runs from disk."""
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                metadata_file = run_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            data = json.load(f)
                            run = TrainingRun(**data)
                            self.runs[run.id] = run
                    except Exception as e:
                        print(f"Error loading run {run_dir.name}: {e}")

    def _save_run(self, run: TrainingRun):
        """Save run metadata to disk."""
        run_dir = self.runs_dir / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(run.model_dump(), f, indent=2, default=str)

    def create_run(self, config: RunConfig, name: Optional[str] = None) -> TrainingRun:
        """Create a new training run."""
        run_id = name or str(uuid.uuid4())
        run = TrainingRun(
            id=run_id,
            status=RunStatus.PENDING,
            config=config,
            created_at=datetime.utcnow(),
        )
        self.runs[run_id] = run
        self._save_run(run)
        self.log_buffers[run_id] = []
        return run

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        """Get a training run by ID."""
        return self.runs.get(run_id)

    def list_runs(self) -> list[TrainingRun]:
        """List all training runs."""
        return list(self.runs.values())

    def start_run(self, run_id: str) -> bool:
        """Start a training run (dummy implementation for now)."""
        run = self.runs.get(run_id)
        if not run or run.status != RunStatus.PENDING:
            return False

        run.status = RunStatus.RUNNING
        run.started_at = datetime.utcnow()
        self._save_run(run)

        # Start a background thread to simulate training
        thread = threading.Thread(target=self._simulate_training, args=(run_id,))
        thread.daemon = True
        thread.start()

        return True

    def _simulate_training(self, run_id: str):
        """Simulate a training run with dummy metrics (for testing)."""
        import time
        import random

        run = self.runs[run_id]
        max_epochs = run.config.max_epochs

        for epoch in range(1, max_epochs + 1):
            if run.status != RunStatus.RUNNING:
                break

            # Simulate training progress
            train_loss = 1.0 / (epoch + 1) + random.uniform(-0.1, 0.1)
            val_loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)

            run.metrics = RunMetrics(
                epoch=epoch,
                step=epoch * 100,
                train_loss=max(0.1, train_loss),
                val_loss=max(0.1, val_loss),
                val_da=min(0.95, 0.5 + epoch * 0.01),
                val_f1=min(0.9, 0.4 + epoch * 0.01),
                val_rmse=max(0.01, 1.0 - epoch * 0.01),
                learning_rate=run.config.learning_rate * (0.99**epoch),
                gpu_memory_mb=random.uniform(2000, 4000),
                gpu_utilization=random.uniform(70, 95),
                eta_minutes=max(0, (max_epochs - epoch) * 0.5),
            )

            # Log message
            log_msg = f"Epoch {epoch}/{max_epochs} - loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
            self.log_buffers[run_id].append(log_msg)

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0 or epoch == max_epochs:
                checkpoint = CheckpointInfo(
                    filename=f"checkpoint-epoch{epoch:02d}.ckpt",
                    epoch=epoch,
                    val_loss=val_loss,
                    val_score=run.metrics.val_f1,
                    size_mb=random.uniform(100, 500),
                    created_at=datetime.utcnow(),
                )
                run.checkpoints.append(checkpoint)

            self._save_run(run)
            time.sleep(0.5)  # Simulate epoch duration

        # Complete the run
        if run.status == RunStatus.RUNNING:
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.utcnow()
            self._save_run(run)

    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running training run."""
        run = self.runs.get(run_id)
        if not run or run.status != RunStatus.RUNNING:
            return False

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.utcnow()
        self._save_run(run)

        # Kill process if exists
        if run_id in self.processes:
            try:
                self.processes[run_id].terminate()
                self.processes[run_id].wait(timeout=5)
            except Exception:
                self.processes[run_id].kill()
            del self.processes[run_id]

        return True

    def get_logs(self, run_id: str, tail: int = 100) -> list[str]:
        """Get recent logs for a run."""
        logs = self.log_buffers.get(run_id, [])
        return logs[-tail:] if tail else logs

    def get_artifacts_dir(self, run_id: str) -> Path:
        """Get the artifacts directory for a run."""
        artifacts_dir = self.runs_dir / run_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir


# Global training manager instance
training_manager = TrainingManager()
