"""Training runs API routes."""

import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from app.models import (
    TrainingRun,
    CreateRunRequest,
    CreateRunResponse,
    RunListResponse,
    StreamEvent,
    ArtifactInfo,
    ArtifactListResponse,
)
from app.training_manager import training_manager

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=CreateRunResponse, status_code=201)
def create_run(request: CreateRunRequest):
    """
    Create a new training run.

    - **config**: Training configuration (Hydra overrides)
    - **name**: Optional run name/ID
    """
    run = training_manager.create_run(request.config, request.name)
    # Auto-start the run
    training_manager.start_run(run.id)
    return CreateRunResponse(
        id=run.id, status=run.status, created_at=run.created_at
    )


@router.get("", response_model=RunListResponse)
def list_runs():
    """
    List all training runs.

    Returns a list of all training runs with their current status and metrics.
    """
    runs = training_manager.list_runs()
    return RunListResponse(runs=runs, total_count=len(runs))


@router.get("/{run_id}", response_model=TrainingRun)
def get_run(run_id: str):
    """
    Get details of a specific training run.

    Returns the run's status, configuration, and current metrics snapshot.
    """
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str):
    """
    Cancel a running training run.

    Stops the training process and marks the run as cancelled.
    """
    success = training_manager.cancel_run(run_id)
    if not success:
        raise HTTPException(
            status_code=400, detail="Cannot cancel run (not running or not found)"
        )
    return {"status": "cancelled", "run_id": run_id}


@router.websocket("/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for streaming live training metrics and logs.

    Emits events with:
    - epoch, step, losses, metrics
    - ETA, GPU memory, GPU utilization
    - Latest checkpoints
    - Console logs
    """
    await websocket.accept()

    run = training_manager.get_run(run_id)
    if not run:
        await websocket.close(code=1008, reason="Run not found")
        return

    try:
        last_log_count = 0
        while True:
            # Get current run state
            run = training_manager.get_run(run_id)
            if not run:
                break

            # Send metrics event
            if run.metrics:
                event = StreamEvent(
                    event_type="metrics",
                    timestamp=datetime.utcnow(),
                    data=run.metrics.model_dump(),
                )
                await websocket.send_json(event.model_dump(mode="json"))

            # Send status event
            status_event = StreamEvent(
                event_type="status",
                timestamp=datetime.utcnow(),
                data={"status": run.status.value, "run_id": run.id},
            )
            await websocket.send_json(status_event.model_dump(mode="json"))

            # Send new logs
            logs = training_manager.get_logs(run_id)
            if len(logs) > last_log_count:
                new_logs = logs[last_log_count:]
                for log in new_logs:
                    log_event = StreamEvent(
                        event_type="log",
                        timestamp=datetime.utcnow(),
                        data={"message": log},
                    )
                    await websocket.send_json(log_event.model_dump(mode="json"))
                last_log_count = len(logs)

            # Send checkpoint events
            if run.checkpoints:
                checkpoint_event = StreamEvent(
                    event_type="checkpoint",
                    timestamp=datetime.utcnow(),
                    data={
                        "checkpoints": [
                            cp.model_dump(mode="json") for cp in run.checkpoints
                        ]
                    },
                )
                await websocket.send_json(checkpoint_event.model_dump(mode="json"))

            # Break if run is completed/failed/cancelled
            if run.status in ["completed", "failed", "cancelled"]:
                await websocket.send_json(
                    {
                        "event_type": "status",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"status": run.status.value, "final": True},
                    }
                )
                break

            # Wait before next update
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@router.get("/{run_id}/artifacts", response_model=ArtifactListResponse)
def list_artifacts(run_id: str):
    """
    List all artifacts for a training run.

    Returns configs, checkpoints, scalers, metrics JSON, and predictions.
    """
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts_dir = training_manager.get_artifacts_dir(run_id)
    artifacts = []

    # Scan artifacts directory
    if artifacts_dir.exists():
        for file_path in artifacts_dir.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                artifact_type = _infer_artifact_type(file_path.name)
                artifacts.append(
                    ArtifactInfo(
                        name=file_path.relative_to(artifacts_dir).as_posix(),
                        size_bytes=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                        artifact_type=artifact_type,
                    )
                )

    return ArtifactListResponse(artifacts=artifacts, total_count=len(artifacts))


@router.get("/{run_id}/artifacts/{artifact_path:path}")
async def download_artifact(run_id: str, artifact_path: str):
    """
    Download a specific artifact file.

    Streams the file for download.
    """
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts_dir = training_manager.get_artifacts_dir(run_id)
    file_path = artifacts_dir / artifact_path

    # Security: ensure file is within artifacts directory
    try:
        file_path = file_path.resolve()
        artifacts_dir = artifacts_dir.resolve()
        if not str(file_path).startswith(str(artifacts_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )


def _infer_artifact_type(filename: str) -> str:
    """Infer artifact type from filename."""
    filename_lower = filename.lower()
    if filename_lower.endswith((".ckpt", ".pth", ".pt")):
        return "checkpoint"
    elif filename_lower.endswith((".yaml", ".yml", ".json")) and "config" in filename_lower:
        return "config"
    elif "scaler" in filename_lower or "calibrator" in filename_lower:
        return "scaler"
    elif filename_lower.endswith(".json") and "metrics" in filename_lower:
        return "metrics"
    elif filename_lower.endswith((".parquet", ".csv")) and "pred" in filename_lower:
        return "predictions"
    else:
        return "other"


import asyncio
