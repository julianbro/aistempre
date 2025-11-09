"""Tests for training runs API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import RunConfig

client = TestClient(app)


def test_create_run():
    """Test creating a new training run."""
    config = {
        "data_source": "./data/example.csv",
        "timeframes": ["1m", "15m", "1h"],
        "variant": "base",
        "max_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.0002,
    }
    response = client.post("/runs", json={"config": config})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["status"] == "pending" or data["status"] == "running"


def test_list_runs():
    """Test listing all training runs."""
    response = client.get("/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert "total_count" in data
    assert isinstance(data["runs"], list)


def test_get_run():
    """Test getting a specific run."""
    # First create a run
    config = {
        "data_source": "./data/example.csv",
        "timeframes": ["1m"],
        "variant": "base",
        "max_epochs": 5,
    }
    create_response = client.post("/runs", json={"config": config})
    run_id = create_response.json()["id"]

    # Get the run
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == run_id
    assert "status" in data
    assert "config" in data


def test_get_nonexistent_run():
    """Test getting a run that doesn't exist."""
    response = client.get("/runs/nonexistent-id")
    assert response.status_code == 404


def test_cancel_run():
    """Test cancelling a running training run."""
    # Create a run first
    config = {
        "data_source": "./data/example.csv",
        "timeframes": ["1m"],
        "variant": "base",
        "max_epochs": 50,  # Long enough to cancel
    }
    create_response = client.post("/runs", json={"config": config})
    run_id = create_response.json()["id"]

    # Wait a bit for it to start
    import time
    time.sleep(0.5)

    # Cancel the run
    response = client.post(f"/runs/{run_id}/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"
    assert data["run_id"] == run_id


def test_list_artifacts():
    """Test listing artifacts for a run."""
    # Create a run first
    config = {
        "data_source": "./data/example.csv",
        "timeframes": ["1m"],
        "variant": "base",
        "max_epochs": 5,
    }
    create_response = client.post("/runs", json={"config": config})
    run_id = create_response.json()["id"]

    # List artifacts
    response = client.get(f"/runs/{run_id}/artifacts")
    assert response.status_code == 200
    data = response.json()
    assert "artifacts" in data
    assert "total_count" in data


def test_download_artifact_not_found():
    """Test downloading an artifact that doesn't exist."""
    # Create a run first
    config = {
        "data_source": "./data/example.csv",
        "timeframes": ["1m"],
        "variant": "base",
        "max_epochs": 5,
    }
    create_response = client.post("/runs", json={"config": config})
    run_id = create_response.json()["id"]

    # Try to download non-existent artifact
    response = client.get(f"/runs/{run_id}/artifacts/nonexistent.txt")
    assert response.status_code == 404
