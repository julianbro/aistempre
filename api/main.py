"""
FastAPI application for neurotrader API.

This provides REST endpoints for:
- Model training
- Predictions
- Backtesting
- Data ingestion
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Neurotrader API",
    description="Multi-input, multi-horizon, probabilistic Transformer for financial time-series",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Neurotrader API", "version": "0.1.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.get("/api/models")
async def list_models():
    """List available models."""
    return {"models": []}


@app.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        "status": "running",
        "models_loaded": 0,
        "backend": "neurotrader",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
