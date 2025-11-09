from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import datasets

app = FastAPI(
    title="AI Trading Platform API",
    description="Multi-input, multi-horizon, probabilistic Transformer for financial time-series",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router)


@app.get("/")
def read_root():
    return {"message": "AI Trading Platform API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
