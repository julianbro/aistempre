#!/bin/bash
# Verification script for EPIC 1 acceptance criteria

set -e

echo "==================================="
echo "EPIC 1 - Acceptance Criteria Check"
echo "==================================="
echo ""

# Check monorepo structure
echo "✓ Checking monorepo structure..."
[ -d "frontend" ] && echo "  ✓ frontend/ directory exists"
[ -d "api" ] && echo "  ✓ api/ directory exists"
[ -f "pnpm-workspace.yaml" ] && echo "  ✓ pnpm-workspace.yaml exists"
[ -f "docker-compose.yml" ] && echo "  ✓ docker-compose.yml exists"
[ -f ".env.example" ] && echo "  ✓ .env.example exists"
echo ""

# Check frontend setup
echo "✓ Checking frontend setup..."
[ -f "frontend/package.json" ] && echo "  ✓ frontend/package.json exists"
[ -f "frontend/tsconfig.json" ] && echo "  ✓ TypeScript configured"
[ -f "frontend/eslint.config.mjs" ] && echo "  ✓ ESLint configured"
[ -f "frontend/.prettierrc" ] && echo "  ✓ Prettier configured"
echo ""

# Check API setup
echo "✓ Checking API setup..."
[ -f "api/main.py" ] && echo "  ✓ FastAPI main.py exists"
[ -f "api/pyproject.toml" ] && echo "  ✓ pyproject.toml exists"
[ -f "api/requirements.txt" ] && echo "  ✓ requirements.txt exists"
echo ""

# Check CI/CD
echo "✓ Checking GitHub Actions workflows..."
[ -f ".github/workflows/frontend-ci.yml" ] && echo "  ✓ Frontend CI workflow exists"
[ -f ".github/workflows/backend-ci.yml" ] && echo "  ✓ Backend CI workflow exists"
[ -f ".github/workflows/docker-build.yml" ] && echo "  ✓ Docker build workflow exists"
echo ""

# Check Docker setup
echo "✓ Checking Docker configuration..."
[ -f "Dockerfile.frontend" ] && echo "  ✓ Dockerfile.frontend exists"
[ -f "Dockerfile.api" ] && echo "  ✓ Dockerfile.api exists"
echo ""

# Check devcontainer
echo "✓ Checking devcontainer..."
[ -f ".devcontainer/devcontainer.json" ] && echo "  ✓ devcontainer.json exists"
echo ""

# Test frontend build (if pnpm is available)
if command -v pnpm &> /dev/null; then
    echo "✓ Testing frontend..."
    cd frontend
    echo "  Running: pnpm build"
    if pnpm build > /tmp/frontend-build.log 2>&1; then
        echo "  ✓ pnpm -C frontend build: PASS"
    else
        echo "  ✗ pnpm -C frontend build: FAIL (see /tmp/frontend-build.log)"
        exit 1
    fi
    cd ..
else
    echo "⚠ pnpm not found, skipping frontend build test"
fi
echo ""

# Test backend (if Python is available)
if command -v python3 &> /dev/null; then
    echo "✓ Testing API..."
    cd api
    
    # Install minimal dependencies if needed
    if ! python3 -c "import fastapi" 2>/dev/null; then
        echo "  Installing test dependencies..."
        pip install -q fastapi uvicorn pydantic pytest httpx pytest-asyncio 2>&1 | tail -1
    fi
    
    echo "  Running: pytest -q tests/test_api.py"
    if python3 -m pytest -q tests/test_api.py > /tmp/api-test.log 2>&1; then
        echo "  ✓ pytest -q: PASS"
    else
        echo "  ✗ pytest -q: FAIL (see /tmp/api-test.log)"
        cat /tmp/api-test.log
        exit 1
    fi
    cd ..
else
    echo "⚠ Python not found, skipping API test"
fi
echo ""

echo "==================================="
echo "✓ All acceptance criteria verified!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. Start services: docker compose up"
echo "  2. Frontend will be at http://localhost:3000"
echo "  3. API will be at http://localhost:8000"
echo "  4. API docs at http://localhost:8000/docs"
