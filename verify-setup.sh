#!/bin/bash
# Verification script for EPIC 1 acceptance criteria

set -e

echo "=== EPIC 1 Acceptance Criteria Verification ==="
echo ""

echo "1. Testing frontend build..."
cd frontend
pnpm build
echo "✅ Frontend build succeeded"
echo ""

echo "2. Testing backend tests..."
cd ../api
pytest -q
echo "✅ Backend tests passed"
echo ""

echo "=== All acceptance criteria verified! ==="
echo ""
echo "To verify Docker setup, run:"
echo "  docker compose up"
echo ""
echo "Expected results:"
echo "  - Frontend accessible at http://localhost:3000"
echo "  - API accessible at http://localhost:8000"
