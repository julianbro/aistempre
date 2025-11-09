# Quick Start Guide

This guide helps you get the Neurotrader monorepo up and running quickly.

## Prerequisites

Install the required tools:
- **Node.js 20+**: https://nodejs.org/
- **Python 3.11+**: https://www.python.org/downloads/
- **Docker**: https://docs.docker.com/get-docker/ (optional, for containerized deployment)

## Option 1: Docker Compose (Recommended)

The fastest way to get started:

```bash
# Start both services
docker compose up

# Or in detached mode
docker compose up -d
```

Services will be available at:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

To stop:
```bash
docker compose down
```

## Option 2: Local Development

### 1. Install Dependencies

```bash
# Install Node.js global package manager
npm install -g pnpm

# Install workspace dependencies
pnpm install

# Install Python backend dependencies
cd api
pip install -r requirements-dev.txt
cd ..
```

### 2. Set Up Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env if needed (optional for development)
```

### 3. Start Development Servers

#### Option A: Both Services (Concurrent)
```bash
pnpm dev
```

#### Option B: Separate Terminals
```bash
# Terminal 1: Frontend
pnpm dev:frontend

# Terminal 2: API
pnpm dev:api
```

### 4. Access the Applications

- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative API Docs: http://localhost:8000/redoc

## Verify Installation

Run the verification script to check your setup:

```bash
./verify-setup.sh
```

This will:
- ✓ Check all required files and directories
- ✓ Build the frontend
- ✓ Run backend tests

## Common Tasks

### Linting & Formatting

```bash
# Lint everything
pnpm lint

# Format everything
pnpm format

# Type check everything
pnpm type-check
```

### Testing

```bash
# Run all tests
pnpm test

# Frontend tests only
pnpm test:frontend

# Backend tests only
pnpm test:api
```

### Building for Production

```bash
# Build frontend
pnpm build

# Or directly
pnpm -C frontend build
```

## Troubleshooting

### Port Already in Use

If you see errors about ports 3000 or 8000 being in use:

```bash
# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Docker Issues

```bash
# Clean up Docker resources
docker compose down -v
docker system prune

# Rebuild images
docker compose build --no-cache
```

### Module Not Found Errors

```bash
# Reinstall Node dependencies
rm -rf node_modules frontend/node_modules
pnpm install

# Reinstall Python dependencies
cd api
pip install --force-reinstall -r requirements-dev.txt
```

### PNPM Not Found

```bash
npm install -g pnpm
```

## Next Steps

1. **Read the main README.md** for detailed documentation
2. **Explore the API** at http://localhost:8000/docs
3. **Check the frontend** at http://localhost:3000
4. **Review the code structure** in `frontend/` and `api/` directories
5. **Run the tests** with `pnpm test`

## Development Workflow

1. Create a new branch for your feature
2. Make changes to frontend or API
3. Run linting and tests: `pnpm lint && pnpm test`
4. Commit and push your changes
5. Open a pull request

GitHub Actions will automatically:
- ✓ Lint your code
- ✓ Run type checks
- ✓ Execute tests
- ✓ Build Docker images

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review [API documentation](http://localhost:8000/docs) when running
- Open an issue on GitHub for bugs or questions

---

Happy coding! 🚀
