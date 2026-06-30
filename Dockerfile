# ─────────────────────────────────────────────────────────────────────────────
# Agentic Test Generator — FastAPI Inference Service
# ─────────────────────────────────────────────────────────────────────────────
# Build (Mac B, CPU-only):
#   docker build -t agentic-test-gen:latest .
#
# Run in MOCK mode (no model weights required — returns stub scripts):
#   docker run --rm -e MOCK_MODEL=true -p 8000:8000 agentic-test-gen:latest
#
# Run with real models (Mac A, MPS — mount local model dirs as volumes):
#   docker run --rm \
#     -v ~/Documents/Dissertation/agentic-test-gen/fine_tuning:/app/fine_tuning:ro \
#     -e WARMUP_MODEL=phi3/cypress \
#     -p 8000:8000 agentic-test-gen:latest
#
# NOTE: MPS (Apple Silicon GPU) is not available inside Docker containers.
#       For real inference, run the service natively on Mac A:
#         uvicorn api.app:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached separately from code)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/

# Default to mock mode so the container starts cleanly without model weights
ENV MOCK_MODEL=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check — waits up to 30s for startup
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
