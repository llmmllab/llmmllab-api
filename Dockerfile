# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# llmmllab-api — FastAPI inference service (no CUDA, no llama.cpp)
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Pull the uv binary from the official image (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    SHARED_VENV="/opt/venv/shared" \
    UV_PROJECT_ENVIRONMENT="/opt/venv/shared" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PORT=9999 \
    IMAGE_DIR="/root/images" \
    IMAGE_GENERATION_ENABLED="true" \
    MAX_IMAGE_SIZE="2048" \
    IMAGE_RETENTION_HOURS="24"

# Runtime-only apt deps + Playwright browser
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create the shared venv
RUN uv venv --python 3.12 ${SHARED_VENV}

# Copy dep manifests first (cached layer)
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install Playwright Chromium browser (headless_shell for WebReader)
RUN ${SHARED_VENV}/bin/playwright install chromium

# Copy application source
COPY . .

# Trim venv weight
RUN find ${SHARED_VENV} -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find ${SHARED_VENV} -type d \( -name "tests" -o -name "test" \) -prune -exec rm -rf {} + \
    && find ${SHARED_VENV} \( -name "*.pyc" -o -name "*.pyo" \) -delete \
    && rm -rf /tmp/* /var/tmp/*

EXPOSE 9999

CMD ["uv", "run", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9999", "--timeout-graceful-shutdown", "30"]
