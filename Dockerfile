# syntax=docker/dockerfile:1

#this part helps to get the uv binary form which is later used in the build stage
ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.8.4

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_NO_PROGRESS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /bin/uv

COPY pyproject.toml uv.lock ./

# Cache uv downloads across rebuilds (BuildKit). Requires DOCKER_BUILDKIT=1.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# lean runtime image 
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    # Quiet ultralytics / torch in containers
    YOLO_VERBOSE=False

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Virtualenv with CPU torch + FastAPI stack
COPY --from=builder /app/.venv /app/.venv

# Application code + YOLO weights (src/app/model/*.pt) + configs
COPY src ./src

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

# Same entrypoint as the project README
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
