FROM python:3.13-slim 

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*


# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./


RUN uv pip install --system \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu


# Install dependencies (skipping local project installation to avoid crashes)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY src ./src

CMD ["uv", "run", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]