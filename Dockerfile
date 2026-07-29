FROM python:3-slim

WORKDIR /app

# 2. Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 4. Copy dependency files
COPY pyproject.toml uv.lock ./

# 5. Build the .venv with standard dependencies first
RUN uv sync --frozen --no-dev --no-install-project

# 6. Force PyTorch CPU directly into the .venv created in the previous step
RUN uv pip install --python /app/.venv \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 7. Copy application source code
COPY src ./src

# 8. Run the app using the .venv
CMD ["uv", "run", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]