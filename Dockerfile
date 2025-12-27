# Build Stage: Install dependencies and build wheels
FROM python:3.11-slim AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install specific Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Build wheels for dependencies to speed up final stage and keep it slim
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# Final Stage: Runtime environment
FROM python:3.11-slim AS final

# Build-time tags
LABEL maintainer="Antigravity"
LABEL version="1.0"
LABEL description="Premier League Prediction API"

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    PORT=5000

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels requirements.txt

# Create application directories and add non-root user for security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser && \
    mkdir -p /app/models /app/data /app/src /app/logs && \
    chown -R appuser:appgroup /app

# Copy application files (leveraging .dockerignore)
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/ ./data/

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE ${PORT}

# Healthcheck to monitor the API status
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/models || exit 1

# Launch the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
