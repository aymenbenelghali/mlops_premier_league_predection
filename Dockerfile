FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Copy application code
COPY . /app

# Create models and data directories
RUN mkdir -p /app/models /app/data

# Expose port for web interface
EXPOSE 5000

# Environment variables
ENV DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    ROLLING_WINDOW=5 \
    SEASONS_BACK=8 \
    LEAGUE_CODE=E0 \
    PYTHONUNBUFFERED=1

# Run the FastAPI app with both models (XGBoost + LSTM)
CMD ["python", "app.py"]

