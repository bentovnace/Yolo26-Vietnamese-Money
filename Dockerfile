FROM python:3.10-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# Install minimal OS dependencies for OpenCV & image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY app/ ./app/
COPY predict_cli.py .
COPY models/ ./models/

# Expose default HTTP port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI web application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
