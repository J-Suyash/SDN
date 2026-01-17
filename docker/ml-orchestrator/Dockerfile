# ML Orchestrator Container
# Python environment for inference and policy engine

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create directories
RUN mkdir -p /app/orchestrator /app/ml/models /app/collector /app/data

# Default command (overridden in docker-compose)
CMD ["python", "-u", "/app/orchestrator/orchestrator.py"]
