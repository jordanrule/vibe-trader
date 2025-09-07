# Use Python 3.11 slim image for Cloud Run
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for logs (will be mounted from GCS in cloud)
RUN mkdir -p /app/logs

# Set Python path
ENV PYTHONPATH=/app

# Set environment variables for Cloud Run
ENV CLOUD_MODE=true
ENV PORT=8080

# Expose port for Cloud Run
EXPOSE 8080

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the Cloud Run HTTP server
CMD ["python", "main_cloud.py"]
