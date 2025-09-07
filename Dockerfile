# Use Python 3.11 slim image for Cloud Functions
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
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

# Expose port for Cloud Functions (optional, Cloud Functions handles this)
EXPOSE 8080

# Run the main application
CMD ["python", "main.py"]
