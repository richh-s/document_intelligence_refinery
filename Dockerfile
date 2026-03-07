# Use a slim Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for PDF tools and local models)
RUN apt-get update && apt-get install -ig \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY pyproject.toml .
RUN pip install .

# Copy source code and artifacts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env.example .env
COPY .refinery/ ./.refinery/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command: Run the demo
CMD ["python", "scripts/demo_full_pipeline.py"]
