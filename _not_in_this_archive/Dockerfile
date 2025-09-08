# Custom Dockerfile for MonoX Training - bypasses git config issues
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies without git config
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a dummy git config to avoid permission errors
RUN mkdir -p /root && echo "[user]" > /root/.gitconfig && echo "    name = HF User" >> /root/.gitconfig && echo "    email = user@huggingface.co" >> /root/.gitconfig

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/previews /app/checkpoints /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Expose port for Gradio
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]