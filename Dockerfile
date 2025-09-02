# MonoX Training - HF Spaces Compatible Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to avoid permission issues
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set working directory
WORKDIR /home/user/app

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

# Pre-configure Git with correct permissions as non-root user
# Multiple approaches to prevent git config errors
RUN git config --global user.name "lukua" && \
    git config --global user.email "lukua@users.noreply.huggingface.co" && \
    chmod 644 $HOME/.gitconfig 2>/dev/null || true

# Additional git config locations as backup
RUN mkdir -p /tmp && \
    echo "[user]" > /tmp/.gitconfig && \
    echo "    name = lukua" >> /tmp/.gitconfig && \
    echo "    email = lukua@users.noreply.huggingface.co" >> /tmp/.gitconfig && \
    chmod 666 /tmp/.gitconfig 2>/dev/null || true

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files with correct ownership
COPY --chown=user . /home/user/app

# Create necessary directories
RUN mkdir -p /home/user/app/previews /home/user/app/checkpoints /home/user/app/logs

# Expose port for Gradio
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]