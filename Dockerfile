FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install StyleGAN-V in development mode
RUN pip install -e .

# Create non-root user for security
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Expose port
EXPOSE 7860

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]