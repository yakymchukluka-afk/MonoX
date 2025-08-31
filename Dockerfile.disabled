FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port that the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]