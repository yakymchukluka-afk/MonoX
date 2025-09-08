#!/bin/bash
set -e

echo "🚀 MonoX StyleGAN2-ADA RunPod Bootstrap Script"
echo "=============================================="

# Check if we're on RunPod
if [ -z "$RUNPOD_POD_ID" ]; then
    echo "⚠️  Warning: This script is designed for RunPod instances"
    echo "   RUNPOD_POD_ID not found, continuing anyway..."
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential dependencies
echo "🔧 Installing essential dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    tmux \
    htop \
    vim \
    python3-dev \
    python3-pip \
    python3-venv \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# Upgrade pip in virtual environment
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install StyleGAN2-ADA dependencies
echo "📚 Installing StyleGAN2-ADA dependencies..."
python3 -m pip install \
    click \
    requests \
    tqdm \
    pyspng \
    ninja \
    imageio-ffmpeg==0.4.3 \
    pillow \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    tensorboard

# Verify CUDA installation
echo "🔍 Verifying CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Create output directories
echo "📁 Creating output directories..."
mkdir -p /workspace/out/sg2
mkdir -p /workspace/datasets
mkdir -p /workspace/logs

# Set up Git LFS if not already installed
if ! command -v git-lfs &> /dev/null; then
    echo "📦 Installing Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install -y git-lfs
    git lfs install
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x /workspace/scripts/runpod/*.sh

echo "✅ Bootstrap completed successfully!"
echo ""
echo "Next steps:"
echo "1. Prepare your dataset: bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /path/to/output.zip 1024x1024"
echo "2. Start training: bash scripts/runpod/train.sh /workspace/out/sg2 /path/to/dataset.zip 8"
echo ""
echo "For monitoring: tmux attach -t monox"