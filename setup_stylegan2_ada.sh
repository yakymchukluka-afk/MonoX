#!/bin/bash
# StyleGAN2-ADA Training Setup for RunPod
# This script sets up the environment for StyleGAN2-ADA training

set -e

echo "🚀 Setting up StyleGAN2-ADA training environment..."

# Check if we're in RunPod
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "✅ Running in RunPod pod: $RUNPOD_POD_ID"
else
    echo "⚠️ Not running in RunPod environment"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "❌ nvidia-smi not found. GPU may not be available."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/checkpoints
mkdir -p /workspace/samples
mkdir -p /workspace/logs
mkdir -p /workspace/datasets
mkdir -p /workspace/output

# Install required system packages
echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-full

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# Install Python dependencies
echo "🔧 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow tqdm click requests

# Install StyleGAN2-ADA dependencies
cd /workspace/stylegan2-ada
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found, installing basic dependencies..."
    pip install torch torchvision numpy pillow tqdm click requests
fi

# Install additional dependencies for training
pip install wandb tensorboard

echo "✅ Setup complete!"
echo "🎯 Ready to start training"
echo ""
echo "Next steps:"
echo "1. Prepare your dataset: bash prepare_dataset.sh /path/to/images"
echo "2. Start training: bash train_stylegan2_ada.sh"