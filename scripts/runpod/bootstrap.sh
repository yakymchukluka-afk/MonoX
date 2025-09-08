#!/usr/bin/env bash
set -euxo pipefail

echo "🚀 MonoX StyleGAN2-ADA RunPod Bootstrap Script"
echo "=============================================="

# System deps (no sudo needed in RunPod as we're already root)
echo "📦 Updating system packages..."
apt-get update -y
apt-get install -y git-lfs tmux jq curl

# Install Git LFS
echo "🔧 Setting up Git LFS..."
git lfs install

# Python deps
echo "🐍 Setting up Python environment..."
python -V || true
pip install --upgrade pip || true

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow tqdm click requests

# Make directories
echo "📁 Creating workspace directories..."
mkdir -p /workspace/{code,data,out}

# Ensure submodule is ready
echo "🔗 Initializing submodules..."
git submodule update --init --recursive

# Check if StyleGAN2-ADA is properly set up
if [ -f "vendor/stylegan2ada/train.py" ]; then
    echo "✅ StyleGAN2-ADA submodule ready"
else
    echo "❌ StyleGAN2-ADA submodule not found"
    exit 1
fi

echo "🎉 Bootstrap complete! Ready for training."
echo ""
echo "Next steps:"
echo "1. Prepare your dataset: bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /path/to/output.zip 1024x1024"
echo "2. Start training: bash scripts/runpod/train.sh /workspace/out/sg2 /path/to/dataset.zip 8"