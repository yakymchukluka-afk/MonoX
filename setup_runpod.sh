#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

cd /workspace

# Install system dependencies
apt-get update -y
apt-get install -y git-lfs tmux curl jq
git lfs install || true

# Check Python and GPU
python -V
nvidia-smi || true

# Clone/update MonoX repository
cd /workspace/code
if [ ! -d MonoX/.git ]; then
  git clone https://github.com/yakymchukluka-afk/MonoX.git
fi
cd MonoX
git fetch --all --prune

# Create the StyleGAN-V loss branch if it doesn't exist
if ! git show-ref --verify --quiet refs/heads/feat/styleganv-loss-default; then
  git checkout -b feat/styleganv-loss-default
else
  git checkout feat/styleganv-loss-default
fi

# Set up the RunPod training structure
mkdir -p train/runpod-hf/{vendor,configs,scripts}

# Install Python dependencies
cd train/runpod-hf
pip install -r requirements.txt
pip install "omegaconf>=2.3.0" "hydra-core>=1.3.2" "einops>=0.6.0"

# Download dataset
cd /workspace
[ -d /workspace/data/monox-dataset/.git ] \
  || git clone https://huggingface.co/datasets/lukua/monox-dataset /workspace/data/monox-dataset
git -C /workspace/data/monox-dataset lfs pull

echo "Setup complete!"