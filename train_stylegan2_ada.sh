#!/bin/bash
# StyleGAN2-ADA Training Script for RunPod
# Starts training with proper configuration

set -e

# Default parameters
DATASET_ZIP="${1:-/workspace/datasets/dataset.zip}"
OUTPUT_DIR="${2:-/workspace/output}"
RESOLUTION="${3:-1024}"
BATCH_SIZE="${4:-8}"
KIMG="${5:-25000}"

echo "🎯 Starting StyleGAN2-ADA training..."
echo "Dataset: $DATASET_ZIP"
echo "Output: $OUTPUT_DIR"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "Batch size: $BATCH_SIZE"
echo "Training kimg: $KIMG"

# Check if dataset exists
if [ ! -f "$DATASET_ZIP" ]; then
    echo "❌ Dataset not found: $DATASET_ZIP"
    echo "Please prepare your dataset first using prepare_dataset.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export CUDA_LAUNCH_BLOCKING=1

# Create torch extensions directory
mkdir -p /tmp/torch_extensions

# Activate virtual environment
source /workspace/venv/bin/activate

# Start training
echo "🚀 Launching training..."
cd /workspace/stylegan2-ada

python train.py \
    --outdir="$OUTPUT_DIR" \
    --data="$DATASET_ZIP" \
    --gpus=1 \
    --batch="$BATCH_SIZE" \
    --kimg="$KIMG" \
    --snap=2500 \
    --metrics=none \
    --resume=latest

echo "✅ Training completed!"