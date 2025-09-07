#!/bin/bash

# StyleGAN2-ADA Training Script for RunPod
# This script sets up and runs StyleGAN2-ADA training with PyTorch compatibility fixes

set -e

# Configuration
DATASET_PATH="/workspace/data/monox-dataset.zip"
OUTPUT_DIR="/workspace/output"
RESOLUTION=1024
BATCH_SIZE=8
GAMMA=10
MIRROR=1
KIMG=25000

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to StyleGAN2-ADA directory
cd /workspace/train/runpod-hf/vendor/stylegan2ada

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Run training with PyTorch compatibility
python train.py \
    --outdir="$OUTPUT_DIR" \
    --data="$DATASET_PATH" \
    --gpus=1 \
    --batch="$BATCH_SIZE" \
    --gamma="$GAMMA" \
    --mirror="$MIRROR" \
    --kimg="$KIMG" \
    --snap=50 \
    --metrics=fid50k_full \
    --resume=ffhq1024 \
    --cfg=auto \
    --aug=ada \
    --p=0.2 \
    --target=0.6 \
    --augpipe=blit,geom,color,filter,noise,cutout

echo "Training completed successfully!"