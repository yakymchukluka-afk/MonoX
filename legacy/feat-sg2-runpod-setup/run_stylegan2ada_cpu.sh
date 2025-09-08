#!/bin/bash

# StyleGAN2-ADA RunPod Wrapper Script (CPU Version)
# This script runs StyleGAN2-ADA commands as the ubuntu user with CPU-only mode

set -e

echo "🚀 StyleGAN2-ADA RunPod Wrapper (CPU Mode)"
echo "=========================================="

# Function to run commands as ubuntu user
run_as_ubuntu() {
    echo "Running as ubuntu user: $*"
    sudo -u ubuntu "$@"
}

# Check if we're in the right directory
if [ ! -f "/workspace/test_pytorch_compatibility.py" ]; then
    echo "❌ Error: Please run this script from /workspace directory"
    echo "Run: cd /workspace && ./run_stylegan2ada_cpu.sh"
    exit 1
fi

# Parse command line arguments
case "${1:-help}" in
    "test")
        echo "🧪 Running PyTorch compatibility test..."
        run_as_ubuntu python3 test_pytorch_compatibility.py
        ;;
    "setup")
        echo "🔧 Running training setup test..."
        run_as_ubuntu python3 train_stylegan2ada_simple.py
        ;;
    "train-cpu")
        echo "🚀 Starting StyleGAN2-ADA training (CPU mode)..."
        if [ -z "$2" ]; then
            echo "❌ Error: Please provide dataset path"
            echo "Usage: ./run_stylegan2ada_cpu.sh train-cpu /path/to/dataset"
            exit 1
        fi
        DATASET_PATH="$2"
        echo "📊 Dataset: $DATASET_PATH"
        echo "📁 Output: /workspace/output"
        echo "🖥️  Mode: CPU only (no GPU required)"
        
        # Create output directory
        mkdir -p /workspace/output
        
        # Run training with CPU-only mode
        run_as_ubuntu bash -c "cd /workspace/train/runpod-hf/vendor/stylegan2ada && CUDA_VISIBLE_DEVICES='' python3 train.py \
            --outdir=/workspace/output \
            --data=$DATASET_PATH \
            --gpus=0 \
            --batch=1 \
            --gamma=10 \
            --mirror=1 \
            --kimg=1 \
            --snap=1 \
            --metrics=none \
            --cfg=auto"
        ;;
    "help"|*)
        echo "Usage: ./run_stylegan2ada_cpu.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  test                    - Run PyTorch compatibility test"
        echo "  setup                   - Run training setup test"
        echo "  train-cpu <dataset>     - Run CPU-only training test"
        echo "  help                    - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_stylegan2ada_cpu.sh test"
        echo "  ./run_stylegan2ada_cpu.sh setup"
        echo "  ./run_stylegan2ada_cpu.sh train-cpu /workspace/data"
        echo ""
        echo "Note: This is for testing without GPU. For actual training, use the GPU version."
        ;;
esac