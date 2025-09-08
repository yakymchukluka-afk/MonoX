#!/bin/bash

# StyleGAN2-ADA RunPod Wrapper Script
# This script runs StyleGAN2-ADA commands as the ubuntu user

set -e

echo "🚀 StyleGAN2-ADA RunPod Wrapper"
echo "================================"

# Function to run commands as ubuntu user
run_as_ubuntu() {
    echo "Running as ubuntu user: $*"
    sudo -u ubuntu "$@"
}

# Check if we're in the right directory
if [ ! -f "/workspace/test_pytorch_compatibility.py" ]; then
    echo "❌ Error: Please run this script from /workspace directory"
    echo "Run: cd /workspace && ./run_stylegan2ada.sh"
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
    "train")
        echo "🚀 Starting StyleGAN2-ADA training..."
        if [ -z "$2" ]; then
            echo "❌ Error: Please provide dataset path"
            echo "Usage: ./run_stylegan2ada.sh train /path/to/dataset.zip"
            exit 1
        fi
        DATASET_PATH="$2"
        echo "📊 Dataset: $DATASET_PATH"
        echo "📁 Output: /workspace/output"
        
        # Create output directory
        mkdir -p /workspace/output
        
        # Run training
        run_as_ubuntu bash -c "cd /workspace/train/runpod-hf/vendor/stylegan2ada && python3 train.py \
            --outdir=/workspace/output \
            --data=$DATASET_PATH \
            --gpus=1 \
            --batch=8 \
            --gamma=10 \
            --mirror=1 \
            --kimg=25000 \
            --snap=50 \
            --metrics=fid50k_full \
            --resume=ffhq1024 \
            --cfg=auto \
            --aug=ada"
        ;;
    "train-small")
        echo "🧪 Running small test training..."
        if [ -z "$2" ]; then
            echo "❌ Error: Please provide dataset path"
            echo "Usage: ./run_stylegan2ada.sh train-small /path/to/dataset"
            exit 1
        fi
        DATASET_PATH="$2"
        echo "📊 Dataset: $DATASET_PATH"
        echo "📁 Output: /workspace/output"
        
        # Create output directory
        mkdir -p /workspace/output
        
        # Run small test training
        run_as_ubuntu bash -c "cd /workspace/train/runpod-hf/vendor/stylegan2ada && python3 train.py \
            --outdir=/workspace/output \
            --data=$DATASET_PATH \
            --gpus=1 \
            --batch=2 \
            --gamma=10 \
            --mirror=1 \
            --kimg=1 \
            --snap=1 \
            --metrics=none"
        ;;
    "help"|*)
        echo "Usage: ./run_stylegan2ada.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  test                    - Run PyTorch compatibility test"
        echo "  setup                   - Run training setup test"
        echo "  train <dataset_path>    - Start full training with your dataset"
        echo "  train-small <dataset>   - Run small test training"
        echo "  help                    - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_stylegan2ada.sh test"
        echo "  ./run_stylegan2ada.sh setup"
        echo "  ./run_stylegan2ada.sh train /workspace/data/monox-dataset.zip"
        echo "  ./run_stylegan2ada.sh train-small /workspace/data"
        echo ""
        echo "Current directory: $(pwd)"
        echo "Available files:"
        ls -la /workspace/test_pytorch_compatibility.py /workspace/train_stylegan2ada_simple.py 2>/dev/null || echo "Files not found"
        ;;
esac