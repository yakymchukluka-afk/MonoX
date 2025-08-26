#!/bin/bash
# MonoX Training Smoke Test
# This script runs a minimal test to verify the training pipeline works

set -e

echo "🚀 Running MonoX Training Smoke Test..."
echo "================================================"

# Check if we're in the right directory
if [[ ! -f "train.py" ]]; then
    echo "❌ Error: train.py not found. Run this script from the MonoX root directory."
    exit 1
fi

# Check if configs exist
if [[ ! -d "configs" ]]; then
    echo "❌ Error: configs directory not found."
    exit 1
fi

# Run smoke test with local launcher (skips actual training)
echo "🔧 Testing config loading and validation..."
# Using preferred dataset.name=ffs format (dataset=ffs also works)
python3 train.py -cp configs -cn config \
    dataset.name=ffs \
    training.steps=10 \
    training.batch=2 \
    training.num_workers=0 \
    training.fp16=false \
    launcher=local

echo ""
echo "✅ Smoke test completed successfully!"
echo "💡 To run actual training, use: launcher=stylegan"
echo ""
echo "Example full training command:"
echo "python train.py dataset.path=/path/to/data dataset=ffs training.total_kimg=3000"