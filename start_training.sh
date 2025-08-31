#!/bin/bash
# MonoX Fresh Training Startup Script
# Use this script to start fresh training in Hugging Face Spaces

echo "🎨 MonoX Fresh Training - Hugging Face Spaces"
echo "🔗 Dataset: lukua/monox-dataset (868 images)"
echo "🎯 Target: lukua/monox model repo"
echo "=" * 70

# Setup environment
export PATH="/home/ubuntu/.local/bin:$PATH"
export PYTHONPATH="/workspace/.external/stylegan-v:/workspace"
export HF_TOKEN="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"
export PYTHONUNBUFFERED=1

# Change to workspace
cd /workspace

echo "🚀 Starting MonoX fresh training..."
echo "📝 This will:"
echo "  • Train StyleGAN-V on 1024x1024 monotype dataset"
echo "  • Save checkpoints every 5 epochs"
echo "  • Upload all outputs to lukua/monox model repo"
echo "  • Provide real-time monitoring"

echo ""
echo "🔥 Launching training..."
python3 final_monox_training.py

echo ""
echo "✅ Training script completed!"
echo "📁 Check lukua/monox model repo for outputs"