#!/bin/bash
set -e

# StyleGAN2-ADA Training Script for RunPod
# Usage: train.sh <outdir> <dataset_zip> [gpus] [config]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <outdir> <dataset_zip> [gpus] [config]"
    echo "Example: $0 /workspace/out/sg2 /workspace/datasets/mydataset.zip 8 auto"
    exit 1
fi

OUTDIR="$1"
DATASET_ZIP="$2"
GPUS="${3:-8}"
CONFIG="${4:-auto}"

echo "🚀 MonoX StyleGAN2-ADA Training Script"
echo "======================================"
echo "Output directory: $OUTDIR"
echo "Dataset: $DATASET_ZIP"
echo "GPUs: $GPUS"
echo "Config: $CONFIG"

# Validate inputs
if [ ! -f "$DATASET_ZIP" ]; then
    echo "❌ Error: Dataset ZIP file '$DATASET_ZIP' does not exist"
    exit 1
fi

# Create output directory
mkdir -p "$OUTDIR"

# Check GPU availability
echo "🔍 Checking GPU availability..."
source /workspace/venv/bin/activate
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available')
    exit(1)
"

# Set up training environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)))
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Create tmux session for training
echo "🎬 Starting training in tmux session 'monox'..."
tmux new-session -d -s monox "bash -lc 'source /workspace/venv/bin/activate && cd /workspace/vendor/stylegan2ada && python train.py --outdir=$OUTDIR --data=$DATASET_ZIP --gpus=$GPUS --cfg=$CONFIG --aug=ada --metrics=fid50k_full --snap=10 --resume=ffhq1024'"

# Wait a moment for tmux to start
sleep 2

# Check if tmux session is running
if tmux has-session -t monox 2>/dev/null; then
    echo "✅ Training started successfully in tmux session 'monox'"
    echo ""
    echo "📊 Monitoring commands:"
    echo "   tmux attach -t monox    # Attach to training session"
    echo "   tmux list-sessions      # List all tmux sessions"
    echo "   tmux kill-session -t monox  # Stop training"
    echo ""
    echo "📁 Output files will be saved to: $OUTDIR"
    echo "📝 Training logs: $OUTDIR/log.txt"
    echo "🖼️  Sample images: $OUTDIR/fakes*.png"
    echo "💾 Model checkpoints: $OUTDIR/network-snapshot-*.pkl"
    echo ""
    echo "🔍 To monitor training progress:"
    echo "   tail -f $OUTDIR/log.txt"
    echo ""
    echo "🎯 Training is now running in the background!"
    echo "   Use 'tmux attach -t monox' to see the training output"
else
    echo "❌ Failed to start training session"
    exit 1
fi