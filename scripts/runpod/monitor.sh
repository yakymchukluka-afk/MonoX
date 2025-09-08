#!/bin/bash
# Training monitoring script for RunPod

OUTDIR="${1:-/workspace/out/sg2}"

echo "📊 MonoX StyleGAN2-ADA Training Monitor"
echo "======================================"
echo "Monitoring directory: $OUTDIR"
echo ""

# Check if training is running
if tmux has-session -t monox 2>/dev/null; then
    echo "✅ Training session 'monox' is running"
else
    echo "❌ No training session found"
    echo "   Start training with: bash scripts/runpod/train.sh"
    exit 1
fi

# Check GPU status
echo "🔥 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""

# Check training progress
if [ -f "$OUTDIR/log.txt" ]; then
    echo "📈 Training Progress:"
    echo "-------------------"
    
    # Get latest kimg
    LATEST_KIMG=$(grep -o "kimg [0-9]*\.[0-9]*" "$OUTDIR/log.txt" | tail -1 | grep -o "[0-9]*\.[0-9]*")
    if [ ! -z "$LATEST_KIMG" ]; then
        echo "Current kimg: $LATEST_KIMG"
    fi
    
    # Get latest FID if available
    LATEST_FID=$(grep -o "fid50k_full [0-9]*\.[0-9]*" "$OUTDIR/log.txt" | tail -1 | grep -o "[0-9]*\.[0-9]*")
    if [ ! -z "$LATEST_FID" ]; then
        echo "Latest FID: $LATEST_FID"
    fi
    
    # Get latest loss
    LATEST_LOSS=$(grep -o "G_loss [0-9]*\.[0-9]*" "$OUTDIR/log.txt" | tail -1 | grep -o "[0-9]*\.[0-9]*")
    if [ ! -z "$LATEST_LOSS" ]; then
        echo "Latest G_loss: $LATEST_LOSS"
    fi
    
    echo ""
    echo "📝 Recent log entries:"
    tail -5 "$OUTDIR/log.txt"
else
    echo "⚠️  No training log found at $OUTDIR/log.txt"
fi

echo ""

# Check output files
echo "📁 Output Files:"
if [ -d "$OUTDIR" ]; then
    echo "Checkpoints: $(ls -1 "$OUTDIR"/network-snapshot-*.pkl 2>/dev/null | wc -l) files"
    echo "Samples: $(ls -1 "$OUTDIR"/fakes*.png 2>/dev/null | wc -l) files"
    echo "Metrics: $(ls -1 "$OUTDIR"/metric-*.jsonl 2>/dev/null | wc -l) files"
    
    # Show latest checkpoint
    LATEST_CHECKPOINT=$(ls -t "$OUTDIR"/network-snapshot-*.pkl 2>/dev/null | head -1)
    if [ ! -z "$LATEST_CHECKPOINT" ]; then
        echo "Latest checkpoint: $(basename "$LATEST_CHECKPOINT")"
    fi
else
    echo "❌ Output directory not found: $OUTDIR"
fi

echo ""
echo "🎯 Monitoring Commands:"
echo "   tmux attach -t monox     # Attach to training session"
echo "   tail -f $OUTDIR/log.txt  # Follow training log"
echo "   watch -n 30 '$0'         # Auto-refresh every 30s"