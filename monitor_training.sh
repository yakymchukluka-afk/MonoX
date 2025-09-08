#!/bin/bash
# Training Monitoring Script
# Monitors training progress and shows real-time stats

OUTPUT_DIR="${1:-/workspace/output}"

echo "📊 Monitoring StyleGAN2-ADA training..."
echo "Output directory: $OUTPUT_DIR"

# Function to show training progress
show_progress() {
    echo "=========================================="
    echo "🕐 $(date)"
    echo "=========================================="
    
    # Check if training is running
    if pgrep -f "train.py" > /dev/null; then
        echo "✅ Training is running"
    else
        echo "❌ Training is not running"
    fi
    
    # Show GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "🔍 GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while read line; do
            echo "   GPU: $line"
        done
    fi
    
    # Show latest checkpoint
    if [ -d "$OUTPUT_DIR" ]; then
        LATEST_PKL=$(find "$OUTPUT_DIR" -name "*.pkl" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_PKL" ]; then
            echo ""
            echo "📁 Latest checkpoint: $(basename "$LATEST_PKL")"
            echo "   Size: $(du -h "$LATEST_PKL" | cut -f1)"
            echo "   Modified: $(stat -c %y "$LATEST_PKL")"
        fi
        
        # Show latest samples
        LATEST_SAMPLE=$(find "$OUTPUT_DIR" -name "fakes*.png" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_SAMPLE" ]; then
            echo ""
            echo "🖼️ Latest sample: $(basename "$LATEST_SAMPLE")"
        fi
    fi
    
    echo ""
}

# Monitor loop
while true; do
    clear
    show_progress
    echo "Press Ctrl+C to stop monitoring"
    sleep 30
done