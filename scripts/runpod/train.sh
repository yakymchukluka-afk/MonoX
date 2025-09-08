#!/usr/bin/env bash
set -euxo pipefail

echo "🎯 MonoX StyleGAN2-ADA Training"
echo "==============================="

# Parse arguments
OUTDIR="${1:-/workspace/out/sg2}"
DATA="${2:-/workspace/data/monox-dataset-1024.zip}"
BATCH="${3:-8}"

echo "Output directory: $OUTDIR"
echo "Dataset ZIP: $DATA"
echo "Batch size: $BATCH"

# Check if dataset exists
if [ ! -f "$DATA" ]; then
    echo "❌ Error: Dataset file '$DATA' does not exist"
    echo ""
    echo "Please prepare your dataset first:"
    echo "  bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /path/to/output.zip 1024x1024"
    exit 1
fi

# Check if StyleGAN2-ADA training script exists
if [ ! -f "vendor/stylegan2ada/train.py" ]; then
    echo "❌ Error: StyleGAN2-ADA training script not found"
    echo "Please run bootstrap.sh first: bash scripts/runpod/bootstrap.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTDIR"

echo "🚀 Starting StyleGAN2-ADA training..."
echo "📊 Training parameters:"
echo "  - Dataset: $DATA"
echo "  - Output: $OUTDIR"
echo "  - Batch size: $BATCH"
echo "  - Resolution: 1024x1024"
echo "  - Configuration: auto"
echo "  - GPUs: 1"
echo "  - Gamma: 10"
echo "  - Mirror: enabled"
echo "  - Snapshots: every 10 epochs"
echo ""

# Run training
python vendor/stylegan2ada/train.py \
  --outdir "$OUTDIR" \
  --data "$DATA" \
  --cfg auto \
  --gpus 1 \
  --batch "$BATCH" \
  --gamma 10 \
  --mirror 1 \
  --snap 10 \
  --metrics none

echo "🎉 Training completed!"
echo "📁 Results saved to: $OUTDIR"