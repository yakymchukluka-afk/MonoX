#!/usr/bin/env bash
set -euxo pipefail

echo "📊 MonoX StyleGAN2-ADA Dataset Preparation"
echo "=========================================="

# Parse arguments
SRC="${1:-/workspace/data/monox-dataset}"
DEST="${2:-/workspace/data/monox-dataset-1024.zip}"
RES="${3:-1024x1024}"

echo "Source directory: $SRC"
echo "Output ZIP: $DEST"
echo "Target resolution: $RES"

# Check if source directory exists
if [ ! -d "$SRC" ]; then
    echo "❌ Error: Source directory '$SRC' does not exist"
    echo ""
    echo "Please provide a valid dataset directory:"
    echo "Usage: $0 <source_directory> <output_zip> <resolution>"
    echo ""
    echo "Example:"
    echo "  $0 /workspace/data/my-images /workspace/data/dataset.zip 1024x1024"
    echo ""
    echo "Available directories in /workspace/data/:"
    ls -la /workspace/data/ 2>/dev/null || echo "  (no data directory found)"
    exit 1
fi

# Check if StyleGAN2-ADA dataset tool exists
if [ ! -f "vendor/stylegan2ada/dataset_tool.py" ]; then
    echo "❌ Error: StyleGAN2-ADA dataset tool not found"
    echo "Please run bootstrap.sh first: bash scripts/runpod/bootstrap.sh"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$DEST")"

echo "🔄 Converting dataset to StyleGAN2-ADA format..."
echo "This may take a while depending on dataset size..."

# Run the dataset conversion
python vendor/stylegan2ada/dataset_tool.py \
  --source="$SRC" \
  --dest="$DEST" \
  --resolution="$RES" \
  --transform=center-crop \
  --workers=8

if [ $? -eq 0 ]; then
    echo "✅ Dataset conversion completed successfully!"
    echo "📁 Output file: $DEST"
    echo "📊 File size: $(du -h "$DEST" | cut -f1)"
else
    echo "❌ Dataset conversion failed"
    exit 1
fi