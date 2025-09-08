#!/bin/bash
set -e

# Dataset preparation script for StyleGAN2-ADA
# Usage: make_dataset_zip.sh <source_dir> <output_zip> [width] [height]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_dir> <output_zip> [width] [height]"
    echo "Example: $0 /path/to/images /workspace/datasets/mydataset.zip 1024 1024"
    exit 1
fi

SOURCE_DIR="$1"
OUTPUT_ZIP="$2"
WIDTH="${3:-1024}"
HEIGHT="${4:-1024}"

echo "📊 MonoX StyleGAN2-ADA Dataset Preparation"
echo "=========================================="
echo "Source directory: $SOURCE_DIR"
echo "Output ZIP: $OUTPUT_ZIP"
echo "Target resolution: ${WIDTH}x${HEIGHT}"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_ZIP")
mkdir -p "$OUTPUT_DIR"

# Count images in source directory
echo "🔍 Analyzing source dataset..."
IMAGE_COUNT=$(find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)
echo "Found $IMAGE_COUNT images"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "❌ Error: No images found in source directory"
    exit 1
fi

# Check if we have enough images for training
if [ "$IMAGE_COUNT" -lt 1000 ]; then
    echo "⚠️  Warning: Dataset has only $IMAGE_COUNT images"
    echo "   StyleGAN2-ADA works best with 1000+ images"
    echo "   Consider using transfer learning for small datasets"
fi

# Create dataset using StyleGAN2-ADA dataset_tool.py
echo "🔄 Creating dataset ZIP file..."
cd /workspace/vendor/stylegan2ada

# Activate virtual environment
source /workspace/venv/bin/activate

python dataset_tool.py \
    --source="$SOURCE_DIR" \
    --dest="$OUTPUT_ZIP" \
    --width="$WIDTH" \
    --height="$HEIGHT" \
    --transform=center-crop

# Verify the created dataset
if [ -f "$OUTPUT_ZIP" ]; then
    echo "✅ Dataset created successfully!"
    echo "📁 Output file: $OUTPUT_ZIP"
    echo "📊 File size: $(du -h "$OUTPUT_ZIP" | cut -f1)"
    
    # Test loading the dataset
    echo "🧪 Testing dataset loading..."
    source /workspace/venv/bin/activate
    python -c "
import zipfile
import json
import os

# Open the ZIP file
with zipfile.ZipFile('$OUTPUT_ZIP', 'r') as zf:
    # Check if dataset.json exists
    if 'dataset.json' in zf.namelist():
        with zf.open('dataset.json') as f:
            dataset_info = json.load(f)
            print(f'Dataset info: {dataset_info}')
    else:
        print('No dataset.json found')
    
    # Count images in ZIP
    image_files = [f for f in zf.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f'Images in ZIP: {len(image_files)}')
"
    
    echo ""
    echo "🎯 Dataset is ready for training!"
    echo "   Use: bash scripts/runpod/train.sh /workspace/out/sg2 $OUTPUT_ZIP 8"
else
    echo "❌ Error: Failed to create dataset ZIP file"
    exit 1
fi