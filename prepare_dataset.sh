#!/bin/bash
# Dataset Preparation Script for StyleGAN2-ADA
# Converts image dataset to StyleGAN2-ADA format

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_directory> [output_zip] [resolution]"
    echo "Example: $0 /workspace/my_images /workspace/dataset.zip 1024"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_ZIP="${2:-/workspace/datasets/dataset.zip}"
RESOLUTION="${3:-1024}"

echo "🖼️ Preparing dataset for StyleGAN2-ADA..."
echo "Input directory: $INPUT_DIR"
echo "Output ZIP: $OUTPUT_ZIP"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_ZIP")"

# Count images
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)
echo "📊 Found $IMAGE_COUNT images"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "❌ No images found in $INPUT_DIR"
    echo "Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    exit 1
fi

# Activate virtual environment
source /workspace/venv/bin/activate

# Use StyleGAN2-ADA dataset tool to create ZIP
echo "🔄 Converting dataset to StyleGAN2-ADA format..."
cd /workspace/stylegan2-ada

python dataset_tool.py \
    --source="$INPUT_DIR" \
    --dest="$OUTPUT_ZIP" \
    --width="$RESOLUTION" \
    --height="$RESOLUTION"

echo "✅ Dataset prepared successfully!"
echo "📦 Output: $OUTPUT_ZIP"
echo "📊 Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "🖼️ Images: $IMAGE_COUNT"

# Show file size
if [ -f "$OUTPUT_ZIP" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
    echo "💾 File size: $FILE_SIZE"
fi