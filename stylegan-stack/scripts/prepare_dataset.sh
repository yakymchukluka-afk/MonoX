#!/usr/bin/env bash
set -euo pipefail

# Prepare images into StyleGAN2/3 dataset zip
# Usage: ./scripts/prepare_dataset.sh /path/to/images /workspace/stylegan-stack/data/artpieces.zip

IMAGES_DIR=${1:-}
OUT_ZIP=${2:-/workspace/stylegan-stack/data/dataset.zip}

if [[ -z "$IMAGES_DIR" ]]; then
  echo "Usage: $0 <images_dir> [out_zip]" >&2
  exit 1
fi

ROOT_DIR="/workspace/stylegan-stack"
REPO_SG2="$ROOT_DIR/repos/stylegan2-ada-pytorch"

if [[ ! -f "$REPO_SG2/dataset_tool.py" ]]; then
  echo "dataset_tool.py not found in stylegan2-ada-pytorch repo" >&2
  exit 2
fi

python "$REPO_SG2/dataset_tool.py" --source "$IMAGES_DIR" --dest "$OUT_ZIP" --resolution 512x512
echo "Dataset prepared at: $OUT_ZIP"
