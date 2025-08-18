#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/train_stylegan3.sh <dataset_zip> <outdir> <cfg> <gpus> <resolution> <batch>

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <dataset_zip> <outdir> <cfg: stylegan3-t|stylegan3-r> <gpus> <resolution> <batch>" >&2
  exit 1
fi

DATASET_ZIP="$1"
OUTDIR="$2"
CFG="$3"
GPUS="$4"
RES="$5"
BATCH="$6"

ROOT_DIR="/workspace/stylegan-stack"
REPO_SG3="$ROOT_DIR/repos/stylegan3"

if [[ ! -f "$REPO_SG3/train.py" ]]; then
  echo "train.py not found in stylegan3 repo" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

python "$REPO_SG3/train.py" \
  --outdir="$OUTDIR" \
  --cfg="$CFG" \
  --data="$DATASET_ZIP" \
  --gpus="$GPUS" \
  --batch="$BATCH" \
  --gamma=8.2 \
  --mirror=1 \
  --snap=10 \
  --kimg=25000 \
  --res="$RES"

