#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/infer_stylegan3.sh <network_pkl> <seed> <outdir>

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <network_pkl> <seed> <outdir>" >&2
  exit 1
fi

NETWORK_PKL="$1"
SEED="$2"
OUTDIR="$3"

ROOT_DIR="/workspace/stylegan-stack"
REPO_SG3="$ROOT_DIR/repos/stylegan3"

if [[ ! -f "$REPO_SG3/gen_images.py" ]]; then
  echo "gen_images.py not found in stylegan3 repo" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

python "$REPO_SG3/gen_images.py" \
  --outdir="$OUTDIR" \
  --seeds="$SEED" \
  --trunc=1.0 \
  --network="$NETWORK_PKL"

