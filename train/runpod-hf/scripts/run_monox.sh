#!/bin/bash
set -e

# Map RunPod secret to HF token
if [ -n "${RUNPOD_SECRET_HF_token:-}" ]; then
  export HF_TOKEN="${RUNPOD_SECRET_HF_token}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "[setup] Starting MonoX training setup..."

# Install non-Torch dependencies
echo "[setup] Installing dependencies..."
pip install -r requirements.txt

# Smart Torch resolver
TORCH_VERSION="${TORCH_VERSION:-2.2.2}"
TVISION_VERSION="${TVISION_VERSION:-0.17.2}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu121}"

ensure_torch() {
  python - <<'PY'
import sys, json
try:
    import torch
    info = {
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(info))
    sys.exit(0 if info["cuda_available"] else 2)
except Exception as e:
    print('{"error":"%s"}' % e)
    sys.exit(1)
PY
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[setup] Installing PyTorch ${TORCH_VERSION}/${TVISION_VERSION} (${TORCH_CUDA_TAG}) ..."
    pip install --upgrade --force-reinstall \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
      "torch==${TORCH_VERSION}" "torchvision==${TVISION_VERSION}"
    python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU OK:", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 3)
PY
  else
    echo "[setup] Using preinstalled Torch (CUDA OK)."
  fi
}
ensure_torch

# Download dataset
echo "[setup] Downloading dataset..."
python scripts/get_dataset.py

# Create output directories
mkdir -p /workspace/out/{checkpoints,samples,logs}

# Launch training and uploader in tmux sessions
echo "[setup] Launching training and uploader..."
tmux new -s monox -d "python monox/train.py --config configs/monox-1024.yaml"
tmux new -s hubpush -d "python scripts/push_to_hub.py"
echo "Launched training (tmux: monox) and uploader (tmux: hubpush)"
