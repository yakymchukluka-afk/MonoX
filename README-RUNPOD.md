# RunPod Training (StyleGAN2-ADA)

This guide will help you set up and run StyleGAN2-ADA training on RunPod using the clean MonoX repository.

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/yakymchukluka-afk/MonoX.git
cd MonoX
git checkout runpod/sg2-1024
git submodule update --init --recursive
```

### 2. Bootstrap Environment
```bash
bash scripts/runpod/bootstrap.sh
```

### 3. Prepare Dataset
```bash
# Create a directory with your images
mkdir -p /workspace/data/my-dataset
# Copy your images to /workspace/data/my-dataset/

# Convert to StyleGAN2-ADA format
bash scripts/runpod/make_dataset_zip.sh /workspace/data/my-dataset /workspace/data/dataset.zip 1024x1024
```

### 4. Start Training
```bash
# Option 1: Direct training
bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/data/dataset.zip 8

# Option 2: Training with tmux (recommended for long runs)
tmux new -s monox -d 'bash -lc "bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/data/dataset.zip 8 | tee -a /workspace/out/train.log"'
tmux attach -t monox
```

## 📋 Prerequisites

### RunPod Instance
- **GPU**: A100 (recommended) or V100
- **Image**: PyTorch with CUDA 12+
- **Storage**: At least 50GB free space
- **RAM**: 32GB+ recommended

### Dataset Requirements
- **Format**: PNG or JPG images
- **Resolution**: Any (will be resized to 1024x1024)
- **Organization**: All images in a single directory
- **Size**: Minimum 1000 images recommended

## 🔧 Scripts Overview

### `bootstrap.sh`
- Updates system packages
- Installs Git LFS, tmux, and other dependencies
- Sets up Python environment with PyTorch
- Initializes StyleGAN2-ADA submodule

### `make_dataset_zip.sh`
- Converts image dataset to StyleGAN2-ADA format
- Creates ZIP file with proper structure
- Handles resizing and center-cropping
- Usage: `bash scripts/runpod/make_dataset_zip.sh <source_dir> <output_zip> <resolution>`

### `train.sh`
- Runs StyleGAN2-ADA training
- Configures training parameters
- Handles output directory creation
- Usage: `bash scripts/runpod/train.sh <output_dir> <dataset_zip> <batch_size>`

## ⚙️ Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--outdir` | `/workspace/out/sg2` | Output directory for checkpoints and samples |
| `--data` | Dataset ZIP path | Path to prepared dataset |
| `--batch` | `8` | Batch size (adjust based on GPU memory) |
| `--gpus` | `1` | Number of GPUs |
| `--cfg` | `auto` | Configuration (auto, stylegan2, paper256, paper512, paper1024) |
| `--gamma` | `10` | R1 regularization weight |
| `--mirror` | `1` | Enable dataset mirroring |
| `--snap` | `10` | Snapshot interval (epochs) |
| `--metrics` | `none` | Metrics to compute |

## 📊 Monitoring Training

### Check Training Progress
```bash
# View training logs
tail -f /workspace/out/train.log

# Check output directory
ls -la /workspace/out/sg2/

# View latest samples
ls -la /workspace/out/sg2/00000-*/reals.png
```

### Tmux Commands
```bash
# Attach to training session
tmux attach -t monox

# Detach from session (Ctrl+B, then D)
# List sessions
tmux list-sessions

# Kill session
tmux kill-session -t monox
```

## 🛠️ Troubleshooting

### Common Issues

1. **"sudo: command not found"**
   - You're already root in RunPod, no sudo needed
   - The bootstrap script handles this automatically

2. **"tmux: command not found"**
   - Run `bash scripts/runpod/bootstrap.sh` first
   - Or install manually: `apt-get update && apt-get install -y tmux`

3. **"Source directory does not exist"**
   - Create your dataset directory first
   - Copy images to the directory
   - Use absolute paths

4. **CUDA out of memory**
   - Reduce batch size: `bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/data/dataset.zip 4`
   - Use smaller resolution: `bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /path/to/output.zip 512x512`

5. **Dataset conversion fails**
   - Check image formats (PNG/JPG supported)
   - Ensure sufficient disk space
   - Verify image files are not corrupted

### Performance Tips

- **Batch Size**: Start with 8, reduce if OOM, increase if you have more GPU memory
- **Resolution**: 1024x1024 is standard, 512x512 trains faster
- **Dataset Size**: 10k+ images recommended for good results
- **Training Time**: Expect 1-3 days for 1024x1024 on A100

## 📁 Output Structure

```
/workspace/out/sg2/
├── 00000-data-mirror-auto1-gamma10-kimg1-batch8/
│   ├── fakes000000.png          # Generated samples
│   ├── reals.png                # Real samples
│   ├── training_options.json    # Training configuration
│   ├── network-snapshot-000000.pkl  # Model checkpoints
│   └── log.txt                  # Training logs
└── ...
```

## 🔄 Resuming Training

To resume from a checkpoint:
```bash
python vendor/stylegan2ada/train.py \
  --outdir /workspace/out/sg2 \
  --data /workspace/data/dataset.zip \
  --resume /workspace/out/sg2/00000-*/network-snapshot-000000.pkl \
  --batch 8
```

## 📞 Support

- **Repository**: https://github.com/yakymchukluka-afk/MonoX
- **StyleGAN2-ADA**: https://github.com/NVlabs/stylegan2-ada-pytorch
- **RunPod Docs**: https://docs.runpod.io/

---

**Happy Training! 🎨**