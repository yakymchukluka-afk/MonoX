---
title: MonoX
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# MonoX

A participative art project powered by StyleGAN-V for generating dynamic visual content.

## Quick Start

MonoX uses **`train.py`** as the single entrypoint for all training operations. It loads Hydra configurations from `configs/` and delegates to the StyleGAN-V submodule.

### Installation

```bash
git clone https://github.com/yakymchukluka-afk/MonoX.git
cd MonoX
pip install -r requirements.txt
```

### Basic Usage

```bash
# Basic training with default settings
python train.py dataset.path=/path/to/your/dataset

# Training with custom parameters
python train.py dataset.path=/path/to/data dataset=ffs training.total_kimg=3000

# Quick smoke test (config validation only)
python train.py dataset=ffs training.steps=10 launcher=local
```

### Colab Instructions

For Google Colab environments, use this minimal smoke test command:

```python
!python train.py \
    dataset=ffs \
    training.steps=10 \
    training.batch=2 \
    training.num_workers=0 \
    training.fp16=false \
    launcher=local
```

For actual training in Colab:

```python
# Set up your dataset path
!python train.py \
    dataset.path=/content/drive/MyDrive/MonoX/dataset \
    dataset=ffs \
    training.total_kimg=3000 \
    training.log_dir=/content/drive/MyDrive/MonoX/logs \
    training.preview_dir=/content/drive/MyDrive/MonoX/previews
```

### Local Testing

Run the smoke test to verify everything is working:

```bash
# Using the test script
./scripts/test_train.sh

# Or manually
python train.py dataset=ffs training.steps=10 launcher=local
```

## Configuration

### Main Parameters

- `dataset.path`: Path to your training dataset
- `dataset`: Dataset configuration (ffs, ucf101, mead, etc.)
- `training.total_kimg`: Total training duration in thousands of images
- `training.steps`: Override for quick testing (use instead of total_kimg)
- `training.batch`: Batch size override
- `training.num_workers`: DataLoader workers (set to 0 for Colab)
- `training.fp16`: Enable/disable mixed precision training
- `launcher`: Training mode (`stylegan` for full training, `local` for testing)

### Config Files

- `configs/config.yaml`: Main configuration file
- `configs/training/base.yaml`: Training-specific settings
- Configuration follows Hydra conventions with support for overrides

### Example Commands

```bash
# Full training run
python train.py dataset.path=/data/videos dataset=ffs training.total_kimg=5000

# Quick test with small batch
python train.py dataset=ffs training.steps=50 training.batch=4 launcher=local

# Resume from checkpoint
python train.py dataset.path=/data/videos training.resume=/path/to/checkpoint.pkl

# Custom output directories
python train.py \
    dataset.path=/data/videos \
    training.log_dir=./logs \
    training.preview_dir=./previews \
    training.checkpoint_dir=./checkpoints
```

## Architecture

- **`train.py`**: Single entrypoint with Hydra configuration loading
- **`configs/`**: Hydra configuration files for datasets, training, sampling
- **`.external/stylegan-v/`**: StyleGAN-V submodule (automatically managed)
- **`src/infra/`**: Legacy infrastructure (deprecated in favor of train.py)

## Troubleshooting

### Common Issues

1. **Missing dataset path**: Set `dataset.path` or `DATASET_DIR` environment variable
2. **Hydra interpolation errors**: Ensure all config files have required fields
3. **Import errors**: Run from MonoX root directory
4. **Memory issues**: Reduce `training.batch` or set `training.fp16=true`

### Environment Variables

- `DATASET_DIR`: Default dataset path
- `LOGS_DIR`: Default logs directory
- `PREVIEWS_DIR`: Default previews directory
- `CKPT_DIR`: Default checkpoints directory

## Development

The project uses:
- **Hydra** for configuration management
- **StyleGAN-V** as the core training engine
- **OmegaConf** for flexible config overrides

See `configs/config.yaml` for the full configuration schema.
