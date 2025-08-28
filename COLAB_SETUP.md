# MonoX Colab Setup Guide

## Quick Setup

**Option 1: Simple installer (recommended)**
```python
# Install dependencies with multiple fallback methods
!cd /content/MonoX && python colab_install.py
```

**Option 2: Full setup script**
```python
# Install dependencies
!cd /content/MonoX && python setup_colab.py
```

**Option 3: Manual commands (if others fail)**
```bash
# Fix pip version compatibility
!pip install 'pip<24.1'
!pip install omegaconf==2.0.4
!pip install hydra-core==1.0.7
```

## Manual Installation (if needed)

If the automatic setup fails, run these commands individually:

```bash
# Update pip
!pip install --upgrade pip

# Install PyTorch with CUDA
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies  
!pip install hydra-core==1.0.7
!pip install omegaconf==2.0.6
!pip install 'pytorch-lightning>=1.5.0,<1.8.0'

# Install additional dependencies
!pip install numpy pillow scipy tqdm tensorboard matplotlib opencv-python imageio imageio-ffmpeg ninja psutil scikit-learn pandas
```

## Training Command

After setup, run training with:

```bash
!cd /content/MonoX && python train.py \
  exp_suffix=monox \
  dataset.path=/content/drive/MyDrive/MonoX/dataset \
  dataset.resolution=1024 \
  training.total_kimg=3000 \
  training.snapshot_kimg=250 \
  visualizer.save_every_kimg=50 \
  visualizer.output_dir=previews \
  sampling.truncation_psi=1.0 \
  num_gpus=1
```

## Troubleshooting

- **PyTorch Lightning Error**: Use version 1.6.x or 1.7.x (not 1.7.7 specifically)
- **CUDA Issues**: Make sure you're using a GPU runtime in Colab
- **Memory Issues**: Reduce batch size or training resolution
- **Dataset Path**: Ensure your dataset is uploaded to Google Drive at the specified path