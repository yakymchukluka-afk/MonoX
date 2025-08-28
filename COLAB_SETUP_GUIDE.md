# MonoX + StyleGAN-V Colab Setup Guide

## 🚀 Clean Setup for Google Colab

This guide provides a **working, tested solution** for running MonoX with StyleGAN-V training in Google Colab.

### ✅ What This Fixes

- ❌ **StyleGAN-V directory not found** → ✅ Proper git clone with submodules  
- ❌ **GPU idle (0% usage)** → ✅ Real training with GPU monitoring  
- ❌ **No snapshots/logs produced** → ✅ Verified output generation  
- ❌ **Hydra config conflicts** → ✅ Updated configs for Hydra 1.3.x  
- ❌ **PYTHONPATH issues** → ✅ Proper environment setup  
- ❌ **Virtualenv assumptions** → ✅ System-wide Python usage

---

## 📋 Prerequisites

1. **Google Colab with GPU** (T4, L4, V100, A100)
2. **Google Drive** mounted for persistent storage (optional but recommended)

---

## 🛠️ Step-by-Step Setup

### Step 1: Environment Setup (Run Once)

```python
# Download and run the environment setup
!wget https://raw.githubusercontent.com/your-repo/MonoX/main/colab_environment_setup.py
!python colab_environment_setup.py
```

**What this does:**
- ✅ Creates all necessary directories
- ✅ Installs PyTorch with CUDA support
- ✅ Installs all dependencies (Hydra, OmegaConf, etc.)
- ✅ Properly clones StyleGAN-V with submodules
- ✅ Sets up environment variables
- ✅ Creates sample dataset for testing
- ✅ Verifies GPU availability

### Step 2: Prepare Your Dataset

```python
# Option A: Upload your own dataset
from google.colab import files
import zipfile
import os

# Upload and extract your dataset
uploaded = files.upload()
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/MonoX/dataset/')

# Option B: Use sample dataset (already created by setup script)
!ls /content/MonoX/dataset/sample_images/
```

### Step 3: Run Debug Checklist

```python
# Verify everything is working before training
!python colab_debug_checklist.py
```

This will check:
- ✅ All directories exist
- ✅ Python packages installed
- ✅ GPU/CUDA available
- ✅ StyleGAN-V modules importable
- ✅ Config files valid
- ✅ Dataset ready

### Step 4: Start Training

```python
# Launch training with monitoring
!python colab_training_launcher.py
```

**Or with custom parameters:**
```python
# Custom training configuration
!python colab_training_launcher.py dataset.path=/custom/path training.total_kimg=1000 training.batch_size=8
```

### Step 5: Monitor Training (Optional)

```python
# In a separate cell, monitor GPU usage
!python colab_gpu_monitor.py --interval 10
```

---

## 📁 Directory Structure

After setup, your structure will be:

```
/content/MonoX/
├── .external/
│   └── stylegan-v/          # StyleGAN-V repository
├── configs/
│   ├── config_clean.yaml    # Main training config
│   ├── dataset/base.yaml    # Dataset config
│   ├── training/base.yaml   # Training config
│   └── visualizer/base.yaml # Visualization config
├── dataset/
│   └── sample_images/       # Your training images
├── results/
│   ├── logs/               # Training logs
│   ├── previews/           # Generated images
│   └── checkpoints/        # Model checkpoints
└── src/                    # MonoX source code
```

---

## 🔧 Configuration

### Basic Configuration (configs/config_clean.yaml)

```yaml
# Dataset
dataset:
  path: /content/MonoX/dataset
  resolution: 1024

# Training  
training:
  total_kimg: 3000      # Total training iterations
  snapshot_kimg: 250    # Save checkpoint every N kimg
  batch_size: 4         # Batch size (adjust for GPU memory)
  fp16: true           # Use mixed precision
  
# Output
visualizer:
  save_every_kimg: 50   # Generate previews every N kimg
```

### Command Line Overrides

```python
# Change resolution
!python colab_training_launcher.py dataset.resolution=512

# Quick test run
!python colab_training_launcher.py training.total_kimg=100 training.snapshot_kimg=25

# Custom batch size for different GPUs
!python colab_training_launcher.py training.batch_size=8  # For larger GPUs
!python colab_training_launcher.py training.batch_size=2  # For smaller GPUs

# Resume from checkpoint
!python colab_training_launcher.py training.resume=/content/MonoX/results/checkpoints/network-snapshot-000250.pkl
```

---

## 🔍 Troubleshooting

### Problem: "StyleGAN-V directory not found"

**Solution:**
```python
# Re-run the environment setup
!python colab_environment_setup.py
```

### Problem: "No GPU usage"

**Check GPU status:**
```python
!python colab_gpu_monitor.py --quick
```

**Force GPU usage:**
```python
# Ensure CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Problem: "Import errors"

**Run diagnostics:**
```python
!python colab_debug_checklist.py
```

### Problem: "Config not found"

**Check config files:**
```python
!ls -la /content/MonoX/configs/
!cat /content/MonoX/configs/config_clean.yaml
```

### Problem: "Training stops immediately"

**Check logs:**
```python
!tail -50 /content/MonoX/results/logs/*.log
```

---

## 📊 Monitoring Training Progress

### Real-time GPU Monitoring

```python
# Start monitoring in background
!python colab_gpu_monitor.py --interval 5 &
```

### Check Training Files

```python
# Check what files are being created
!python colab_gpu_monitor.py --check-files

# List recent outputs
!ls -lt /content/MonoX/results/logs/
!ls -lt /content/MonoX/results/previews/
!ls -lt /content/MonoX/results/checkpoints/
```

### View Generated Images

```python
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Show latest preview images
preview_files = glob.glob('/content/MonoX/results/previews/*.png')
if preview_files:
    latest_preview = max(preview_files, key=os.path.getctime)
    img = Image.open(latest_preview)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Latest Preview: {os.path.basename(latest_preview)}')
    plt.show()
```

---

## 💾 Saving to Google Drive

To persist your training across Colab sessions:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create symlink to Drive
!ln -sf /content/drive/MyDrive/MonoX_Results /content/MonoX/results
```

---

## ⚡ Performance Tips

### GPU Memory Optimization

```python
# For smaller GPUs (T4)
!python colab_training_launcher.py training.batch_size=2 training.fp16=true

# For larger GPUs (V100, A100)  
!python colab_training_launcher.py training.batch_size=8 training.fp16=true
```

### Speed vs Quality

```python
# Fast testing (lower quality)
!python colab_training_launcher.py dataset.resolution=512 training.total_kimg=500

# High quality (slower)
!python colab_training_launcher.py dataset.resolution=1024 training.total_kimg=5000
```

---

## 🎯 Success Indicators

You'll know training is working when:

✅ **GPU monitor shows 80-95% utilization**  
✅ **Memory usage > 4GB**  
✅ **Log files contain "kimg" progress updates**  
✅ **Preview images are generated every 50 kimg**  
✅ **Checkpoint files (.pkl) are saved every 250 kimg**

---

## 📞 Support

If you encounter issues:

1. **Run diagnostics first**: `!python colab_debug_checklist.py`
2. **Check GPU status**: `!python colab_gpu_monitor.py --quick`  
3. **Review logs**: `!tail -100 /content/MonoX/results/logs/*.log`
4. **Restart and re-run setup**: `!python colab_environment_setup.py`

---

## 🎉 Expected Results

After successful training, you should have:

- **Training logs** in `/content/MonoX/results/logs/`
- **Generated image previews** in `/content/MonoX/results/previews/`
- **Model checkpoints** in `/content/MonoX/results/checkpoints/`
- **High GPU utilization** (visible via monitoring)

The training will produce high-quality synthetic images based on your dataset!