# 🚀 MonoX Nuclear GPU Setup - Complete Fresh Session Command

## 🔥 COMPLETE FRESH COLAB SETUP (Copy-Paste Ready)

**Use this in a completely fresh Colab session after restart:**

```python
# ===== STEP 1: Mount Drive and Setup =====
from google.colab import drive
drive.mount('/content/drive')

# ===== STEP 2: Nuclear GPU Environment Setup =====
import os
import torch
import subprocess

# Force GPU environment at system level
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['FORCE_CUDA'] = '1'
os.environ['USE_CUDA'] = '1'

# Verify GPU immediately
print("🔥 NUCLEAR GPU VERIFICATION:")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # Force GPU allocation
    torch.cuda.set_device(0)
    test = torch.randn(1000, 1000, device='cuda')
    result = torch.mm(test, test)
    print(f"✅ GPU test: {result.device}, Memory: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    del test, result
    torch.cuda.empty_cache()
else:
    print("❌ CUDA NOT AVAILABLE - CHECK RUNTIME!")

# ===== STEP 3: Clone and Setup MonoX =====
!rm -rf /content/MonoX  # Clean slate
!git clone https://github.com/yakymchukluka-afk/MonoX
%cd /content/MonoX

# Initialize submodules
!git submodule update --init --recursive

# Pull latest nuclear patches
!git pull origin main
!git submodule update --remote

# ===== STEP 4: Install Dependencies =====
!python colab_install_v2.py

# ===== STEP 5: Apply Nuclear GPU Patches =====
!python patch_stylegan_gpu.py

# ===== STEP 6: NUCLEAR GPU TRAINING =====
!python train_super_gpu_forced.py \
  exp_suffix=nuclear_fresh_session \
  dataset.path=/content/drive/MyDrive/MonoX/dataset \
  dataset.resolution=256 \
  training.total_kimg=50 \
  training.snapshot_kimg=10 \
  visualizer.save_every_kimg=5 \
  num_gpus=1
```

## 🎯 Alternative: Step-by-Step Debugging

**If the above doesn't work, use this step-by-step approach:**

```python
# Step 1: Basic setup
from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/MonoX
!git clone https://github.com/yakymchukluka-afk/MonoX
%cd /content/MonoX
!git submodule update --init --recursive
!git pull origin main && git submodule update --remote
```

```python
# Step 2: GPU verification
import torch
import os

# Force GPU environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'

print("🔍 GPU Status:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    torch.cuda.set_device(0)
    test = torch.randn(2000, 2000, device='cuda')
    _ = torch.mm(test, test)
    print(f"GPU memory: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    del test
    torch.cuda.empty_cache()
```

```python
# Step 3: Install and patch
!python colab_install_v2.py
!python patch_stylegan_gpu.py
```

```python
# Step 4: Nuclear training
!python train_super_gpu_forced.py \
  exp_suffix=nuclear_debug \
  dataset.path=/content/drive/MyDrive/MonoX/dataset \
  dataset.resolution=256 \
  training.total_kimg=25 \
  training.snapshot_kimg=5 \
  visualizer.save_every_kimg=2 \
  num_gpus=1
```

## 🔍 Debugging Commands

**Check if training is progressing:**

```python
# Check experiment directories
!ls -la /content/MonoX/experiments/

# Check if any files were created
!find /content/MonoX/experiments/ -name "*" -type f 2>/dev/null | head -20

# Check logs
!ls -la /content/MonoX/logs/
!tail -20 /content/MonoX/logs/train_super_gpu_*.log

# GPU status during training
!nvidia-smi

# Process check
!ps aux | grep python
```

## 🎯 Expected Success Signs

**When GPU forcing works, you should see:**

```
🔥 MONOX TRAIN: Starting subprocess for rank 0
🔥 MONOX TRAIN: Set CUDA device to 0  
🔥 MONOX TRAIN: GPU 0 working, memory: XXX MB
🔥 MONOX: Forcing GPU device for rank 0
🔥 MONOX: GPU verification successful on cuda:0
🔥 MONOX: GPU memory allocated: XXX MB

Loading training set...
Num videos: XXXX
Constructing networks...
Setting up augmentation...
```

**And GPU RAM should jump to 2-8GB in Colab's resource monitor!**

## 🚨 If Still Not Working

**Try this minimal test:**

```python
# Direct StyleGAN-V test
%cd /content/MonoX/.external/stylegan-v

# Set environment
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'

# Direct train call with minimal parameters
!python src/train.py \
  --outdir=/content/MonoX/test_output \
  --data=/content/drive/MyDrive/MonoX/dataset \
  --gpus=1 \
  --batch_size=4 \
  --snap=1 \
  --kimg=5
```

This bypasses all MonoX wrapper scripts and calls StyleGAN-V directly.