# 🚨 MODULE PATH FIX FOR COLAB

## Current Error:
```
ModuleNotFoundError: No module named 'src'
```

## ✅ SOLUTIONS (Try in order):

### Solution 1: Ensure Correct Directory
```bash
cd /content/MonoX
pwd  # Should show /content/MonoX
ls   # Should show src/ directory
```

### Solution 2: Run with Explicit Python Path
```bash
cd /content/MonoX
PYTHONPATH=/content/MonoX python3 -m src.infra.launch \
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

### Solution 3: Run Script Directly
```bash
cd /content/MonoX
python3 src/infra/launch.py \
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

### Solution 4: Use train.py Instead
```bash
cd /content/MonoX
python3 train.py \
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

## 🔧 DEBUG STEPS:

1. **Verify directory structure:**
   ```bash
   cd /content/MonoX
   ls -la src/infra/
   ```

2. **Check Python can find the module:**
   ```bash
   cd /content/MonoX
   python3 -c "import src.infra.launch; print('Module found!')"
   ```

3. **Test with sys.path:**
   ```bash
   cd /content/MonoX
   python3 -c "import sys; sys.path.insert(0, '.'); import src.infra.launch"
   ```

## 🎯 RECOMMENDED APPROACH:

Use **Solution 2** (with PYTHONPATH) or **Solution 3** (direct script) as they're most reliable for Colab environments.