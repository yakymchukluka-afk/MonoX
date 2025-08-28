# 🚀 MonoX Colab Quick Start

## The Path Issue Fix

The error you saw happens because the scripts need to be in the right location. Here's the **correct setup**:

## ✅ Method 1: Single-File Setup (Recommended)

```python
# 1. Download and run the all-in-one setup
!wget https://raw.githubusercontent.com/your-repo/MonoX/main/setup_monox_colab.py
!python setup_monox_colab.py

# 2. Verify setup worked
!python /content/MonoX/verify_monox_setup.py

# 3. Start training
!python /content/MonoX/launch_training.py
```

## ✅ Method 2: Clone Repository First

```python
# 1. Clone the repository
!cd /content && git clone https://github.com/your-repo/MonoX.git

# 2. Run setup from correct location
!cd /content/MonoX && python setup_monox_colab.py

# 3. Verify setup
!python /content/MonoX/verify_monox_setup.py

# 4. Start training  
!python /content/MonoX/launch_training.py
```

## 🔍 Quick Verification Commands

```python
# Check if setup completed correctly
!ls -la /content/MonoX/

# Check if StyleGAN-V was cloned
!ls -la /content/MonoX/.external/stylegan-v/

# Check GPU status
!nvidia-smi

# Verify Python environment
!python /content/MonoX/verify_monox_setup.py
```

## 📊 Monitor Training

```python
# Check GPU usage (should be 80-95% when training)
!nvidia-smi

# Check training logs  
!tail -f /content/MonoX/results/logs/*.log

# Check generated files
!ls -la /content/MonoX/results/previews/
!ls -la /content/MonoX/results/checkpoints/
```

## 🎯 Expected Results

After running the setup, you should see:

✅ **Directory structure created** at `/content/MonoX/`  
✅ **StyleGAN-V cloned** with submodules  
✅ **Dependencies installed** (PyTorch, Hydra, etc.)  
✅ **Sample dataset created** for testing  
✅ **Configuration files** ready  
✅ **GPU detected** and ready  

When training starts:

✅ **GPU utilization 80-95%**  
✅ **Memory usage >4GB**  
✅ **Log files** being written  
✅ **Preview images** generated every 50 kimg  
✅ **Checkpoints** saved every 250 kimg  

## 🔧 Troubleshooting

### Problem: "No such file or directory"
**Solution:** Make sure you're using the full paths:
```python
!python /content/MonoX/setup_monox_colab.py     # Not just setup_monox_colab.py
!python /content/MonoX/verify_monox_setup.py    # Not just verify_monox_setup.py
```

### Problem: "Module not found"
**Solution:** Run the verification script:
```python
!python /content/MonoX/verify_monox_setup.py
```

### Problem: "GPU not being used"
**Solution:** Check GPU status and restart if needed:
```python
!nvidia-smi
# If 0% usage, restart Colab runtime and re-run setup
```

## 📱 Complete Colab Cell Sequence

Copy these cells in order:

### Cell 1: Setup
```python
!wget https://raw.githubusercontent.com/your-repo/MonoX/main/setup_monox_colab.py
!python setup_monox_colab.py
```

### Cell 2: Verify
```python
!python /content/MonoX/verify_monox_setup.py
```

### Cell 3: Train
```python
!python /content/MonoX/launch_training.py
```

### Cell 4: Monitor (run in parallel)
```python
!watch -n 10 nvidia-smi
```

### Cell 5: Check Results
```python
# View latest generated images
import matplotlib.pyplot as plt
from PIL import Image
import glob

preview_files = glob.glob('/content/MonoX/results/previews/*.png')
if preview_files:
    latest = max(preview_files, key=os.path.getctime)
    img = Image.open(latest)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f'Latest: {os.path.basename(latest)}')
    plt.axis('off')
    plt.show()
```

That's it! This should resolve the path issues and get your training working.