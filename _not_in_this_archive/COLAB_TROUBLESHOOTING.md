# 🚨 COLAB TROUBLESHOOTING GUIDE

## Current Error:
```
Cannot find primary config 'config_safe'. Check that it's in your config search path.
```

## ✅ IMMEDIATE SOLUTIONS:

### Option 1: Use Default Config (Recommended)
```bash
python3 -m src.infra.launch \
  exp_suffix=monox \
  dataset.path=/content/drive/MyDrive/MonoX/dataset \
  training.total_kimg=3000 \
  num_gpus=1
```

### Option 2: Force Pull Latest Changes
```bash
cd /content/MonoX
git pull origin main
```

### Option 3: Clone Fresh Repository
```bash
cd /content
rm -rf MonoX
git clone https://github.com/yakymchukluka-afk/MonoX.git
cd MonoX
```

### Option 4: Verify Config Files Exist
```bash
ls -la /content/MonoX/configs/
```

## 🔧 DEBUG STEPS:

1. **Check current directory:**
   ```bash
   pwd
   ls -la configs/
   ```

2. **Verify you're in the right path:**
   ```bash
   cd /content/MonoX
   ```

3. **Check git status:**
   ```bash
   git status
   git log --oneline -3
   ```

## 🎯 WORKING COMMAND:
Once in `/content/MonoX/`, this should work:
```bash
python3 -m src.infra.launch \
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

## 🚨 If Still Failing:
The issue is with Hydra version compatibility. StyleGAN-V uses ancient hydra-core==1.0.7 which doesn't support modern features.