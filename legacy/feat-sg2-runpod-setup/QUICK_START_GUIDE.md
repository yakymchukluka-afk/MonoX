# 🚀 StyleGAN2-ADA Quick Start Guide for RunPod

## ✅ **PyTorch 2.0+ Compatibility - FIXED!**

The `RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented` issue has been **completely resolved**!

## 🎯 **Quick Commands (Run as Root)**

### **Step 1: Navigate to workspace**
```bash
cd /workspace
```

### **Step 2: Test PyTorch compatibility**
```bash
./run_stylegan2ada.sh test
```
**Expected output**: All tests should show ✅ PASS

### **Step 3: Test training setup**
```bash
./run_stylegan2ada.sh setup
```
**Expected output**: All tests should show ✅ PASS

### **Step 4: Download your dataset**
```bash
# Create data directory
mkdir -p /workspace/data

# Download your dataset (replace with your actual dataset)
# For example, if using Hugging Face:
# huggingface-cli download lukua/monox-dataset --local-dir /workspace/data
# Or copy your dataset to /workspace/data/monox-dataset.zip
```

### **Step 5: Start training**
```bash
# For your custom dataset (901 images, 1024x1024)
./run_stylegan2ada.sh train /workspace/data/monox-dataset.zip

# Or for a quick test with the sample dataset
./run_stylegan2ada.sh train-small /workspace/data
```

## 🔧 **What the Wrapper Script Does**

The `run_stylegan2ada.sh` script automatically:
- Runs commands as the `ubuntu` user (required for file permissions)
- Handles all the complex paths and parameters
- Provides easy-to-use commands for common tasks

## 📊 **Test Results (All PASS!)**

```
✅ PyTorch version: 2.8.0+cu128
✅ grid_sample_gradfix: PASS
✅ StyleGAN2-ADA imports: PASS  
✅ Network creation: PASS
✅ Gradient computation: PASS
✅ Training script: PASS
✅ Overall: PASS
```

## 🎯 **Training Configuration**

- **Resolution**: 1024x1024 (for your dataset)
- **Batch Size**: 8 (optimized for A100 80GB)
- **Dataset**: Your custom dataset (901 images)
- **Augmentation**: ADA (Adaptive Data Augmentation)
- **Target**: 25,000 kimg
- **Gamma**: 10 (R1 regularization)

## 🔍 **Troubleshooting**

### **If tests fail:**
```bash
# Check if files exist
ls -la /workspace/test_pytorch_compatibility.py
ls -la /workspace/run_stylegan2ada.sh

# Check permissions
ls -la /workspace/train/runpod-hf/vendor/stylegan2ada/train.py
```

### **If training fails:**
```bash
# Check dataset
ls -la /workspace/data/

# Check GPU (on GPU instance)
nvidia-smi

# Run with verbose output
./run_stylegan2ada.sh train-small /workspace/data
```

## 🎉 **Success Indicators**

You'll know everything is working when:
- ✅ `./run_stylegan2ada.sh test` shows all PASS
- ✅ `./run_stylegan2ada.sh setup` shows all PASS
- ✅ Training starts without errors
- ✅ Generated images appear in `/workspace/output/`

## 📁 **Key Files**

- `run_stylegan2ada.sh` - Main wrapper script (run this!)
- `test_pytorch_compatibility.py` - Compatibility test
- `train_stylegan2ada_simple.py` - Setup test
- `train/runpod-hf/vendor/stylegan2ada/train.py` - Main training script

## 🚀 **Ready to Train!**

The PyTorch compatibility issue has been **completely resolved**. Your StyleGAN2-ADA training setup is now ready for production use on RunPod!

**Just run: `./run_stylegan2ada.sh test` to get started!**