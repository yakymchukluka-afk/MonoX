# StyleGAN2-ADA RunPod Setup Guide

This guide provides complete setup instructions for running StyleGAN2-ADA training on RunPod with PyTorch 2.0+ compatibility.

## 🚀 Quick Start

### 1. Run the Setup Script
```bash
python setup_runpod_stylegan2ada.py
```

### 2. Start Training
```bash
./train_stylegan2ada.sh
```

## 📋 Prerequisites

- RunPod instance with A100 80GB GPU (recommended)
- Ubuntu 22.04.3 LTS
- CUDA 12.8
- Python 3.11

## 🔧 What the Setup Does

### PyTorch Compatibility Fix
- **Issue**: `RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented`
- **Solution**: Added fallback mechanism in `grid_sample_gradfix.py`
- **Compatibility**: Works with both PyTorch 1.x and 2.x

### Key Changes Made
1. **Updated `_should_use_custom_op()`**: Tests operation availability before using custom ops
2. **Enhanced `_GridSample2dBackward`**: Uses `torch.autograd.grad` as fallback for newer PyTorch versions
3. **Maintained backward compatibility**: Still works with older PyTorch versions

### Files Modified
- `train/runpod-hf/vendor/stylegan2ada/torch_utils/ops/grid_sample_gradfix.py`

## 📁 Directory Structure

```
/workspace/
├── train/runpod-hf/
│   ├── vendor/stylegan2ada/          # StyleGAN2-ADA submodule (patched)
│   └── scripts/sg2/train.sh          # Training script
├── data/
│   └── monox-dataset.zip             # Custom dataset (901 images, 1024x1024)
├── output/                            # Training outputs
├── setup_runpod_stylegan2ada.py      # Setup script
├── test_pytorch_compatibility.py     # Compatibility test
└── train_stylegan2ada.sh             # Main training script
```

## 🎯 Training Configuration

- **Resolution**: 1024x1024
- **Batch Size**: 8
- **Gamma**: 10
- **Mirror**: Enabled
- **Kimg**: 25,000
- **Augmentation**: ADA (Adaptive Data Augmentation)

## 🧪 Testing Compatibility

Run the compatibility test to verify everything works:

```bash
python test_pytorch_compatibility.py
```

Expected output:
```
✓ PyTorch version: 2.0.0+cu117
✓ CUDA available: True
✓ grid_sample_gradfix is working correctly
✓ All StyleGAN2-ADA imports successful
✓ Network creation and forward pass successful
🎉 All tests passed!
```

## 📊 Dataset Information

- **Source**: Hugging Face `lukua/monox-dataset`
- **Images**: 901 images
- **Resolution**: 1024x1024 (resized from original)
- **Format**: PNG in ZIP file
- **Location**: `/workspace/data/monox-dataset.zip`

## 🚀 Training Commands

### Basic Training
```bash
./train_stylegan2ada.sh
```

### Custom Configuration
```bash
cd /workspace/train/runpod-hf/vendor/stylegan2ada
python train.py \
    --outdir=/workspace/output \
    --data=/workspace/data/monox-dataset.zip \
    --gpus=1 \
    --batch=8 \
    --gamma=10 \
    --mirror=1 \
    --kimg=25000 \
    --snap=50 \
    --metrics=fid50k_full \
    --resume=ffhq1024 \
    --cfg=auto \
    --aug=ada
```

## 📈 Monitoring Training

### View Training Logs
```bash
tail -f /workspace/output/training_log.txt
```

### Check GPU Usage
```bash
nvidia-smi
```

### Monitor Output Images
```bash
ls -la /workspace/output/
```

## 🔍 Troubleshooting

### Common Issues

1. **PyTorch Compatibility Error**
   ```bash
   python test_pytorch_compatibility.py
   ```
   If this fails, the patch wasn't applied correctly.

2. **CUDA Out of Memory**
   - Reduce batch size: `--batch=4`
   - Use gradient accumulation
   - Enable mixed precision training

3. **Dataset Not Found**
   ```bash
   ls -la /workspace/data/
   ```
   Ensure the dataset was downloaded correctly.

4. **Import Errors**
   ```bash
   cd /workspace/train/runpod-hf/vendor/stylegan2ada
   python -c "import torch_utils.ops.grid_sample_gradfix; print('Import successful')"
   ```

### Debug Mode
Run with verbose output:
```bash
cd /workspace/train/runpod-hf/vendor/stylegan2ada
python train.py --outdir=/workspace/output --data=/workspace/data/monox-dataset.zip --gpus=1 --batch=4 --verbose
```

## 📚 Technical Details

### PyTorch Compatibility Fix

The main issue was that `torch._C._jit_get_operation('aten::grid_sampler_2d_backward')` is not available in PyTorch 2.0+. The fix:

1. **Detection**: Tests if the operation is available before using it
2. **Fallback**: Uses `torch.autograd.grad` when the operation is not available
3. **Compatibility**: Maintains full backward compatibility with older PyTorch versions

### Performance Considerations

- **GPU Memory**: A100 80GB recommended for 1024x1024 training
- **Batch Size**: Start with 8, adjust based on available memory
- **Mixed Precision**: Automatically enabled for better performance
- **Data Loading**: Optimized for fast data loading

## 🎉 Expected Results

After successful training, you should see:
- Generated images in `/workspace/output/`
- Training metrics (FID, IS, etc.)
- Model checkpoints every 50 kimg
- Progressive quality improvement over time

## 📞 Support

If you encounter issues:
1. Run the compatibility test first
2. Check the troubleshooting section
3. Verify GPU memory and CUDA installation
4. Ensure all dependencies are installed correctly

## 🔄 Updates

To update the StyleGAN2-ADA submodule:
```bash
cd /workspace/train/runpod-hf/vendor/stylegan2ada
git pull origin pytorch-2.0-compatibility
```

To update the main repository:
```bash
cd /workspace
git pull origin feat/sg2-runpod-setup
```