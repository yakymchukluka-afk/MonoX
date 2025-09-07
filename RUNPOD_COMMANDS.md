# RunPod StyleGAN2-ADA Commands

## ✅ Setup Complete!

The StyleGAN2-ADA PyTorch 2.0+ compatibility fix has been successfully implemented and tested. Here are the correct commands to run on your RunPod instance:

## 🚀 Quick Start Commands

### 1. Navigate to the workspace directory
```bash
cd /workspace
```

### 2. Test PyTorch compatibility
```bash
python3 test_pytorch_compatibility.py
```

### 3. Run training setup test
```bash
python3 train_stylegan2ada_simple.py
```

### 4. Start actual training (when ready)
```bash
# First, download your dataset to /workspace/data/monox-dataset.zip
# Then run:
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

## 🔧 What Was Fixed

### PyTorch 2.0+ Compatibility Issue
- **Problem**: `RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented`
- **Solution**: Added fallback mechanism in `grid_sample_gradfix.py`
- **Result**: ✅ Works with PyTorch 2.8.0+ and maintains backward compatibility

### Key Changes Made
1. **Enhanced `_should_use_custom_op()`**: Tests operation availability before using custom ops
2. **Updated `_GridSample2dBackward`**: Uses `torch.autograd.grad` as fallback for newer PyTorch versions
3. **Maintained compatibility**: Still works with older PyTorch versions

## 📊 Test Results

```
✅ PyTorch version: 2.8.0+cu128
✅ grid_sample_gradfix: PASS
✅ StyleGAN2-ADA imports: PASS  
✅ Network creation: PASS
✅ Gradient computation: PASS
✅ Overall: PASS
```

## 📁 Directory Structure

```
/workspace/
├── train/runpod-hf/vendor/stylegan2ada/    # StyleGAN2-ADA (patched)
├── data/                                   # Your dataset goes here
├── output/                                 # Training outputs
├── test_pytorch_compatibility.py          # Compatibility test
├── train_stylegan2ada_simple.py           # Setup test
└── STYLEGAN2_ADA_RUNPOD_SETUP.md          # Full documentation
```

## 🎯 Next Steps

1. **Download your dataset** to `/workspace/data/monox-dataset.zip`
2. **Run the compatibility test** to verify everything works
3. **Start training** with the command above
4. **Monitor progress** in `/workspace/output/`

## 🔍 Troubleshooting

If you encounter any issues:

1. **Check PyTorch version**: `python3 -c "import torch; print(torch.__version__)"`
2. **Test compatibility**: `python3 test_pytorch_compatibility.py`
3. **Verify dataset**: `ls -la /workspace/data/`
4. **Check GPU**: `nvidia-smi` (on GPU instance)

## 🎉 Success!

The PyTorch compatibility issue has been completely resolved. StyleGAN2-ADA is now ready to train on your RunPod instance with PyTorch 2.0+!