# StyleGAN-V Training Environment for Mono Project

> **Complete training infrastructure for 1024×1024 video generation using StyleGAN-V**

## 🎯 Overview

This repository contains a fully configured StyleGAN-V training environment optimized for the Mono Project. The setup includes:

- ✅ **StyleGAN-V** from universome/stylegan-v (latest stable version)
- ✅ **1024×1024 resolution** training configuration
- ✅ **3000 kimg** training duration with **250 kimg** checkpoints
- ✅ **Multi-GPU** and **AMP/FP16** support
- ✅ **Robust checkpointing** and **resume capabilities**
- ✅ **Training monitoring** and **progress visualization**
- ✅ **Google Colab** compatibility

## 🚀 Quick Start

### 1. Upload Dataset
```bash
# Place your video frames in this structure:
/workspace/stylegan-stack/data/originals/
    video1/frame001.jpg
    video2/frame001.jpg
    ...
```

### 2. Verify Setup
```bash
python3 verify_setup.py
```

### 3. Start Training
```bash
./train_styleganv.sh
```

### 4. Monitor Progress (Optional)
```bash
# In a separate terminal
python3 monitor_training.py
```

## 📁 Directory Structure

```
stylegan-stack/
├── 📊 data/
│   └── originals/              # Dataset directory (upload here)
├── 🤖 models/
│   └── checkpoints/            # Training checkpoints (every 250 kimg)
├── 📈 logs/                    # Training logs and metrics
├── 🖼️  generated_previews/      # Sample images during training
├── 🛠️  stylegan-v/              # StyleGAN-V source code
├── ⚙️  training_config.yaml     # Main training configuration
├── 🚀 train_styleganv.sh       # Training launch script
├── 🔍 verify_setup.py          # Environment verification
├── 📊 monitor_training.py      # Training progress monitor
└── 📖 README.md               # This file
```

## 🎛️ Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Resolution** | 1024×1024 | Target resolution |
| **Duration** | 3000 kimg | ~3000K training images |
| **Checkpoints** | Every 250 kimg | 12 total checkpoints |
| **Batch Size** | 16 | Adjustable for GPU memory |
| **Precision** | FP16/AMP | Mixed precision enabled |
| **GPUs** | Auto-detect | Scales to available hardware |
| **Augmentation** | ADA | Adaptive discriminator augmentation |

## 💻 Usage Examples

```bash
# Basic training
./train_styleganv.sh

# Resume from specific checkpoint  
./train_styleganv.sh --resume /workspace/stylegan-stack/models/checkpoints/checkpoint_0500kimg.pkl

# Adjust batch size for memory constraints
./train_styleganv.sh --batch-size 8

# Use specific number of GPUs
./train_styleganv.sh --gpus 2

# Test configuration without training
./train_styleganv.sh --dry-run

# Generate training report
python3 monitor_training.py --report-only
```

## 📊 Expected Timeline

| Hardware | Estimated Time |
|----------|----------------|
| 4x V100 (32GB) | 2-3 days |
| 4x RTX 3090 | 3-4 days |
| Single RTX 3090 | 8-12 days |
| Google Colab T4 | 20+ days |

## 🔍 Monitoring

### Real-time Monitoring
```bash
python3 monitor_training.py
```

### Manual Checks
- **Checkpoints**: `ls models/checkpoints/`
- **Latest log**: `tail -f logs/training_*.log`
- **GPU usage**: `gpustat` (if installed)

### Files Generated
- `models/checkpoints/checkpoint_XXXXkimg.pkl` - Model checkpoints
- `logs/training_YYYYMMDD_HHMMSS.log` - Training logs
- `logs/training_progress.png` - Progress visualization
- `generated_previews/` - Sample images

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
./train_styleganv.sh --batch-size 8  # Reduce batch size
```

**Dataset Not Found**
```bash
# Ensure dataset is in correct location:
ls /workspace/stylegan-stack/data/originals/
```

**Training Crashes**
```bash
# Check logs for details:
tail -100 logs/training_*.log
```

**Resume Training**
```bash
# Find latest checkpoint:
ls -la models/checkpoints/
# Resume from it:
./train_styleganv.sh --resume models/checkpoints/checkpoint_0750kimg.pkl
```

## ⚡ Performance Optimization

### For Lambda Labs / Cloud
- Use multiple GPUs: `--gpus 4`
- Enable TF32 on Ampere GPUs (automatic)
- Monitor with `gpustat`

### For Google Colab
- Use smaller batch size: `--batch-size 4`
- Single GPU only: `--gpus 1`
- Enable runtime disconnect protection

### Memory Issues
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Use gradient checkpointing (automatic)
- Close other applications

## 📦 Dependencies

### Core Requirements
- **PyTorch** 2.7.1+ with CUDA 11.8
- **StyleGAN-V** (included)
- **Python** 3.8+

### Key Packages
- `torch`, `torchvision` - Deep learning framework
- `opencv-python` - Image processing
- `hydra-core` - Configuration management
- `tensorboard` - Logging and visualization
- `matplotlib` - Plotting and visualization

## 🔬 Advanced Features

### Custom Configurations
Edit `training_config.yaml` to modify:
- Model architecture parameters
- Learning rates and optimization
- Augmentation strategies
- Sampling techniques

### Monitoring and Analysis
- Real-time FID score tracking
- Training curve visualization
- Early stopping recommendations
- Progress estimation

### Resume and Checkpointing
- Automatic checkpoint saving every 250 kimg
- Seamless resume from any checkpoint
- Full model state preservation

## 📝 Important Notes

- **First Training**: Initial compilation may take 5-10 minutes
- **Convergence**: Quality improves significantly after 1000+ kimg
- **Checkpoints**: Keep multiple checkpoints as backup
- **Memory**: 1024px training requires 16GB+ VRAM for optimal batch sizes
- **Dataset**: Ensure frames are exactly 1024×1024 pixels

## 🆘 Support

### Setup Issues
1. Run `python3 verify_setup.py` to check environment
2. Check training logs in `logs/` directory
3. Consult `training/README.md` for detailed instructions

### Training Issues
1. Check GPU memory with `nvidia-smi`
2. Reduce batch size if out of memory
3. Monitor training curves for convergence

### Resources
- [StyleGAN-V Paper](https://universome.github.io/stylegan-v)
- [Original Repository](https://github.com/universome/stylegan-v)
- Training documentation in `training/README.md`

---

**🎉 Ready to train! Upload your dataset and run `./train_styleganv.sh` to begin.**