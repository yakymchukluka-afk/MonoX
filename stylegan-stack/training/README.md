# StyleGAN-V Training Environment for Mono Project

This directory contains a complete setup for training StyleGAN-V on 1024×1024 video data for the Mono Project.

## 🚀 Quick Start

### Prerequisites
- Dataset uploaded to `/workspace/stylegan-stack/data/originals/`
- GPU(s) available (CUDA recommended)
- All dependencies installed (see Installation section)

### Start Training
```bash
cd /workspace/stylegan-stack
./train_styleganv.sh
```

### Resume from Checkpoint
```bash
./train_styleganv.sh --resume /workspace/stylegan-stack/models/checkpoints/checkpoint_0250kimg.pkl
```

## 📁 Directory Structure

```
stylegan-stack/
├── data/
│   └── originals/              # Dataset directory (video frames organized by video)
├── models/
│   └── checkpoints/            # Training checkpoints (every 250 kimg)
├── logs/                       # Training logs and metrics
├── generated_previews/         # Sample images generated during training
├── training/                   # This documentation directory
├── stylegan-v/                 # StyleGAN-V source code
├── training_config.yaml        # Main training configuration
└── train_styleganv.sh         # Training launch script
```

## ⚙️ Configuration

### Dataset Format
Your dataset should be organized as:
```
data/originals/
    video1/
        frame001.jpg
        frame002.jpg
        ...
    video2/
        frame001.jpg
        frame002.jpg
        ...
```

All frames should be 1024×1024 pixels in JPEG or PNG format.

### Training Parameters
- **Resolution**: 1024×1024 pixels
- **Training Duration**: 3000 kimg (thousands of images)
- **Checkpoint Interval**: Every 250 kimg
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Precision**: Mixed precision (FP16/AMP) enabled
- **Augmentation**: Adaptive discriminator augmentation (ADA)

### Key Configuration Files
- `training_config.yaml` - Main training configuration
- `stylegan-v/configs/dataset/mono_project.yaml` - Dataset configuration
- `stylegan-v/configs/training/mono_training.yaml` - Training parameters

## 🎮 Usage Examples

### Basic Training
```bash
./train_styleganv.sh
```

### Custom GPU Count
```bash
./train_styleganv.sh --gpus 2
```

### Custom Batch Size
```bash
./train_styleganv.sh --batch-size 8
```

### Dry Run (Test Configuration)
```bash
./train_styleganv.sh --dry-run
```

### Resume Training
```bash
./train_styleganv.sh --resume /workspace/stylegan-stack/models/checkpoints/checkpoint_0500kimg.pkl
```

## 📊 Monitoring Training

### Logs
Training logs are saved to `/workspace/stylegan-stack/logs/` with timestamps.

### Checkpoints
Model checkpoints are saved every 250 kimg to `/workspace/stylegan-stack/models/checkpoints/`:
- `checkpoint_0250kimg.pkl`
- `checkpoint_0500kimg.pkl`
- `checkpoint_0750kimg.pkl`
- etc.

### Sample Images
Generated preview images are saved to `/workspace/stylegan-stack/generated_previews/` during training.

### TensorBoard (Optional)
If you want to use TensorBoard for monitoring:
```bash
tensorboard --logdir /workspace/stylegan-stack/logs/
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Reduce number of GPUs if using multiple

**Dataset Not Found**
- Ensure dataset is uploaded to `/workspace/stylegan-stack/data/originals/`
- Check that frames are properly organized by video

**Training Crashes**
- Check logs in `/workspace/stylegan-stack/logs/`
- Ensure all dependencies are installed
- Verify GPU memory and CUDA installation

### Performance Optimization

**For Lambda Labs / Cloud Training:**
- Use multiple GPUs if available: `--gpus 4`
- Enable TF32 on Ampere GPUs (RTX 30/40 series, A100, etc.)
- Monitor GPU utilization with `gpustat`

**For Google Colab:**
- Use smaller batch size: `--batch-size 4`
- Single GPU only: `--gpus 1`
- Enable runtime disconnect protection

## 🎯 Expected Training Time

On modern hardware:
- **4x V100 (32GB)**: ~2-3 days for 3000 kimg
- **4x RTX 3090**: ~3-4 days for 3000 kimg  
- **Single RTX 3090**: ~10-12 days for 3000 kimg
- **Google Colab (T4)**: ~20+ days for 3000 kimg

## 📈 Evaluation Metrics

The training automatically computes:
- **FVD (Fréchet Video Distance)**: Video quality metric
- **FID (Fréchet Inception Distance)**: Image quality metric
- Various temporal consistency metrics

## 🔄 Advanced Configuration

### Modifying Training Parameters
Edit `training_config.yaml` to adjust:
- Learning rates
- Regularization parameters
- Augmentation settings
- Model architecture parameters

### Custom Sampling Strategies
Edit `stylegan-v/configs/sampling/` to modify temporal sampling behavior.

### Multi-GPU Scaling
The setup automatically scales to available GPUs. For best performance:
- Use batch sizes divisible by GPU count
- Ensure sufficient VRAM (16GB+ recommended for 1024px)

## 📝 Notes

- First-time training may take longer due to JIT compilation
- Checkpoints include full model state for resuming
- Generated samples improve significantly after 1000+ kimg
- Monitor training curves for convergence
- Keep multiple checkpoint files as backup

## 🆘 Support

For issues specific to this setup, check:
1. Training logs in `/workspace/stylegan-stack/logs/`
2. StyleGAN-V documentation in `stylegan-v/README.md`
3. Original paper: https://universome.github.io/stylegan-v