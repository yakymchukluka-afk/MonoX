# StyleGAN2-ADA Training Setup for RunPod

This repository provides a complete setup for training StyleGAN2-ADA models on RunPod cloud GPU instances.

## 🚀 Quick Start

### 1. Setup Environment
```bash
bash setup_stylegan2_ada.sh
```

### 2. Prepare Dataset
```bash
# Prepare your image dataset
bash prepare_dataset.sh /path/to/your/images /workspace/datasets/dataset.zip 1024
```

### 3. Start Training
```bash
# Start training with default parameters
bash train_stylegan2_ada.sh

# Or with custom parameters
bash train_stylegan2_ada.sh /workspace/datasets/dataset.zip /workspace/output 1024 8 25000
```

### 4. Monitor Training
```bash
# Monitor training progress
bash monitor_training.sh
```

## 📁 File Structure

```
/workspace/
├── setup_stylegan2_ada.sh      # Environment setup script
├── prepare_dataset.sh           # Dataset preparation script
├── train_stylegan2_ada.sh      # Training script
├── monitor_training.sh          # Training monitoring script
├── generate_samples.py          # Sample generation script
├── runpod_config.yaml          # Configuration file
├── stylegan2-ada/              # Official StyleGAN2-ADA repository
├── datasets/                   # Dataset storage
├── output/                     # Training output
├── checkpoints/                # Model checkpoints
├── samples/                    # Generated samples
└── logs/                       # Training logs
```

## 🔧 Configuration

Edit `runpod_config.yaml` to customize your training:

### GPU Selection
- **T4**: 16GB memory, $0.20/hour, batch size 4
- **V100**: 32GB memory, $0.50/hour, batch size 8
- **A100**: 40GB memory, $1.50/hour, batch size 12
- **RTX 4090**: 24GB memory, $0.80/hour, batch size 6

### Training Parameters
- `batch_size`: Adjust based on GPU memory
- `kimg`: Total training images (25M = 25000 kimg)
- `resolution`: Image resolution (256, 512, 1024)
- `snapshot_kimg`: Checkpoint frequency

## 📊 Monitoring

### Real-time Monitoring
```bash
bash monitor_training.sh
```

### Check Training Logs
```bash
tail -f /workspace/logs/train.log
```

### View Generated Samples
```bash
ls -la /workspace/output/
```

## 🎨 Generating Samples

After training, generate samples from your model:

```bash
python generate_samples.py /workspace/output/00000-stylegan2-ada-custom-res256-batch8/network-snapshot-000250.pkl
```

## 💰 Cost Optimization

### Spot Instances
- Use spot instances for 50-70% cost savings
- Set `spot_instance: true` in config

### Right-sizing
- Match GPU to your dataset size
- Smaller datasets can use T4
- Large datasets need A100

### Auto-shutdown
- Enable `shutdown_on_complete: true`
- Set `max_runtime` to prevent runaway costs

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use smaller resolution
   - Enable mixed precision

2. **Dataset Issues**
   - Ensure images are properly formatted
   - Check minimum image count (1000+)
   - Verify file permissions

3. **Training Stalls**
   - Check GPU utilization
   - Monitor disk space
   - Verify dataset integrity

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Check disk space
df -h

# Check training process
ps aux | grep train.py

# View recent logs
tail -100 /workspace/logs/train.log
```

## 📈 Performance Tips

1. **Use Mixed Precision**: Enable `fp32: false`
2. **Optimize Batch Size**: Find the largest batch that fits in memory
3. **Use Data Augmentation**: Enable ADA augmentation
4. **Monitor Progress**: Use monitoring scripts
5. **Save Checkpoints**: Regular snapshots prevent data loss

## 🎯 Expected Results

### Training Times (25M images)
- **T4**: ~8-12 hours
- **V100**: ~4-6 hours  
- **A100**: ~2-3 hours
- **RTX 4090**: ~3-4 hours

### Quality Progression
- **5k images**: Basic shapes and colors
- **10k images**: Recognizable features
- **15k images**: Good quality samples
- **20k+ images**: High-quality, diverse samples

## 📚 Additional Resources

- [StyleGAN2-ADA Paper](https://arxiv.org/abs/2006.06676)
- [Official Repository](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [RunPod Documentation](https://docs.runpod.io/)
- [StyleGAN2-ADA Training Guide](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/docs/training.md)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review training logs
3. Verify configuration settings
4. Check RunPod instance status