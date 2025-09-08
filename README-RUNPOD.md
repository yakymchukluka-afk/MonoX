# MonoX StyleGAN2-ADA RunPod Training Guide

🎯 **Complete setup and training guide for StyleGAN2-ADA on RunPod A100 instances**

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yakymchukluka-afk/MonoX.git
cd MonoX
git checkout runpod/sg2-1024
git submodule update --init --recursive

# 2. RunPod setup
bash scripts/runpod/bootstrap.sh

# 3. Prepare dataset
bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /workspace/datasets/mydataset.zip 1024 1024

# 4. Start training
bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/datasets/mydataset.zip 8

# 5. Monitor training
tmux attach -t monox
```

## 📁 Repository Structure

```
runpod/sg2-1024/
├── vendor/stylegan2ada/          # Official NVLabs StyleGAN2-ADA submodule
├── scripts/runpod/
│   ├── bootstrap.sh              # System setup & dependencies
│   ├── make_dataset_zip.sh       # Dataset preparation
│   └── train.sh                  # Training execution
├── configs/runpod/
│   └── sg2-1024.example.yaml     # Training configuration
├── README-RUNPOD.md              # This guide
└── .gitattributes                # LFS patterns for images/zips
```

## 🔧 Prerequisites

### RunPod Instance Requirements
- **GPU**: A100 (recommended) or V100
- **Memory**: 80GB+ RAM
- **Storage**: 100GB+ free space
- **CUDA**: 12.1+ (included in RunPod A100 images)

### Dataset Requirements
- **Format**: PNG, JPG, or other common image formats
- **Resolution**: Any (will be resized to target resolution)
- **Count**: 1000+ images recommended (can work with fewer using transfer learning)
- **Organization**: Single folder with all images

## 📊 Dataset Preparation

### Step 1: Prepare Your Images
```bash
# Organize your dataset in a single folder
/path/to/dataset/
├── image1.jpg
├── image2.png
├── image3.jpg
└── ...
```

### Step 2: Create Dataset ZIP
```bash
# Basic usage
bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /workspace/datasets/mydataset.zip

# With custom resolution
bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /workspace/datasets/mydataset.zip 512 512

# For 1024x1024 (default)
bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /workspace/datasets/mydataset.zip 1024 1024
```

### Step 3: Verify Dataset
The script will automatically verify your dataset and show:
- Number of images found
- Dataset file size
- Loading test results

## 🚀 Training

### Basic Training
```bash
# Start training with default settings
bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/datasets/mydataset.zip 8
```

### Advanced Training Options
```bash
# Custom configuration
bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/datasets/mydataset.zip 8 stylegan2

# Transfer learning from FFHQ
bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/datasets/mydataset.zip 8 auto
```

### Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `outdir` | Output directory | `/workspace/out/sg2` | Any path |
| `dataset_zip` | Dataset ZIP file | Required | Path to .zip file |
| `gpus` | Number of GPUs | `8` | 1-8 |
| `config` | Training config | `auto` | `auto`, `stylegan2`, `paper1024` |

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Attach to training session
tmux attach -t monox

# View training logs
tail -f /workspace/out/sg2/log.txt

# List tmux sessions
tmux list-sessions
```

### Training Outputs
- **Logs**: `/workspace/out/sg2/log.txt`
- **Samples**: `/workspace/out/sg2/fakes*.png`
- **Checkpoints**: `/workspace/out/sg2/network-snapshot-*.pkl`
- **Metrics**: `/workspace/out/sg2/metric-fid50k_full.jsonl`

### Stopping Training
```bash
# Stop training session
tmux kill-session -t monox

# Or press Ctrl+C in the tmux session
```

## 🔄 Resuming Training

```bash
# Resume from last checkpoint
bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/datasets/mydataset.zip 8

# Resume from specific checkpoint
# Edit train.sh to add --resume=/path/to/checkpoint.pkl
```

## ⚙️ Configuration

### Training Configurations

| Config | Description | Best For |
|--------|-------------|----------|
| `auto` | Automatic selection | General use, new datasets |
| `stylegan2` | StyleGAN2 config F | High-quality results |
| `paper1024` | Paper configuration | 1024x1024 datasets |
| `paper512` | Paper configuration | 512x512 datasets |

### Dataset-Specific Settings

#### Small Datasets (< 1000 images)
```yaml
config: "paper1024"
gamma: 10.0
aug_target: 0.7
kimg: 10000
```

#### Medium Datasets (1000-10000 images)
```yaml
config: "auto"
gamma: 10.0
aug_target: 0.6
kimg: 25000
```

#### Large Datasets (> 10000 images)
```yaml
config: "stylegan2"
gamma: 10.0
aug_target: 0.6
kimg: 25000
```

## 🎯 Expected Training Times

| Resolution | GPUs | 1000 kimg | 25000 kimg | GPU Memory |
|------------|------|-----------|------------|------------|
| 512x512    | 8    | 2h 48m    | 2d 22h     | 7.8 GB     |
| 1024x1024  | 8    | 5h 54m    | 6d 03h     | 8.3 GB     |

*Times based on NVIDIA A100 GPUs*

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in train.sh
--batch=16 --batch-gpu=2
```

#### Dataset Loading Errors
```bash
# Check dataset format
python3 -c "
import zipfile
with zipfile.ZipFile('/workspace/datasets/mydataset.zip', 'r') as zf:
    print('Files in ZIP:', zf.namelist()[:10])
"
```

#### Training Diverges
- Try different `gamma` values (5.0, 10.0, 20.0)
- Enable transfer learning: `--resume=ffhq1024`
- Check dataset quality and size

### Getting Help

1. Check training logs: `tail -f /workspace/out/sg2/log.txt`
2. Verify GPU status: `nvidia-smi`
3. Check dataset: Use the verification in `make_dataset_zip.sh`
4. Review StyleGAN2-ADA documentation in `vendor/stylegan2ada/README.md`

## 📚 Additional Resources

- [StyleGAN2-ADA Paper](https://arxiv.org/abs/2006.06676)
- [Official StyleGAN2-ADA Repository](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [RunPod Documentation](https://docs.runpod.io/)

## 🎉 Success Checklist

- [ ] RunPod instance running with A100 GPU
- [ ] Dataset prepared and verified
- [ ] Training started successfully
- [ ] Monitoring setup working
- [ ] Checkpoints being saved
- [ ] Sample images being generated
- [ ] FID metrics being calculated

---

**Happy Training! 🚀**

For questions or issues, please check the troubleshooting section or refer to the official StyleGAN2-ADA documentation.