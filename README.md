# MonoX StyleGAN-V Training

**Complete StyleGAN-V training implementation with multiple deployment options**

## 🚀 Branches

This repository is organized into three specialized branches for different training environments:

### 🎨 [collab-stylegen-training](https://github.com/yakymchukluka-afk/MonoX/tree/collab-stylegen-training)
**Google Colab GPU Training**
- Free GPU training using Google Colab
- Complete Jupyter notebook setup
- 30x faster than CPU training
- Perfect for experimentation and learning

### 🤗 [hf-training](https://github.com/yakymchukluka-afk/MonoX/tree/hf-training)
**Hugging Face Spaces Deployment**
- Deploy directly to Hugging Face Spaces
- Web interface with Gradio
- Automatic model uploads to HF Hub
- Docker-based deployment

### ☁️ [runpod-training](https://github.com/yakymchukluka-afk/MonoX/tree/runpod-training)
**RunPod Cloud GPU Training**
- High-performance cloud GPU training
- Multiple GPU options (T4, V100, A100, RTX 4090)
- Cost-effective training solutions
- Production-ready deployment

## 🎯 Quick Start

Choose your preferred training environment:

### Google Colab (Free)
```bash
git checkout collab-stylegen-training
# Open MonoX_GPU_Colab.ipynb in Google Colab
# Enable GPU runtime and run all cells
```

### Hugging Face Spaces
```bash
git checkout hf-training
# Deploy to HF Spaces with GPU support
# Set HF_TOKEN as Space secret
```

### RunPod Cloud
```bash
git checkout runpod-training
# Deploy to RunPod with your preferred GPU
# Configure runpod_config.yaml
```

## 🎨 Model Architecture

- **Generator**: L4Generator1024 (512 noise → 1024x1024 images)
- **Resolution**: 1024x1024 pixels
- **Training**: 50 epochs with checkpoints
- **Output**: High-quality monotype-style artwork

## 📊 Performance Comparison

| Platform | GPU | Time per Epoch | Total Time | Cost |
|----------|-----|---------------|------------|------|
| **Google Colab** | T4 | 30 seconds | 25 minutes | FREE |
| **HF Spaces** | T4 | 30 seconds | 25 minutes | $0.60/hour |
| **RunPod** | T4 | 45 seconds | 37 minutes | $0.20/hour |
| **RunPod** | A100 | 15 seconds | 12 minutes | $1.50/hour |

## 🔧 Features

- **StyleGAN-V Implementation**: Complete L4Generator1024 architecture
- **Multiple Platforms**: Colab, HF Spaces, RunPod support
- **GPU Optimization**: Mixed precision, gradient accumulation
- **Checkpoint Resuming**: Resume training from any checkpoint
- **Real-time Monitoring**: Live training progress updates
- **Auto Upload**: Results automatically uploaded to HF Hub

## 📝 Usage

### Basic Training
```python
from train import L4Generator1024, start_training

# Initialize model
generator = L4Generator1024()

# Start training
start_training(generator, epochs=50)
```

### Web Interface
```python
# Run Gradio interface
python app.py
```

## 🔗 Related

- **Colab Training**: See `collab-stylegen-training` branch
- **HF Training**: See `hf-training` branch  
- **RunPod**: See `runpod-training` branch

## 📄 License

MIT License