# MonoX StyleGAN-V Hugging Face Training

**Complete StyleGAN-V training on Hugging Face Spaces with GPU support**

## 🚀 Features

- **Hugging Face Spaces**: Deploy directly to HF Spaces
- **GPU Support**: T4 GPU acceleration for fast training
- **Web Interface**: Gradio-based training dashboard
- **Auto Upload**: Models and samples uploaded to HF Hub
- **Checkpoint Resuming**: Resume training from any checkpoint
- **Real-time Monitoring**: Live training progress updates

## 📁 Files

### Core Application
- `app.py` - Main Gradio application with training interface
- `streamlit_app.py` - Alternative Streamlit interface
- `train.py` - Core training script with HF integration

### Docker & Deployment
- `Dockerfile` - Main Docker configuration
- `Dockerfile.gpu` - GPU-optimized Docker image
- `Dockerfile.minimal` - Minimal deployment image
- `requirements.txt` - Python dependencies
- `requirements.gpu.txt` - GPU-specific dependencies

### HF Space Integration
- `hf_space/` - HF Space specific files
- `setup_hf_space.py` - HF Space setup script
- `test_space_auth.py` - Authentication testing
- `test_upload.py` - Upload testing

### Authentication & Security
- `hybrid_authentication.py` - Multi-auth system
- `monox_hybrid_auth.py` - MonoX specific auth
- `secure_setup.py` - Secure environment setup
- `ssh_key_authentication.py` - SSH key auth

### Training Infrastructure
- `src/` - Core training modules
- `configs/` - Training configurations
- `scripts/` - Training scripts
- `logs/` - Training logs and outputs

## 🎯 Quick Start

### 1. Deploy to HF Spaces
```bash
# Clone and setup
git clone <your-repo>
cd monox-hf-training

# Set HF token as Space secret
# Name: HF_TOKEN
# Value: your_huggingface_token

# Deploy to HF Spaces
```

### 2. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

### 3. GPU Training
```bash
# Use GPU Docker
docker build -f Dockerfile.gpu -t monox-gpu .
docker run --gpus all monox-gpu
```

## 🔧 Configuration

### Environment Variables
- `HF_TOKEN` - Hugging Face authentication token
- `CUDA_VISIBLE_DEVICES` - GPU device selection
- `MODEL_REPO` - Target model repository

### Training Parameters
- **Resolution**: 1024x1024 pixels
- **Epochs**: 50 (configurable)
- **Batch Size**: 2 (GPU optimized)
- **Learning Rate**: 2e-4
- **Checkpoints**: Every 1M epochs

## 📊 Performance

| Platform | GPU | Time per Epoch | Total Time | Cost |
|----------|-----|---------------|------------|------|
| HF Spaces T4 | T4 | 30 seconds | 25 minutes | $0.60/hour |
| Local GPU | V100 | 20 seconds | 17 minutes | Free |
| CPU | - | 15 minutes | 12+ hours | Free |

## 🎨 Model Architecture

- **Generator**: L4Generator1024
- **Noise Dimension**: 512
- **Output Resolution**: 1024x1024
- **Architecture**: Progressive upsampling with batch normalization
- **Activation**: ReLU + Tanh

## 📝 Usage

### Web Interface
1. Open the Gradio interface
2. Check system status
3. Start training with one click
4. Monitor progress in real-time
5. Download results

### Programmatic Training
```python
from train import L4Generator1024, start_training

# Initialize model
generator = L4Generator1024()

# Start training
start_training(generator, epochs=50)
```

### Checkpoint Management
```python
from train import find_latest_checkpoint, load_checkpoint

# Find latest checkpoint
ckpt_path, epoch = find_latest_checkpoint()

# Load and resume
load_checkpoint(generator, optimizer, scaler, ckpt_path)
```

## 🔗 Related

- **Colab Training**: See `collab-stylegen-training` branch
- **RunPod**: See `runpod-training` branch
- **Main**: See `main` branch for complete implementation

## 📄 License

MIT License