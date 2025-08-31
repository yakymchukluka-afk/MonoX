---
title: MonoX StyleGAN-V Training
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# MonoX StyleGAN-V Training Interface

A Hugging Face Space for training StyleGAN-V models using the MonoX framework with monotype-inspired artwork generation.

## Features

- **Fresh Training**: Train from scratch using 1024px monotype dataset
- **Web Interface**: Easy-to-use Gradio interface for training control
- **CPU/GPU Support**: Works with both CPU and GPU compute
- **Automatic Uploads**: All outputs uploaded to lukua/monox model repo
- **Real-time Monitoring**: Live training progress and status updates

## Quick Start

1. **Set HF Token**: Add your HF token as a Space secret named `HF_TOKEN`
2. **Start Training**: Use the web interface or run training scripts
3. **Monitor Progress**: Check real-time status and outputs
4. **View Results**: All outputs automatically uploaded to lukua/monox

## Training Configuration

- **Dataset**: lukua/monox-dataset (868 images at 1024×1024)
- **Architecture**: GAN with Generator and Discriminator
- **Training**: 50 epochs with checkpoints every 5 epochs
- **Outputs**: Checkpoints, preview images, and comprehensive logs

## Usage

### Web Interface
Access the Gradio interface to:
- Check system status and configuration
- Start/stop training processes
- Monitor training progress in real-time
- View generated samples and checkpoints

### Direct Training
```bash
python3 simple_gan_training.py
```

### Monitor Progress
```bash
python3 monitor_training.py
```

## Output Structure

All outputs are uploaded to `lukua/monox`:
- `/checkpoints` - Model checkpoints every 5 epochs
- `/previews` - Generated sample images per epoch
- `/logs` - Training logs and progress reports

## Hardware Requirements

- **CPU**: Works with free CPU compute (slower but functional)
- **GPU**: Faster training with paid GPU compute
- **Memory**: Optimized for available resources

## Security

- All authentication via environment variables
- No hardcoded tokens in source code
- Secure HF token handling

## License

MIT License