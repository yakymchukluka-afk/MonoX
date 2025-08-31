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
python_version: 3.9
---

# MonoX StyleGAN-V Training Interface

A Hugging Face Space for training StyleGAN-V models using the MonoX framework.

## Features

- **Web Interface**: Easy-to-use Gradio interface for training control
- **JSON API**: RESTful API endpoints for programmatic access
- **Real-time Monitoring**: Live training logs and status updates
- **GPU Support**: Automatic CUDA detection and GPU utilization
- **Checkpoint Management**: List and manage training checkpoints

## API Endpoints

### System Information
- `GET /` - API information and available endpoints
- `GET /system/info` - System status, GPU info, workspace status
- `GET /health` - Health check endpoint

### Training Control
- `POST /training/start` - Start training with configuration
- `POST /training/stop` - Stop current training process
- `GET /training/status` - Get current training status
- `GET /training/logs?lines=500` - Get training logs

### Checkpoint Management
- `GET /checkpoints/list` - List available checkpoints

## Usage

### Web Interface
Access the Gradio interface directly in your browser to:
- Check system status and GPU availability
- Start/stop training with custom parameters
- Monitor training progress in real-time
- View and manage checkpoints

### API Usage
Use the JSON API for programmatic control:

```python
import requests

# Check system status
response = requests.get("https://your-space-url/system/info")
print(response.json())

# Start training
training_config = {
    "dataset_path": "/workspace/dataset",
    "total_kimg": 1000,
    "resolution": 1024,
    "num_gpus": 1
}
response = requests.post("https://your-space-url/training/start", json=training_config)
print(response.json())

# Check training status
response = requests.get("https://your-space-url/training/status")
print(response.json())
```

## Configuration

### Environment Variables
- `DATASET_DIR`: Path to training dataset (default: `/workspace/dataset`)
- `LOGS_DIR`: Directory for training logs (default: `logs`)
- `CKPT_DIR`: Directory for checkpoints (default: `checkpoints`)

### Training Parameters
- **Dataset Path**: Path to your training images
- **Total KImg**: Total training iterations in thousands
- **Resolution**: Output image resolution (e.g., 1024)
- **Num GPUs**: Number of GPUs to use for training

## Troubleshooting

### Common Issues

1. **"Unexpected token '<', DOCTYPE..." Error**
   - This occurs when accessing HTML endpoints as JSON API
   - Use the correct API endpoints listed above
   - Ensure you're sending proper HTTP headers for JSON

2. **GPU Not Detected**
   - Check if CUDA drivers are available in the environment
   - Verify GPU quota and availability in Hugging Face Spaces

3. **Training Won't Start**
   - Ensure dataset path exists and contains valid images
   - Check system logs for detailed error messages
   - Verify all dependencies are installed

### Getting Help

If you encounter issues:
1. Check the system info endpoint for environment status
2. Review training logs for specific error messages
3. Ensure your dataset is properly formatted
4. Check GPU availability and CUDA installation

## Development

This space uses:
- **StyleGAN-V**: Advanced video-capable StyleGAN implementation
- **Hydra**: Configuration management
- **Gradio**: Web interface framework
- **FastAPI**: JSON API backend
- **PyTorch**: Deep learning framework

## License

MIT License - see LICENSE file for details.