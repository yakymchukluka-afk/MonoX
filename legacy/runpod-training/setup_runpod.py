#!/usr/bin/env python3
"""
RunPod Setup Script for MonoX StyleGAN-V Training
Configures environment and dependencies for RunPod deployment
"""

import os
import subprocess
import sys
import torch
import yaml
from pathlib import Path

def check_runpod_environment():
    """Check if running in RunPod environment."""
    runpod_pod_id = os.environ.get('RUNPOD_POD_ID')
    if runpod_pod_id:
        print(f"✅ Running in RunPod pod: {runpod_pod_id}")
        return True
    else:
        print("⚠️ Not running in RunPod environment")
        return False

def check_gpu_availability():
    """Check GPU availability and type."""
    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return False
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU Available: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f}GB")
    print(f"   Count: {gpu_count}")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n🔧 Installing dependencies...")
    
    # Install PyTorch with CUDA support
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "--index-url", 
        "https://download.pytorch.org/whl/cu118"
    ], check=True)
    
    # Install other dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "huggingface_hub", "gradio", "Pillow", 
        "numpy", "matplotlib", "tqdm", "pyyaml"
    ], check=True)
    
    print("✅ Dependencies installed")

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "checkpoints",
        "samples", 
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created")

def load_runpod_config():
    """Load RunPod configuration."""
    config_path = Path("runpod_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ RunPod configuration loaded")
        return config
    else:
        print("⚠️ RunPod configuration not found, using defaults")
        return None

def optimize_for_gpu():
    """Optimize environment for GPU training."""
    print("\n⚡ Optimizing for GPU...")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Create torch extensions directory
    Path('/tmp/torch_extensions').mkdir(exist_ok=True)
    
    print("✅ GPU optimization complete")

def main():
    """Main setup function."""
    print("🚀 RunPod Setup for MonoX StyleGAN-V Training")
    print("=" * 50)
    
    # Check environment
    check_runpod_environment()
    
    if not check_gpu_availability():
        print("❌ GPU not available, exiting")
        return False
    
    # Setup
    install_dependencies()
    setup_directories()
    load_runpod_config()
    optimize_for_gpu()
    
    print("\n✅ RunPod setup complete!")
    print("🎯 Ready to start training")
    
    return True

if __name__ == "__main__":
    main()