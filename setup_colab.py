#!/usr/bin/env python3
"""
Setup script for MonoX training in Google Colab
Installs all required dependencies and prepares the environment
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and print status"""
    if description:
        print(f"🔧 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up MonoX for Google Colab training...")
    
    # Downgrade pip to avoid metadata validation issues with older packages
    run_command("pip install 'pip<24.1'", "Downgrading pip for compatibility")
    
    # Install PyTorch with CUDA support for Colab
    run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch with CUDA support"
    )
    
    # Install compatible versions that work together
    run_command("pip install omegaconf==2.0.4", "Installing compatible OmegaConf")
    run_command("pip install hydra-core==1.0.7", "Installing Hydra")
    
    # Install compatible pytorch-lightning
    run_command("pip install 'pytorch-lightning>=1.5.0,<1.8.0'", "Installing PyTorch Lightning")
    
    # Install other ML dependencies
    dependencies = [
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "scipy>=1.7.0", 
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "matplotlib>=3.4.0",
        "opencv-python>=4.5.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.5",
        "ninja>=1.10.0",
        "psutil>=5.8.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0"
    ]
    
    for dep in dependencies:
        run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}")
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import hydra
        import omegaconf
        import torch
        import pytorch_lightning
        print("✅ All core dependencies installed successfully!")
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        print(f"🔥 Hydra version: {hydra.__version__}")
        print(f"⚡ Lightning version: {pytorch_lightning.__version__}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n🎉 MonoX setup complete! Ready for training.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)