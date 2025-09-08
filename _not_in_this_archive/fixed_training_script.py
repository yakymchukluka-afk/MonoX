#!/usr/bin/env python3
"""
Fixed MonoX Training Script with Proper HF Authentication
This script fixes the 401 authentication errors and upload issues.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, login, upload_file
from PIL import Image
import torchvision.utils as vutils
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_hf_authentication():
    """Setup Hugging Face authentication properly."""
    print("🔧 Setting up HF Authentication...")
    
    # Check for HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("⚠️ No HF authentication - samples saved locally only")
        return False
    
    try:
        # Login with token
        login(token=hf_token)
        print("✅ HF authentication successful")
        return True
    except Exception as e:
        print(f"❌ HF authentication failed: {e}")
        return False

def upload_to_hf_repo(file_path: str, repo_id: str = "lukua/monox-model") -> bool:
    """Upload file to HF repository with proper authentication."""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("⚠️ No HF token available - saving locally only")
        return False
    
    try:
        # Determine upload path based on file type
        file_name = os.path.basename(file_path)
        if file_path.endswith(('.pkl', '.pt', '.pth', '.ckpt')):
            repo_path = f"checkpoints/{file_name}"
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            repo_path = f"samples/{file_name}"
        elif file_path.endswith('.log'):
            repo_path = f"logs/{file_name}"
        else:
            repo_path = file_name
        
        # Upload with authentication
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            token=hf_token,
            repo_type="model"
        )
        
        print(f"✅ Uploaded: {file_name} -> {repo_path}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability and setup."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        
        print(f"🔍 GPU Available: True")
        print(f"🚀 GPU Detected: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        print(f"⚡ CUDA Version: {cuda_version}")
        
        return device, gpu_name, gpu_memory
    else:
        print("⚠️ No GPU detected, using CPU")
        return torch.device('cpu'), "CPU", 0

class SimpleGenerator(nn.Module):
    """Simple generator for testing."""
    def __init__(self, nz=100, ngf=64, nc=3):
        super(SimpleGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def train_monox_model():
    """Main training function with fixed authentication."""
    print("🎯 MonoX Training with Fixed Authentication")
    print("=" * 50)
    
    # Setup authentication
    hf_authenticated = setup_hf_authentication()
    
    # Check GPU
    device, gpu_name, gpu_memory = check_gpu_availability()
    
    # Create directories
    Path("samples").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Training parameters
    batch_size = 32 if device.type == 'cuda' else 8
    image_size = 64  # Start with smaller size for testing
    nz = 100
    num_epochs = 10  # Reduced for testing
    lr = 0.0002
    
    print(f"🎯 Training Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Epochs: {num_epochs}")
    
    # Create models
    netG = SimpleGenerator(nz=nz).to(device)
    criterion = nn.MSELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr)
    
    # Fixed noise for consistent samples
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    
    print(f"\\n🎨 Starting training loop...")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Simple training step (dummy data for testing)
        for i in range(5):  # 5 batches per epoch
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_data = netG(noise)
            
            # Simple loss (just for demonstration)
            loss = torch.mean(fake_data)
            
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()
        
        # Generate sample
        with torch.no_grad():
            fake_samples = netG(fixed_noise)
            sample_path = f"samples/monox_epoch_{epoch:05d}.png"
            vutils.save_image(fake_samples, sample_path, normalize=True, nrow=4)
        
        epoch_time = time.time() - epoch_start
        print(f"🎨 Epoch {epoch+1:02d}/{num_epochs}, Loss: {loss.item():.4f}, Time: {epoch_time:.1f}s")
        
        # Upload sample
        if hf_authenticated:
            upload_to_hf_repo(sample_path)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/monox_checkpoint_epoch_{epoch+1:05d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            if hf_authenticated:
                upload_to_hf_repo(checkpoint_path)
        
        # Save training log
        log_path = f"logs/training_log_epoch_{epoch:05d}.txt"
        with open(log_path, 'w') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\\n")
            f.write(f"Loss: {loss.item():.4f}\\n")
            f.write(f"Time: {epoch_time:.1f}s\\n")
            f.write(f"Device: {device}\\n")
        
        if hf_authenticated:
            upload_to_hf_repo(log_path)
    
    print("\\n✅ Training completed!")
    print("📁 Files saved locally:")
    print(f"   Samples: {len(list(Path('samples').glob('*.png')))}")
    print(f"   Checkpoints: {len(list(Path('checkpoints').glob('*.pth')))}")
    print(f"   Logs: {len(list(Path('logs').glob('*.txt')))}")

if __name__ == "__main__":
    train_monox_model()