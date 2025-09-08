#!/usr/bin/env python3
"""
MonoX Training with Hybrid Authentication
Uses both SSH key and HF token for maximum reliability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
import torchvision.utils as vutils
import logging
from monox_hybrid_auth import MonoXHybridAuth

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup GPU with optimal settings."""
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

class MonoXGenerator(nn.Module):
    """Enhanced generator for MonoX training."""
    def __init__(self, nz=100, ngf=128, nc=3):
        super(MonoXGenerator, self).__init__()
        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # State: (ngf*16) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 32 x 32
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) x 64 x 64
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State: (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

class MonoXDiscriminator(nn.Module):
    """Enhanced discriminator for MonoX training."""
    def __init__(self, nc=3, ndf=128):
        super(MonoXDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf) x 64 x 64
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 32 x 32
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 16 x 16
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 8 x 8
            
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*16) x 4 x 4
            
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # State: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def train_monox_with_hybrid_auth():
    """Main training function with hybrid authentication."""
    print("🎯 MonoX Training with Hybrid Authentication")
    print("=" * 60)
    
    # Setup authentication
    auth = MonoXHybridAuth()
    if not auth.setup_authentication():
        print("❌ Authentication setup failed")
        return False
    
    print(f"✅ Authentication method: {auth.auth_method}")
    
    # Setup GPU
    device, gpu_name, gpu_memory = setup_gpu()
    
    # Create directories
    Path("samples").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Training parameters
    batch_size = 32 if device.type == 'cuda' else 8
    image_size = 128
    nz = 100
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5
    
    print(f"\\n🎯 Training Configuration:")
    print(f"   Device: {device}")
    print(f"   GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f}GB")
    print(f"   Batch Size: {batch_size}")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {lr}")
    
    # Create models
    netG = MonoXGenerator(nz=nz).to(device)
    netD = MonoXDiscriminator().to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Fixed noise for consistent samples
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    
    print(f"\\n🎨 Starting training loop...")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training batches
        for i in range(10):  # 10 batches per epoch
            batch_size_current = min(batch_size, 16)
            
            # Create dummy data for this example
            real_data = torch.randn(batch_size_current, 3, image_size, image_size, device=device)
            
            # Train Discriminator
            netD.zero_grad()
            real_label = torch.ones(batch_size_current, device=device)
            fake_label = torch.zeros(batch_size_current, device=device)
            
            # Real data
            output_real = netD(real_data)
            errD_real = criterion(output_real, real_label)
            errD_real.backward()
            
            # Fake data
            noise = torch.randn(batch_size_current, nz, 1, 1, device=device)
            fake_data = netG(noise)
            output_fake = netD(fake_data.detach())
            errD_fake = criterion(output_fake, fake_label)
            errD_fake.backward()
            optimizerD.step()
            
            # Train Generator
            netG.zero_grad()
            output = netD(fake_data)
            errG = criterion(output, real_label)
            errG.backward()
            optimizerG.step()
        
        # Generate sample
        with torch.no_grad():
            fake_samples = netG(fixed_noise)
            sample_path = f"samples/monox_epoch_{epoch:05d}.png"
            vutils.save_image(fake_samples, sample_path, normalize=True, nrow=4)
        
        epoch_time = time.time() - epoch_start
        print(f"🎨 Epoch {epoch+1:02d}/{num_epochs}, Loss: {errG.item():.4f}, Time: {epoch_time:.1f}s")
        
        # Upload sample
        auth.upload_file(sample_path)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/monox_checkpoint_epoch_{epoch+1:05d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            # Upload checkpoint
            auth.upload_file(checkpoint_path)
        
        # Save training log
        log_path = f"logs/training_log_epoch_{epoch:05d}.txt"
        with open(log_path, 'w') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\\n")
            f.write(f"Generator Loss: {errG.item():.4f}\\n")
            f.write(f"Discriminator Loss: {(errD_real + errD_fake).item():.4f}\\n")
            f.write(f"Time: {epoch_time:.1f}s\\n")
            f.write(f"Device: {device}\\n")
            f.write(f"GPU: {gpu_name}\\n")
        
        # Upload log
        auth.upload_file(log_path)
    
    print("\\n✅ Training completed!")
    print("📁 Files saved locally:")
    print(f"   Samples: {len(list(Path('samples').glob('*.png')))}")
    print(f"   Checkpoints: {len(list(Path('checkpoints').glob('*.pth')))}")
    print(f"   Logs: {len(list(Path('logs').glob('*.txt')))}")
    
    return True

if __name__ == "__main__":
    train_monox_with_hybrid_auth()