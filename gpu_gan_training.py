#!/usr/bin/env python3
"""
MonoX GPU-Optimized GAN Training
High-performance training with GPU acceleration
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
from huggingface_hub import HfApi, login
from PIL import Image
import torchvision.utils as vutils

# GPU Configuration
def setup_gpu():
    """Setup GPU with optimal settings."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU Available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimize GPU settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return device
    else:
        print("⚠️  GPU not available, using CPU")
        return torch.device('cpu')

# Enhanced Generator for GPU
class GPUGenerator(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=3):
        super(GPUGenerator, self).__init__()
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

# Enhanced Discriminator for GPU
class GPUDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(GPUDiscriminator, self).__init__()
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

def train_gpu_gan():
    """Main GPU training function."""
    print("🚀 Starting GPU-Optimized MonoX Training")
    print("=" * 50)
    
    # Setup
    device = setup_gpu()
    
    # Training parameters - optimized for GPU
    batch_size = 32 if device.type == 'cuda' else 8  # Larger batches on GPU
    image_size = 128  # Start with 128x128 for speed
    nz = 100  # Noise vector size
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5
    
    print(f"🎯 Training Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {lr}")
    
    # Create directories
    Path("previews").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Data loading with GPU optimization
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset("lukua/monox-dataset", split="train")
        print(f"✅ Loaded dataset: {len(dataset)} images")
        
        # Convert HF dataset to PyTorch dataset
        class HFDataset(torch.utils.data.Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.dataset = hf_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                image = item['image']
                if self.transform:
                    image = self.transform(image)
                return image
        
        pytorch_dataset = HFDataset(dataset, transform=transform)
        dataloader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        print(f"✅ DataLoader created with {len(dataloader)} batches")
        
    except Exception as e:
        print(f"⚠️  Dataset loading failed: {e}")
        print("   Creating dummy dataset for testing...")
        # Fallback to dummy data
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return torch.randn(3, image_size, image_size)
        
        pytorch_dataset = DummyDataset()
        dataloader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=True)
    
    # Create models
    netG = GPUGenerator(nz=nz).to(device)
    netD = GPUDiscriminator().to(device)
    
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
    
    print(f"\n🎨 Starting training loop...")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train on real data
        for i, real_data in enumerate(dataloader):
            if i >= 10:  # Limit batches per epoch for speed
                break
                
            real_data = real_data.to(device)
            batch_size_current = real_data.size(0)
            
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
            sample_path = f"previews/gpu_samples_epoch_{epoch+1:04d}.png"
            vutils.save_image(fake_samples, sample_path, normalize=True, nrow=4)
        
        epoch_time = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1:02d}/{num_epochs} - {epoch_time:.1f}s - Sample: {sample_path}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/gpu_checkpoint_epoch_{epoch+1:04d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Upload checkpoint if HF_TOKEN is available
            upload_file_to_hf(checkpoint_path)
        
        # Upload preview every epoch
        upload_file_to_hf(sample_path)

def upload_file_to_hf(file_path):
    """Upload file to HuggingFace model repo."""
    try:
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print("⚠️  HF_TOKEN not found, skipping upload")
            return
        
        from huggingface_hub import upload_file
        
        # Upload to lukua/monox model repo
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,  # Keep same relative path structure
            repo_id="lukua/monox",
            repo_type="model",
            token=hf_token
        )
        print(f"📤 Uploaded: {file_path} to lukua/monox")
        
    except Exception as e:
        print(f"❌ Upload failed for {file_path}: {e}")

if __name__ == "__main__":
    train_gpu_gan()