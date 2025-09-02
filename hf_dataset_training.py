#!/usr/bin/env python3
"""
MonoX Training with HuggingFace Dataset
Uses the lukua/monox-dataset for proper training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np
import os
from pathlib import Path
import time
from huggingface_hub import HfApi, login
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
Path("previews").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

class HFDataset:
    """Dataset wrapper for HuggingFace dataset."""
    
    def __init__(self, hf_dataset, transform=None, image_size=512):
        self.dataset = hf_dataset
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image

class MonoXGenerator(nn.Module):
    """Enhanced Generator for MonoX artwork."""
    
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
            
            # State: ngf x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 128 x 128
        )
    
    def forward(self, x):
        return self.main(x)

class MonoXDiscriminator(nn.Module):
    """Enhanced Discriminator for MonoX artwork."""
    
    def __init__(self, nc=3, ndf=128):
        super(MonoXDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: nc x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: ndf x 64 x 64
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
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

def create_sample_image(generator, epoch, device, num_samples=16):
    """Create and save a sample image."""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        fake_images = generator(noise)
        
        filename = f"previews/samples_epoch_{epoch:03d}.png"
        vutils.save_image(fake_images, filename, normalize=True, nrow=4, padding=2)
        logger.info(f"📸 Saved: {filename}")
    
    generator.train()

def upload_to_hf(api, repo_name, local_path, hf_path):
    """Upload file to HuggingFace repository."""
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=repo_name,
            repo_type="space"
        )
        logger.info(f"✅ Uploaded: {hf_path}")
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")

def train():
    """Main training function using HuggingFace dataset."""
    logger.info("🎨 MonoX Training with HuggingFace Dataset")
    logger.info("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Device: {device}")
    
    # Load HuggingFace dataset
    logger.info("📊 Loading HuggingFace dataset...")
    try:
        dataset = load_dataset("lukua/monox-dataset", split="train")
        logger.info(f"✅ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return False
    
    # Create dataset and dataloader
    hf_dataset = HFDataset(dataset, image_size=128)  # Smaller for faster training
    dataloader = DataLoader(hf_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # Initialize models
    generator = MonoXGenerator().to(device)
    discriminator = MonoXDiscriminator().to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    lr = 0.0002
    beta1 = 0.5
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Training parameters
    num_epochs = 50
    fixed_noise = torch.randn(16, 100, 1, 1, device=device)
    
    # HuggingFace API setup
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            api = HfApi(token=hf_token)
            logger.info("✅ HuggingFace authentication successful")
        except Exception as e:
            logger.warning(f"⚠️ HF authentication failed: {e}")
            api = None
    else:
        logger.warning("⚠️ No HF_TOKEN found")
        api = None
    
    logger.info("🚀 Starting training...")
    
    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # Update Discriminator
            discriminator.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            
            # Train with real data
            real_labels = torch.ones(batch_size, device=device)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            d_loss_real.backward()
            
            # Train with fake data
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = generator(noise)
            fake_labels = torch.zeros(batch_size, device=device)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss_fake.backward()
            
            optimizer_D.step()
            
            # Update Generator
            generator.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                          f"D Loss: {(d_loss_real + d_loss_fake).item():.4f} "
                          f"G Loss: {g_loss.item():.4f}")
        
        # Generate sample every epoch
        create_sample_image(generator, epoch + 1, device)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/monox_checkpoint_epoch_{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, checkpoint_path)
            logger.info(f"💾 Saved: {checkpoint_path}")
            
            # Upload to HF if available
            if api:
                try:
                    upload_to_hf(api, "lukua/monox", checkpoint_path, f"checkpoints/monox_checkpoint_epoch_{epoch+1:03d}.pth")
                except Exception as e:
                    logger.warning(f"⚠️ HF upload failed: {e}")
        
        # Upload latest sample to HF
        if api and (epoch + 1) % 5 == 0:
            try:
                sample_path = f"previews/samples_epoch_{epoch+1:03d}.png"
                upload_to_hf(api, "lukua/monox", sample_path, f"previews/samples_epoch_{epoch+1:03d}.png")
            except Exception as e:
                logger.warning(f"⚠️ Sample upload failed: {e}")
    
    logger.info("🎉 Training completed!")
    logger.info(f"📁 Generated {len(list(Path('previews').glob('*.png')))} samples")
    logger.info(f"💾 Saved {len(list(Path('checkpoints').glob('*.pth')))} checkpoints")
    
    return True

if __name__ == "__main__":
    try:
        success = train()
        if success:
            logger.info("✅ Training completed successfully!")
        else:
            logger.error("❌ Training failed!")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()