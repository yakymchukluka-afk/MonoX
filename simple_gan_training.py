#!/usr/bin/env python3
"""
Simple GAN Training for MonoX
Direct PyTorch implementation that will definitely work for fresh training.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """Simple image dataset for GAN training."""
    
    def __init__(self, image_dir, transform=None, image_size=512):
        self.image_dir = Path(image_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Find all image files
        self.image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_files.extend(list(self.image_dir.glob(ext)))
        
        logger.info(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

class HFMonoxDataset(Dataset):
    """HF dataset wrapper for lukua/monox-dataset."""
    def __init__(self, split: str = "train", image_size: int = 512):
        self.ds = load_dataset("lukua/monox-dataset", split=split)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample.get("image", sample)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        return self.transform(img)

class Generator(nn.Module):
    """Simple Generator network."""
    
    def __init__(self, nz=100, ngf=64, nc=3, img_size=512):
        super(Generator, self).__init__()
        self.img_size = img_size
        
        # Calculate the initial size for the first layer
        init_size = img_size // 16  # For 512: 32, for 1024: 64
        
        self.init_size = init_size
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 8 * init_size * init_size))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """Simple Discriminator network."""
    
    def __init__(self, nc=3, ndf=64, img_size=512):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(nc, ndf, bn=False),
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            *discriminator_block(ndf * 4, ndf * 8),
        )
        
        # Calculate the size after convolutions
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(ndf * 8 * ds_size * ds_size, 1), nn.Sigmoid())
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def upload_to_model_repo(file_path: str) -> bool:
    """Upload to HF model repo securely."""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        return False
    
    try:
        from huggingface_hub import upload_file
        
        file_name = os.path.basename(file_path)
        if file_path.endswith(('.pkl', '.pt', '.pth', '.ckpt')):
            repo_path = f"checkpoints/{file_name}"
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            repo_path = f"previews/{file_name}"
        elif file_path.endswith('.log'):
            repo_path = f"logs/{file_name}"
        else:
            repo_path = file_name
        
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_path,
            repo_id="lukua/monox",
            token=hf_token,
            repo_type="model"
        )
        
        logger.info(f"✅ Uploaded: {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False

def save_samples(generator, device, epoch, output_dir):
    """Generate and save sample images."""
    generator.eval()
    with torch.no_grad():
        # Generate samples
        noise = torch.randn(16, 100, device=device)
        fake_images = generator(noise)
        
        # Convert to PIL images and save
        fake_images = (fake_images + 1) / 2  # Denormalize
        fake_images = fake_images.clamp(0, 1)
        
        # Create grid
        from torchvision.utils import make_grid, save_image
        grid = make_grid(fake_images, nrow=4, normalize=False)
        
        # Save image
        sample_path = output_dir / f"samples_epoch_{epoch:04d}.png"
        save_image(grid, sample_path)
        
        logger.info(f"📸 Saved samples: {sample_path.name}")
        return str(sample_path)

def train_simple_gan():
    """Train a simple GAN on the MonoX dataset."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Using device: {device}")
    
    # Hyperparameters
    img_size = 512  # Reduced for faster training
    batch_size = 4 if device.type == "cpu" else 16
    lr = 0.0002
    epochs = 50  # 50 epochs for good results
    
    # Create dataset (HF dataset for CPU or GPU)
    try:
        dataset = HFMonoxDataset(image_size=img_size)
        logger.info(f"✅ Loaded HF dataset: lukua/monox-dataset ({len(dataset)} images)")
    except Exception as e:
        logger.error(f"⚠️  Failed to load HF dataset: {e}")
        logger.info("   Falling back to local /workspace/dataset if available...")
        dataset = ImageDataset("/workspace/dataset", image_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # No multiprocessing to avoid shared memory issues
    
    # Create models
    generator = Generator(img_size=img_size).to(device)
    discriminator = Discriminator(img_size=img_size).to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Create output directories (relative to Space working dir)
    output_dir = Path("training_output")
    checkpoints_dir = Path("checkpoints")
    previews_dir = Path("previews")
    
    for directory in [output_dir, checkpoints_dir, previews_dir]:
        directory.mkdir(exist_ok=True)
    
    logger.info("🎨 Starting Simple GAN Training")
    logger.info(f"📊 Dataset: {len(dataset)} images")
    logger.info(f"🖥️ Device: {device}")
    logger.info(f"📏 Image Size: {img_size}x{img_size}")
    logger.info(f"🔢 Batch Size: {batch_size}")
    logger.info(f"📈 Epochs: {epochs}")
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        for i, real_imgs in enumerate(dataloader):
            batch_size_actual = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size_actual, 1, device=device, requires_grad=False)
            fake = torch.zeros(batch_size_actual, 1, device=device, requires_grad=False)
            
            # Train Generator
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size_actual, 100, device=device)
            gen_imgs = generator(z)
            
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} Batch {i}/{len(dataloader)} "
                          f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item()
            }
            
            ckpt_path = checkpoints_dir / f"monox_checkpoint_epoch_{epoch+1:04d}.pth"
            torch.save(checkpoint, ckpt_path)
            logger.info(f"💾 Saved checkpoint: {ckpt_path.name}")
            
            # Upload checkpoint
            upload_to_model_repo(str(ckpt_path))
        
        # Generate samples every epoch
        sample_path = save_samples(generator, device, epoch + 1, previews_dir)
        upload_to_model_repo(sample_path)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    # Save final model
    final_model_path = checkpoints_dir / "monox_final_model.pth"
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'config': {
            'img_size': img_size,
            'epochs': epochs,
            'dataset_size': len(dataset)
        }
    }, final_model_path)
    
    logger.info(f"💾 Saved final model: {final_model_path.name}")
    upload_to_model_repo(str(final_model_path))
    
    return True

def main():
    """Main execution."""
    print("🎨 Simple MonoX GAN Training")
    print("🎯 Direct PyTorch implementation for reliable training")
    print("=" * 70)
    
    # Optional authentication (uploads enabled if token present)
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            from huggingface_hub import whoami
            user_info = whoami(token=hf_token)
            print(f"✅ Authenticated as: {user_info['name']}")
        except Exception as e:
            print(f"⚠️ Authentication failed, continuing without uploads: {e}")
            hf_token = None
    else:
        print("ℹ️ No HF_TOKEN set. Training will run; uploads disabled.")
    
    print("📤 Upload target: lukua/monox (uploads only if HF_TOKEN is set)")
    
    print("\\n🚀 Starting GAN training...")
    print("📝 Training will:")
    print("  • Train for 50 epochs")
    print("  • Save checkpoints every 5 epochs")
    print("  • Generate preview images each epoch")
    print("  • Upload everything to lukua/monox")
    
    # Start training
    try:
        success = train_simple_gan()
        
        if success:
            print("\\n🎉 Training completed successfully!")
            print("📁 Check lukua/monox for:")
            print("  • Checkpoints in /checkpoints")
            print("  • Preview images in /previews")
            print("  • Training logs in /logs")
            return 0
        else:
            print("\\n❌ Training failed")
            return 1
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())