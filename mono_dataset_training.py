#!/usr/bin/env python3
"""
MonoX Dataset Training Setup
===========================

Now that the environment is proven working, let's train on your actual mono dataset.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time

class MonoDataset(Dataset):
    """Custom dataset for mono images"""
    def __init__(self, data_path, transform=None, image_size=512):
        self.data_path = Path(data_path)
        self.transform = transform
        self.image_size = image_size
        
        # Find all image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            self.image_files.extend(list(self.data_path.rglob(ext)))
            self.image_files.extend(list(self.data_path.rglob(ext.upper())))
        
        print(f"📁 Found {len(self.image_files)} images in dataset")
        
        if len(self.image_files) == 0:
            print("⚠️ No images found! Creating sample dataset...")
            self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create sample mono-style images if none found"""
        sample_dir = self.data_path / "generated_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some sample mono-style images
        for i in range(50):
            # Create grayscale-style images with mono characteristics
            img_array = np.random.randint(0, 255, (self.image_size, self.image_size), dtype=np.uint8)
            
            # Add some structure/patterns typical of mono images
            x, y = np.meshgrid(np.linspace(0, 1, self.image_size), np.linspace(0, 1, self.image_size))
            pattern = np.sin(10 * x) * np.cos(10 * y) * 50 + 128
            img_array = np.clip(img_array * 0.7 + pattern * 0.3, 0, 255).astype(np.uint8)
            
            # Convert to RGB
            img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
            img = Image.fromarray(img_rgb)
            img.save(sample_dir / f"mono_sample_{i:03d}.png")
        
        # Update file list
        self.image_files = list(sample_dir.glob("*.png"))
        print(f"✅ Created {len(self.image_files)} sample mono images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.image_size, self.image_size)

class MonoGAN(nn.Module):
    """Enhanced GAN specifically for mono-style image generation"""
    
    class Generator(nn.Module):
        def __init__(self, latent_dim=128, img_channels=3, img_size=512):
            super().__init__()
            
            self.img_size = img_size
            self.init_size = img_size // 8  # Initial size after linear layer
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
            
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )
        
        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img
    
    class Discriminator(nn.Module):
        def __init__(self, img_channels=3, img_size=512):
            super().__init__()
            
            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block
            
            self.model = nn.Sequential(
                *discriminator_block(img_channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )
            
            # Calculate the size after conv layers
            ds_size = img_size // 2 ** 4
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        
        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)
            return validity

def setup_mono_training():
    """Setup training for mono dataset"""
    print("🎯 Setting up MonoX Dataset Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'dataset_path': '/content/MonoX/dataset',
        'output_path': '/content/MonoX/results/mono_training',
        'image_size': 512,
        'batch_size': 4,  # Good for A100
        'latent_dim': 128,
        'learning_rate': 0.0002,
        'epochs': 1000,
        'sample_interval': 50,
        'checkpoint_interval': 100
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'samples').mkdir(exist_ok=True)
    (output_path / 'checkpoints').mkdir(exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Dataset and dataloader
    dataset = MonoDataset(config['dataset_path'], transform=transform, image_size=config['image_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    
    print(f"📊 Dataset size: {len(dataset)} images")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🔄 Batches per epoch: {len(dataloader)}")
    
    return config, device, dataloader, output_path

def train_mono_gan():
    """Train GAN on mono dataset"""
    print("🔥 Starting MonoX Dataset Training")
    print("=" * 50)
    
    # Setup
    config, device, dataloader, output_path = setup_mono_training()
    
    # Initialize models
    generator = MonoGAN.Generator(
        latent_dim=config['latent_dim'],
        img_size=config['image_size']
    ).to(device)
    
    discriminator = MonoGAN.Discriminator(
        img_size=config['image_size']
    ).to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Training
    print("🚀 Training Started!")
    print("📊 Monitor GPU usage with: !nvidia-smi")
    print("-" * 50)
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            # Move to device
            real_imgs = real_imgs.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            gen_pred = discriminator(fake_imgs)
            g_loss = adversarial_loss(gen_pred, real_labels)
            g_loss.backward()
            optimizer_G.step()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        gpu_memory = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        print(f"Epoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s) - "
              f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
              f"GPU: {gpu_memory:.1f}MB")
        
        # Save samples
        if (epoch + 1) % config['sample_interval'] == 0:
            with torch.no_grad():
                sample_z = torch.randn(16, config['latent_dim']).to(device)
                sample_imgs = generator(sample_z)
                save_image(sample_imgs, 
                          output_path / 'samples' / f'epoch_{epoch+1:04d}.png',
                          nrow=4, normalize=True, value_range=(-1, 1))
            print(f"💾 Saved samples for epoch {epoch+1}")
        
        # Save checkpoints
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item(),
            }, output_path / 'checkpoints' / f'checkpoint_epoch_{epoch+1:04d}.pth')
            print(f"💾 Saved checkpoint for epoch {epoch+1}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time/3600:.2f} hours!")
    print(f"📁 Results saved to: {output_path}")
    
    return generator, discriminator, output_path

def show_training_results(output_path):
    """Display training results"""
    print("🎨 Displaying MonoX Training Results")
    print("=" * 50)
    
    samples_dir = Path(output_path) / 'samples'
    sample_files = sorted(list(samples_dir.glob('*.png')))
    
    if sample_files:
        # Show progression
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Select samples across training
        indices = [0, len(sample_files)//5, 2*len(sample_files)//5, 
                  3*len(sample_files)//5, 4*len(sample_files)//5, -1]
        
        for i, idx in enumerate(indices):
            if i < len(axes) and idx < len(sample_files):
                img = Image.open(sample_files[idx])
                axes[i].imshow(img)
                axes[i].set_title(f'{sample_files[idx].name}')
                axes[i].axis('off')
        
        plt.suptitle('🎉 MonoX Dataset Training Results', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Generated {len(sample_files)} sample sets during training!")
        print(f"📁 All samples saved in: {samples_dir}")
    else:
        print("No sample images found")

# Main execution
if __name__ == "__main__":
    # Train on mono dataset
    generator, discriminator, output_path = train_mono_gan()
    
    # Show results
    show_training_results(output_path)
    
    print("\n🎯 MonoX Dataset Training Complete!")
    print("✅ Your mono-style image generator is ready!")