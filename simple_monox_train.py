#!/usr/bin/env python3
"""
Simple MonoX Training Script for RunPod
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_dataset(config):
    class ImageDataset:
        def __init__(self, root, resolution):
            self.root = root
            self.resolution = resolution
            self.images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
            print(f"Found {len(self.images)} images")
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1) * 2.0 - 1.0
            return image
    
    return ImageDataset(config['dataset']['root'], config['dataset']['resolution'])

def create_model(config):
    class SimpleGenerator(nn.Module):
        def __init__(self, z_dim, resolution):
            super().__init__()
            self.fc = nn.Linear(z_dim, 512 * 4 * 4)
            self.conv_layers = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.BatchNorm2d(16), nn.ReLU(),
                nn.ConvTranspose2d(16, 3, 4, 2, 1),
                nn.Tanh()
            )
        
        def forward(self, z):
            x = self.fc(z)
            x = x.view(-1, 512, 4, 4)
            return self.conv_layers(x)
    
    return SimpleGenerator(config['model']['z_dim'], config['model']['resolution'])

def main():
    # Load config
    config = load_config('configs/monox-1024.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['output']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['output']['samples_dir'], exist_ok=True)
    os.makedirs(config['output']['tb_dir'], exist_ok=True)
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config['dataset']['batch_size'], shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} images")
    
    # Create model
    print("Creating model...")
    model = create_model(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and writer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    writer = SummaryWriter(config['output']['tb_dir'])
    
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        model.train()
        total_loss = 0
        criterion = nn.MSELoss()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            z = torch.randn(batch_size, config['model']['z_dim'], device=device)
            fake_images = model(z)
            
            loss = criterion(fake_images, real_images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 100 == 0:
                writer.add_scalar('Loss/Train', loss.item(), epoch * len(dataloader) + batch_idx)
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(config['output']['checkpoints_dir'], f'checkpoint_epoch_{epoch}.pkl')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save samples
        if epoch % config['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                z = torch.randn(4, config['model']['z_dim'], device=device)
                samples = model(z)
                samples = (samples + 1) / 2
                for i, sample in enumerate(samples):
                    img = sample.cpu().permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(os.path.join(config['output']['samples_dir'], f'sample_epoch_{epoch}_{i}.png'))
            model.train()
    
    print("Training completed!")
    writer.close()

if __name__ == '__main__':
    main()