#!/usr/bin/env python3
import os, sys, yaml, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import glob

def load_config():
    with open('configs/monox-1024.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_dataset(config):
    class ImageDataset:
        def __init__(self, root, resolution):
            self.images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
            print(f"Found {len(self.images)} images")
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            image = image.resize((1024, 1024), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1) * 2.0 - 1.0
            return image
    return ImageDataset(config['dataset']['root'], 1024)

def create_model():
    class SimpleGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 512 * 4 * 4)
            self.conv_layers = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Tanh()
            )
        def forward(self, z):
            x = self.fc(z)
            x = x.view(-1, 512, 4, 4)
            return self.conv_layers(x)
    return SimpleGenerator()

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('/workspace/out/checkpoints', exist_ok=True)
    os.makedirs('/workspace/out/samples', exist_ok=True)
    
    print("Creating dataset...")
    dataset = create_dataset(config)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} images")
    
    print("Creating model...")
    model = create_model().to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(10):
        print(f"Epoch {epoch}/10")
        model.train()
        total_loss = 0
        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            z = torch.randn(batch_size, 512, device=device)
            fake_images = model(z)
            loss = criterion(fake_images, real_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        if epoch % 2 == 0:
            checkpoint_path = f'/workspace/out/checkpoints/checkpoint_epoch_{epoch}.pkl'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                z = torch.randn(4, 512, device=device)
                samples = model(z)
                samples = (samples + 1) / 2
                for i, sample in enumerate(samples):
                    img = sample.cpu().permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(f'/workspace/out/samples/sample_epoch_{epoch}_{i}.png')
            model.train()
    
    print("Training completed!")

if __name__ == '__main__':
    main()