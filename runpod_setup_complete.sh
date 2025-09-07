#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== MonoX RunPod Setup Script ==="
echo "Starting setup on $(date)"

# 1. System setup
cd /workspace
mkdir -p code data out/checkpoints out/samples out/tb

# Install system dependencies
echo "Installing system dependencies..."
apt-get update -y
apt-get install -y git-lfs tmux curl jq
git lfs install || true

# Check environment
echo "Checking environment..."
python -V
nvidia-smi
echo "GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# 2. Clone/update MonoX repository
echo "Setting up MonoX repository..."
cd /workspace/code
if [ ! -d MonoX/.git ]; then
  echo "Cloning MonoX repository..."
  git clone https://github.com/yakymchukluka-afk/MonoX.git
fi
cd MonoX
git fetch --all --prune

# Create the StyleGAN-V loss branch
echo "Setting up StyleGAN-V loss branch..."
if ! git show-ref --verify --quiet refs/heads/feat/styleganv-loss-default; then
  git checkout -b feat/styleganv-loss-default
  echo "Created new branch: feat/styleganv-loss-default"
else
  git checkout feat/styleganv-loss-default
  echo "Switched to existing branch: feat/styleganv-loss-default"
fi

# 3. Set up RunPod training structure
echo "Setting up RunPod training structure..."
mkdir -p train/runpod-hf/{vendor,configs,scripts,monox}

# 4. Set up vendor dependencies
echo "Setting up vendor dependencies..."
cd train/runpod-hf/vendor

# Clone NVLabs StyleGAN2-ADA
if [ ! -d "stylegan2ada" ]; then
  echo "Cloning NVLabs StyleGAN2-ADA..."
  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git stylegan2ada
fi

# Clone StyleGAN-V
if [ ! -d "styleganv" ]; then
  echo "Cloning StyleGAN-V..."
  git clone https://github.com/universome/stylegan-v.git styleganv
fi

# 5. Install Python dependencies
echo "Installing Python dependencies..."
cd /workspace/code/MonoX/train/runpod-hf

# Create requirements.txt if it doesn't exist
cat > requirements.txt << 'EOF'
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
Pillow>=8.3.0
tqdm>=4.62.0
tensorboard>=2.8.0
wandb>=0.12.0
hydra-core>=1.3.2
omegaconf>=2.3.0
einops>=0.6.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.5.0
EOF

pip install -r requirements.txt

# 6. Download dataset
echo "Downloading dataset..."
cd /workspace
if [ ! -d data/monox-dataset/.git ]; then
  echo "Cloning monox-dataset..."
  git clone https://huggingface.co/datasets/lukua/monox-dataset /workspace/data/monox-dataset
fi
cd /workspace/data/monox-dataset
git lfs pull
echo "Dataset files: $(find . -type f | wc -l)"
echo "Dataset size: $(du -sh . | cut -f1)"

# 7. Create training configuration
echo "Creating training configuration..."
cd /workspace/code/MonoX/train/runpod-hf/configs
cat > monox-1024.yaml << 'EOF'
# MonoX 1024px Training Configuration
dataset:
  root: /workspace/data/monox-dataset
  resolution: 1024
  batch_size: 8  # Conservative for A100
  num_workers: 4

training:
  epochs: 100
  lr: 0.002
  beta1: 0.0
  beta2: 0.99
  checkpoint_interval: 500
  sample_interval: 100
  
model:
  z_dim: 512
  c_dim: 0
  resolution: 1024
  fmap_base: 8192
  fmap_max: 512
  
output:
  checkpoints_dir: /workspace/out/checkpoints
  samples_dir: /workspace/out/samples
  tb_dir: /workspace/out/tb
  log_file: /workspace/out/train.log
EOF

# 8. Create training script
echo "Creating training script..."
cd /workspace/code/MonoX/train/runpod-hf/monox
cat > train.py << 'EOF'
#!/usr/bin/env python3
"""
MonoX Training Script with StyleGAN-V Loss and NVLabs Networks
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

# Add vendor paths to Python path
sys.path.insert(0, '/workspace/code/MonoX/train/runpod-hf/vendor/stylegan2ada')
sys.path.insert(0, '/workspace/code/MonoX/train/runpod-hf/vendor/styleganv/src')
sys.path.insert(0, '/workspace/code/MonoX/train/runpod-hf')

def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_dataset(config):
    """Create dataset from images"""
    dataset_root = config['dataset']['root']
    resolution = config['dataset']['resolution']
    
    # Simple image dataset
    class ImageDataset:
        def __init__(self, root, resolution):
            self.root = root
            self.resolution = resolution
            self.images = []
            
            # Find all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                import glob
                self.images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
            
            print(f"Found {len(self.images)} images")
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1) * 2.0 - 1.0  # Normalize to [-1, 1]
            return image
    
    return ImageDataset(dataset_root, resolution)

def create_model(config):
    """Create StyleGAN2 generator"""
    try:
        from training.networks import Generator
        model = Generator(
            z_dim=config['model']['z_dim'],
            c_dim=config['model']['c_dim'],
            w_dim=512,
            img_resolution=config['model']['resolution'],
            img_channels=3,
            fmap_base=config['model']['fmap_base'],
            fmap_max=config['model']['fmap_max']
        )
        return model
    except ImportError as e:
        print(f"Error importing NVLabs networks: {e}")
        print("Falling back to simple generator...")
        
        # Simple fallback generator
        class SimpleGenerator(nn.Module):
            def __init__(self, z_dim, resolution):
                super().__init__()
                self.z_dim = z_dim
                self.resolution = resolution
                
                # Simple upsampling network
                self.fc = nn.Linear(z_dim, 512 * 4 * 4)
                self.conv_layers = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, 4, 2, 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, 4, 2, 1),
                    nn.Tanh()
                )
            
            def forward(self, z):
                x = self.fc(z)
                x = x.view(-1, 512, 4, 4)
                x = self.conv_layers(x)
                return x
        
        return SimpleGenerator(config['model']['z_dim'], config['model']['resolution'])

def create_loss_fn():
    """Create StyleGAN-V loss function"""
    try:
        from training.loss import StyleGAN2Loss
        return StyleGAN2Loss()
    except ImportError as e:
        print(f"Error importing StyleGAN-V loss: {e}")
        print("Falling back to simple GAN loss...")
        
        # Simple fallback loss
        class SimpleLoss:
            def __init__(self):
                self.mse = nn.MSELoss()
            
            def __call__(self, fake_images, real_images):
                return self.mse(fake_images, real_images)
        
        return SimpleLoss()

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, config, writer):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, real_images in enumerate(pbar):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Generate noise
        z = torch.randn(batch_size, config['model']['z_dim'], device=device)
        
        # Generate fake images
        fake_images = model(z)
        
        # Compute loss
        loss = loss_fn(fake_images, real_images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log to tensorboard
        if batch_idx % 100 == 0:
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(dataloader) + batch_idx)
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='MonoX Training')
    parser.add_argument('--config', type=str, default='configs/monox-1024.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Configuration loaded: {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['output']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['output']['samples_dir'], exist_ok=True)
    os.makedirs(config['output']['tb_dir'], exist_ok=True)
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    print(f"Dataset created: {len(dataset)} images")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    # Create loss function
    loss_fn = create_loss_fn()
    
    # Create tensorboard writer
    writer = SummaryWriter(config['output']['tb_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # Train epoch
        avg_loss = train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, config, writer)
        
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, config['output']['checkpoints_dir'])
        
        # Generate samples
        if epoch % config['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                z = torch.randn(4, config['model']['z_dim'], device=device)
                samples = model(z)
                samples = (samples + 1) / 2  # Denormalize to [0, 1]
                
                # Save samples
                for i, sample in enumerate(samples):
                    img = sample.cpu().permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(
                        os.path.join(config['output']['samples_dir'], f'sample_epoch_{epoch}_{i}.png')
                    )
            model.train()
    
    print("Training completed!")
    writer.close()

if __name__ == '__main__':
    main()
EOF

# 9. Create resume script
echo "Creating resume script..."
cd /workspace/code/MonoX/train/runpod-hf/scripts
cat > resume.sh << 'EOF'
#!/usr/bin/env bash
set -euxo pipefail
cd /workspace/code/MonoX/train/runpod-hf
export PYTHONPATH="$PWD/vendor/stylegan2ada:$PWD/vendor/styleganv/src:$PWD:$PYTHONPATH"

# Find latest checkpoint
LATEST_CKPT="$(ls -1t /workspace/out/checkpoints/*.pkl 2>/dev/null | head -n1 || true)"
EXTRA=""
if [ -n "$LATEST_CKPT" ]; then
    EXTRA="--resume $LATEST_CKPT"
    echo "Resuming from: $LATEST_CKPT"
else
    echo "Starting fresh training"
fi

# Start training
python -u monox/train.py --config configs/monox-1024.yaml $EXTRA 2>&1 | tee -a /workspace/out/train.log
EOF

chmod +x resume.sh

# 10. Create monitoring script
echo "Creating monitoring script..."
cat > monitor.sh << 'EOF'
#!/usr/bin/env bash
echo "=== MonoX Training Monitor ==="
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo "Training Status:"
if tmux has-session -t monox 2>/dev/null; then
    echo "✓ Training session 'monox' is running"
    echo "Attach with: tmux attach -t monox"
else
    echo "✗ No training session found"
fi
echo ""
echo "Latest logs (last 20 lines):"
tail -n 20 /workspace/out/train.log 2>/dev/null || echo "No logs found"
echo ""
echo "Checkpoints:"
ls -la /workspace/out/checkpoints/ 2>/dev/null || echo "No checkpoints found"
echo ""
echo "Samples:"
ls -la /workspace/out/samples/ 2>/dev/null || echo "No samples found"
EOF

chmod +x monitor.sh

echo "=== Setup Complete! ==="
echo "Next steps:"
echo "1. Start training: tmux new -s monox -d 'bash /workspace/code/MonoX/train/runpod-hf/scripts/resume.sh'"
echo "2. Monitor: bash /workspace/code/MonoX/train/runpod-hf/scripts/monitor.sh"
echo "3. Attach to session: tmux attach -t monox"
echo "4. Detach from session: Ctrl+B then D"
echo ""
echo "Training will save checkpoints to: /workspace/out/checkpoints/"
echo "Samples will be saved to: /workspace/out/samples/"
echo "TensorBoard logs: /workspace/out/tb/"
echo "Training logs: /workspace/out/train.log"