#!/usr/bin/env python3
"""
MonoX L4 GPU Training with HF Integration
Complete training script with authentication, checkpoint resuming, and optimized saves
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from pathlib import Path
from huggingface_hub import HfApi, login, upload_file, list_repo_files
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

print("=" * 50)
print(f"===== Application Startup at {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
print("=" * 50)

# L4 Generator for 1024px
class L4Generator1024(nn.Module):
    def __init__(self, noise_dim=512):
        super().__init__()
        self.noise_dim = noise_dim
        
        # Progressive upsampling layers
        self.fc = nn.Linear(noise_dim, 4*4*1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)  # 4x4 -> 8x8
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)   # 8x8 -> 16x16
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)   # 16x16 -> 32x32
        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)    # 32x32 -> 64x64
        self.conv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1)     # 64x64 -> 128x128
        self.conv6 = nn.ConvTranspose2d(32, 16, 4, 2, 1)     # 128x128 -> 256x256
        self.conv7 = nn.ConvTranspose2d(16, 8, 4, 2, 1)      # 256x256 -> 512x512
        self.conv8 = nn.ConvTranspose2d(8, 3, 4, 2, 1)       # 512x512 -> 1024x1024
        
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(16)
        self.bn8 = nn.BatchNorm2d(8)
        
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        
    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(-1, 1024, 4, 4)
        
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        
        x = self.relu(self.bn3(x))
        x = self.conv3(x)
        
        x = self.relu(self.bn4(x))
        x = self.conv4(x)
        
        x = self.relu(self.bn5(x))
        x = self.conv5(x)
        
        x = self.relu(self.bn6(x))
        x = self.conv6(x)
        
        x = self.relu(self.bn7(x))
        x = self.conv7(x)
        
        x = self.relu(self.bn8(x))
        x = self.conv8(x)
        
        return self.tanh(x)

def setup_hf_auth():
    """Setup HF authentication using Space secret."""
    try:
        hf_token = os.environ.get('token')  # HF Space secret name
        if not hf_token:
            logger.warning("❌ No HF token found in environment")
            return False
        
        login(token=hf_token)
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"✅ Already authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        logger.error(f"❌ HF authentication failed: {e}")
        return False

def create_hf_model_repo():
    """Create or verify HF model repository exists."""
    try:
        api = HfApi()
        repo_id = "lukua/monox-model"
        
        # Check if repo exists
        try:
            api.repo_info(repo_id, repo_type="model")
            logger.info("✅ HF model repository exists")
            return True
        except:
            # Create repo if it doesn't exist
            api.create_repo(repo_id, repo_type="model", private=True)
            logger.info(f"✅ Created HF model repository: {repo_id}")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to create/access repository: {e}")
        return False

def find_latest_checkpoint():
    """Find the latest checkpoint from local and HF, return the absolute latest."""
    local_ckpts = []
    hf_ckpts = []
    
    # Check local checkpoints
    local_dir = Path("checkpoints")
    if local_dir.exists():
        for pattern in ["l4_generator_epoch_*.pth", "monox_checkpoint_epoch_*.pth"]:
            for ckpt in local_dir.glob(pattern):
                try:
                    epoch = int(ckpt.stem.split('_')[-1])
                    local_ckpts.append((epoch, str(ckpt)))
                except:
                    continue
    
    # Check HF checkpoints
    try:
        api = HfApi()
        files = list_repo_files("lukua/monox-model", repo_type="model")
        for file in files:
            if file.startswith("checkpoints/") and file.endswith(".pth"):
                try:
                    if "l4_generator_epoch_" in file:
                        epoch = int(file.split("l4_generator_epoch_")[1].split(".")[0])
                    elif "monox_checkpoint_epoch_" in file:
                        epoch = int(file.split("monox_checkpoint_epoch_")[1].split(".")[0])
                    else:
                        continue
                    hf_ckpts.append((epoch, file))
                except:
                    continue
    except Exception as e:
        logger.warning(f"⚠️ Could not check HF checkpoints: {e}")
    
    # Find absolute latest
    all_ckpts = local_ckpts + hf_ckpts
    if not all_ckpts:
        return None, 0
    
    latest_epoch, latest_path = max(all_ckpts, key=lambda x: x[0])
    
    # If it's an HF checkpoint, download it
    if latest_path.startswith("checkpoints/"):
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id="lukua/monox-model",
                filename=latest_path,
                local_dir=".",
                token=os.environ.get('token')
            )
            logger.info(f"📥 Downloaded HF checkpoint: {Path(latest_path).name} (epoch {latest_epoch})")
            return local_path, latest_epoch
        except Exception as e:
            logger.warning(f"⚠️ Failed to download HF checkpoint: {e}")
            return None, 0
    
    logger.info(f"📥 Found newer HF checkpoint: {latest_path} (epoch {latest_epoch})")
    return latest_path, latest_epoch

def load_checkpoint(generator, optimizer, scaler, ckpt_path):
    """Load checkpoint and return epoch number."""
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"✅ Loaded checkpoint epoch={epoch}, loss={loss:.6f}")
        return epoch
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return 0

def upload_sample_to_hf(sample_path):
    """Upload sample to HF model repository."""
    try:
        upload_file(
            path_or_fileobj=sample_path,
            path_in_repo=f"samples/{Path(sample_path).name}",
            repo_id="lukua/monox-model",
            token=os.environ.get('token'),
            repo_type="model"
        )
    except Exception as e:
        raise Exception(f"Sample upload failed: {e}")

def upload_checkpoint_to_hf(ckpt_path):
    """Upload checkpoint to HF model repository."""
    try:
        upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"checkpoints/{Path(ckpt_path).name}",
            repo_id="lukua/monox-model",
            token=os.environ.get('token'),
            repo_type="model"
        )
    except Exception as e:
        raise Exception(f"Checkpoint upload failed: {e}")

def upload_log_to_hf(log_path):
    """Upload log to HF model repository."""
    try:
        upload_file(
            path_or_fileobj=log_path,
            path_in_repo=f"logs/{Path(log_path).name}",
            repo_id="lukua/monox-model",
            token=os.environ.get('token'),
            repo_type="model"
        )
    except Exception as e:
        raise Exception(f"Log upload failed: {e}")

def main():
    """Main training function."""
    logger.info("🎯 MonoX L4 GPU Training Auto-Starter (Resume Enabled)")
    logger.info("=" * 50)
    
    # GPU Detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        logger.info(f"🚀 GPU Detected: {gpu_name}")
        logger.info(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        logger.info(f"⚡ CUDA Version: {cuda_version}")
    else:
        logger.warning("⚠️ No GPU detected - training will be very slow")
    
    # ---------- Training ----------
    def start_l4_1024px_training():
        logger.info("🎯 L4 1024px Training Starting...")
        
        Path("samples").mkdir(exist_ok=True)
        Path("checkpoints").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        setup_hf_auth()
        create_hf_model_repo()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = L4Generator1024().to(device)
        optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
        
        # Resume
        ckpt_path, start_epoch = find_latest_checkpoint()
        if ckpt_path:
            start_epoch = load_checkpoint(generator, optimizer, scaler, ckpt_path)
            logger.info(f"🚀 Resuming training from epoch {start_epoch}")
        else:
            logger.info("🆕 Starting fresh training from epoch 0")
            start_epoch = 0
        
        logger.info("🚀 Starting L4 1024px training loop...")
        max_epochs = 50_000
        sample_every = 100_000      # 10x less frequent
        ckpt_every = 1_000_000      # 10x less frequent  
        log_every = 50_000          # 10x less frequent
        
        # FIXED: Start from start_epoch + 1 to avoid re-saving the same epoch
        for epoch in range(start_epoch + 1, max_epochs):
            batch_size = 2
            noise = torch.randn(batch_size, 512, device=device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                fake_images = generator(noise)
                loss = -torch.mean(torch.std(fake_images.view(fake_images.size(0), -1), dim=1))
                loss += 0.01 * torch.mean(torch.abs(fake_images))
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Log progress every 10,000 epochs (more frequent for monitoring)
            if epoch % 10_000 == 0:
                mem_gb = (torch.cuda.memory_allocated() / 1024**3) if device.type == "cuda" else 0.0
                logger.info(f"🎨 L4 Epoch {epoch}, Loss: {loss.item():.4f}, GPU Mem: {mem_gb:.1f}GB")
            
            # Save sample every 100,000 epochs (much less frequent)
            if epoch % sample_every == 0:
                sample_path = f"samples/l4_1024px_epoch_{epoch:05d}.png"
                save_image(fake_images, sample_path, normalize=True, nrow=1)
                logger.info(f"💾 Saved 1024px sample: {sample_path}")
                try:
                    upload_sample_to_hf(sample_path)
                    logger.info("📤 Sample uploaded to HF")
                except Exception as e:
                    logger.warning(f"⚠️ Sample upload failed: {e}")
            
            # Save checkpoint every 1,000,000 epochs (much less frequent)
            if epoch % ckpt_every == 0 and epoch > 0:
                ckpt = f"checkpoints/l4_generator_epoch_{epoch:05d}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(loss.item()),
                    "scaler_state_dict": scaler.state_dict(),
                }, ckpt)
                logger.info(f"💾 Saved checkpoint: {ckpt}")
                try:
                    upload_checkpoint_to_hf(ckpt)
                    logger.info("📤 Checkpoint uploaded to HF")
                except Exception as e:
                    logger.warning(f"⚠️ Checkpoint upload failed: {e}")
            
            # Save log every 50,000 epochs (much less frequent)
            if epoch % log_every == 0:
                log_path = f"logs/training_log_epoch_{epoch:05d}.txt"
                content = f"""L4 GPU Training Log
Epoch: {epoch}
Loss: {loss.item():.6f}
GPU Memory: {(torch.cuda.memory_allocated()/1024**3 if device.type=='cuda' else 0):.2f}GB
Model: L4Generator1024
Resolution: 1024x1024
Batch Size: 2
Mixed Precision: {'Enabled' if device.type=='cuda' else 'Disabled'}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                Path(log_path).write_text(content)
                logger.info(f"📝 Saved training log: {log_path}")
                try:
                    upload_log_to_hf(log_path)
                    logger.info("📤 Log uploaded to HF")
                except Exception as e:
                    logger.warning(f"⚠️ Log upload failed: {e}")
            
            if epoch % 100 == 0:
                time.sleep(0.01)
    
    # Start training
    start_l4_1024px_training()

if __name__ == "__main__":
    main()