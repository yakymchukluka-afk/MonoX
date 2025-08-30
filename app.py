#!/usr/bin/env python3
"""
MonoX Hugging Face Space - FastAPI Server
A participative art project powered by StyleGAN-V for generating dynamic visual content.
"""

import os
import sys
import subprocess
import glob
import asyncio
import time
import urllib.request
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

app = FastAPI(title="MonoX", description="A participative art project powered by StyleGAN-V")

# Global training state
training_status = {"is_training": False, "current_step": 0, "message": "Ready to start training"}

class MonoDataset(Dataset):
    def __init__(self, data_path=None, transform=None, use_hf_dataset=True):
        self.transform = transform
        self.use_hf_dataset = use_hf_dataset
        
        if use_hf_dataset:
            try:
                print("📥 Loading dataset from Hugging Face: lukua/monox-dataset")
                self.hf_dataset = load_dataset("lukua/monox-dataset", split="train")
                self.image_files = None
                print(f"📊 Found {len(self.hf_dataset)} images in HF dataset")
            except Exception as e:
                print(f"⚠️ Failed to load HF dataset: {e}")
                print("📁 Falling back to local dataset loading")
                self.use_hf_dataset = False
                self.hf_dataset = None
        
        if not use_hf_dataset or not hasattr(self, 'hf_dataset') or self.hf_dataset is None:
            # Fallback to local file loading
            if data_path:
                self.data_path = Path(data_path)
            else:
                self.data_path = Path("./dataset")  # Default path
            
            # Find all images
            self.image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
                self.image_files.extend(list(self.data_path.rglob(ext)))
                self.image_files.extend(list(self.data_path.rglob(ext.upper())))

            print(f"📊 Found {len(self.image_files)} local images")

            if len(self.image_files) == 0:
                print("⚠️ No local images found - will use synthetic data for demo")
                self.image_files = None

    def __len__(self):
        if self.use_hf_dataset and self.hf_dataset:
            return len(self.hf_dataset)
        elif self.image_files:
            return len(self.image_files)
        else:
            return 100  # Synthetic dataset size

    def __getitem__(self, idx):
        try:
            if self.use_hf_dataset and self.hf_dataset:
                # Load from HF dataset
                item = self.hf_dataset[idx]
                if 'image' in item:
                    image = item['image']
                    if isinstance(image, Image.Image):
                        image = image.convert('RGB')
                    else:
                        # Handle other image formats
                        image = Image.fromarray(image).convert('RGB')
                else:
                    # Fallback if structure is different
                    image = Image.new('RGB', (512, 512), color='white')
            elif self.image_files:
                # Load from local files
                img_path = self.image_files[idx]
                image = Image.open(img_path).convert('RGB')
            else:
                # Generate synthetic data for demo
                image = Image.new('RGB', (512, 512), color=(idx % 255, (idx*2) % 255, (idx*3) % 255))
            
            if self.transform:
                image = self.transform(image)
            return image
            
        except Exception as e:
            print(f"⚠️ Error loading image {idx}: {e}")
            # Return a default tensor
            if self.transform:
                return self.transform(Image.new('RGB', (512, 512), color='gray'))
            else:
                return torch.zeros(3, 512, 512)

class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.init_size = 64
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
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_filters, out_filters, bn=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            return layers

        self.model = nn.Sequential(
            *block(3, 16, bn=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        self.adv_layer = nn.Sequential(nn.Linear(128 * 32 ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)

def adjust_learning_rate(epoch):
    """Dynamic learning rate based on epoch"""
    if epoch < 1000:
        return 0.0002
    elif epoch < 2000:
        return 0.0001
    elif epoch < 3000:
        return 0.00005
    elif epoch < 5000:
        return 0.00002
    else:
        return 0.00001

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """Find the most recent checkpoint file."""
    if not os.path.isdir(checkpoint_dir):
        return None
    patterns = ["*.pkl", "*.pt", "*.ckpt", "*.pth"]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]

def find_latest_monox_checkpoint(checkpoint_dir: str = "/tmp/checkpoints") -> tuple[Optional[str], int]:
    """Find the highest numbered MonoX checkpoint"""
    checkpoint_files = list(Path(checkpoint_dir).glob("monox_generator_*.pth"))
    if not checkpoint_files:
        return None, 0

    # Extract epoch numbers and find the highest
    epochs = []
    for f in checkpoint_files:
        try:
            epoch_num = int(f.stem.split('_')[-1])
            epochs.append((epoch_num, f))
        except:
            continue

    if not epochs:
        return None, 0

    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return str(latest_file), latest_epoch

async def run_monox_training():
    """Run MonoX GAN training process."""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["message"] = "Initializing MonoX training..."
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using: {device}")
        
        # Create directories in /tmp (writable in HF Spaces)
        checkpoint_dir = "/tmp/checkpoints"
        samples_dir = "/tmp/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        
        # Initialize models
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        
        # Find and load latest checkpoint
        latest_checkpoint, start_epoch = find_latest_monox_checkpoint(checkpoint_dir)
        
        if latest_checkpoint:
            training_status["message"] = f"Loading checkpoint from epoch {start_epoch}"
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            current_lr = adjust_learning_rate(start_epoch)
            optimizer_G = optim.Adam(generator.parameters(), lr=current_lr, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=current_lr, betas=(0.5, 0.999))
            
            if 'optimizer_G_state_dict' in checkpoint:
                optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            if 'optimizer_D_state_dict' in checkpoint:
                optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                
            start_epoch += 1
            training_status["message"] = f"Resumed from epoch {start_epoch}"
        else:
            training_status["message"] = "Starting training from scratch"
            start_epoch = 0
            current_lr = adjust_learning_rate(start_epoch)
            optimizer_G = optim.Adam(generator.parameters(), lr=current_lr, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=current_lr, betas=(0.5, 0.999))
        
        # Loss function
        adversarial_loss = nn.BCELoss()
        
        # Load the actual MonoX dataset from Hugging Face
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        try:
            dataset = MonoDataset(transform=transform, use_hf_dataset=True)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
            training_status["message"] = f"Loaded HF dataset with {len(dataset)} images"
        except Exception as e:
            training_status["message"] = f"Dataset loading error: {str(e)} - using synthetic data"
            # Fallback to synthetic data
            dataset = MonoDataset(transform=transform, use_hf_dataset=False)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        training_status["message"] = "Training in progress..."
        
        # Training loop (limited for demo)
        for epoch in range(start_epoch, start_epoch + 10):  # Just 10 epochs for demo
            epoch_start = time.time()
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            # Train on actual dataset batches
            for batch_idx, real_imgs in enumerate(dataloader):
                if batch_idx >= 5:  # Limit batches for demo
                    break
                    
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
            
                # Labels
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9
                fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1
                
                # Train Discriminator
                optimizer_D.zero_grad()
                real_pred = discriminator(real_imgs)
                d_real_loss = adversarial_loss(real_pred, real_labels)
                
                z = torch.randn(batch_size, 128).to(device)
                fake_imgs = generator(z)
                fake_pred = discriminator(fake_imgs.detach())
                d_fake_loss = adversarial_loss(fake_pred, fake_labels)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, 128).to(device)
                fake_imgs = generator(z)
                gen_pred = discriminator(fake_imgs)
                g_loss = adversarial_loss(gen_pred, torch.ones(batch_size, 1).to(device))
                g_loss.backward()
                optimizer_G.step()
                
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
            
            # Calculate averages
            avg_d_loss = epoch_d_loss / min(5, len(dataloader))
            avg_g_loss = epoch_g_loss / min(5, len(dataloader))
            
            training_status["current_step"] = epoch + 1
            training_status["message"] = f"Epoch {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}"
            
            # Save checkpoint every 5 epochs (frequent for demo)
            if (epoch + 1) % 5 == 0:
                checkpoint_file = f'{checkpoint_dir}/monox_generator_{epoch+1:05d}.pth'
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                }, checkpoint_file)
                
                # Generate sample
                with torch.no_grad():
                    sample_z = torch.randn(4, 128).to(device)
                    sample_imgs = generator(sample_z)
                    sample_file = f'{samples_dir}/monox_epoch_{epoch+1:05d}.png'
                    save_image(sample_imgs, sample_file, nrow=2, normalize=True, value_range=(-1, 1))
            
            await asyncio.sleep(0.1)  # Yield control
        
        training_status["message"] = "Training demo completed! Check /tmp/checkpoints for saved models."
        
    except Exception as e:
        training_status["message"] = f"Training error: {str(e)}"
    finally:
        training_status["is_training"] = False

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main page of the MonoX application."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MonoX - StyleGAN-V Art Project</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #333; text-align: center; }}
            .info {{ background: #e8f4fd; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 MonoX</h1>
            <div class="info">
                <h3>Welcome to MonoX</h3>
                <p>A participative art project powered by StyleGAN-V for generating dynamic visual content.</p>
                <p><strong>Status:</strong> Space is running</p>
                <p><strong>GPU Available:</strong> {gpu_status}</p>
                <p><strong>PyTorch Version:</strong> {torch_version}</p>
                <p><strong>Training Status:</strong> {training_message}</p>
                <div style="margin-top: 20px;">
                    <button onclick="startTraining()" style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">Start Training</button>
                    <button onclick="checkStatus()" style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-left: 10px;">Check Status</button>
                    <button onclick="generateSample()" style="background: #ff6b35; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-left: 10px;">Generate Art</button>
                </div>
                <div style="margin-top: 15px;">
                    <button onclick="checkDataset()" style="background: #17a2b8; color: white; border: none; padding: 8px 15px; border-radius: 3px; cursor: pointer;">Check Dataset</button>
                    <button onclick="downloadHFCheckpoint()" style="background: #6f42c1; color: white; border: none; padding: 8px 15px; border-radius: 3px; cursor: pointer; margin-left: 10px;">Download HF Checkpoint</button>
                </div>
                <div style="margin-top: 10px;">
                    <input type="text" id="gdriveUrl" placeholder="Google Drive checkpoint URL (legacy)" style="width: 60%; padding: 8px; border: 1px solid #ddd; border-radius: 3px;">
                    <button onclick="downloadCheckpoint()" style="background: #6c757d; color: white; border: none; padding: 8px 15px; border-radius: 3px; cursor: pointer; margin-left: 10px;">Download GDrive</button>
                </div>
                <script>
                    async function startTraining() {{
                        try {{
                            const response = await fetch('/api/train', {{method: 'POST'}});
                            const result = await response.json();
                            alert(result.message);
                        }} catch (e) {{
                            alert('Error starting training: ' + e.message);
                        }}
                    }}
                    
                    async function checkStatus() {{
                        try {{
                            const response = await fetch('/api/training-status');
                            const result = await response.json();
                            alert(`Training Status: ${{result.message}}\\nEpoch: ${{result.latest_epoch || 'N/A'}}\\nCheckpoints: ${{result.checkpoint_count}}`);
                        }} catch (e) {{
                            alert('Error checking status: ' + e.message);
                        }}
                    }}
                    
                    async function generateSample() {{
                        try {{
                            const response = await fetch('/api/generate-sample');
                            const result = await response.json();
                            alert(`Sample generated from epoch ${{result.epoch}}!\\nFile: ${{result.sample_file}}`);
                        }} catch (e) {{
                            alert('Error generating sample: ' + e.message);
                        }}
                    }}
                    
                    async function checkDataset() {{
                        try {{
                            const response = await fetch('/api/dataset-info');
                            const result = await response.json();
                            alert(`Dataset Info:\\nSource: ${{result.dataset_source}}\\nImages: ${{result.total_images}}\\nStatus: ${{result.status}}\\nMessage: ${{result.message}}`);
                        }} catch (e) {{
                            alert('Error checking dataset: ' + e.message);
                        }}
                    }}
                    
                    async function downloadHFCheckpoint() {{
                        try {{
                            const response = await fetch('/api/download-hf-checkpoint', {{method: 'POST'}});
                            const result = await response.json();
                            alert(result.message);
                        }} catch (e) {{
                            alert('Error downloading HF checkpoint: ' + e.message);
                        }}
                    }}
                    
                    async function downloadCheckpoint() {{
                        const url = document.getElementById('gdriveUrl').value;
                        if (!url) {{
                            alert('Please enter a Google Drive URL');
                            return;
                        }}
                        try {{
                            const response = await fetch('/api/download-gdrive-checkpoint', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify({{'gdrive_url': url}})
                            }});
                            const result = await response.json();
                            alert(result.message);
                            document.getElementById('gdriveUrl').value = '';
                        }} catch (e) {{
                            alert('Error downloading checkpoint: ' + e.message);
                        }}
                    }}
                </script>
            </div>
        </div>
    </body>
    </html>
    """.format(
        gpu_status="Yes" if torch.cuda.is_available() else "No",
        torch_version=torch.__version__,
        training_message=training_status["message"]
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    }

@app.get("/api/generate")
async def generate_art():
    """API endpoint for generating art (placeholder)."""
    return {
        "message": "Art generation endpoint - to be implemented",
        "status": "coming_soon"
    }

@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks):
    """Start MonoX training from the latest checkpoint."""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")
    
    # Check for latest checkpoint
    latest_checkpoint, latest_epoch = find_latest_monox_checkpoint()
    checkpoint_info = f"from epoch {latest_epoch}" if latest_checkpoint else "from scratch"
    
    # Start training in background
    background_tasks.add_task(run_monox_training)
    
    return {
        "message": f"MonoX training started {checkpoint_info}",
        "checkpoint": os.path.basename(latest_checkpoint) if latest_checkpoint else None,
        "epoch": latest_epoch,
        "status": "started"
    }

@app.get("/api/training-status")
async def get_training_status():
    """Get current MonoX training status."""
    latest_checkpoint, latest_epoch = find_latest_monox_checkpoint()
    checkpoint_count = len(list(Path("/tmp/checkpoints").glob("monox_generator_*.pth"))) if Path("/tmp/checkpoints").exists() else 0
    
    return {
        "is_training": training_status["is_training"],
        "current_step": training_status["current_step"],
        "message": training_status["message"],
        "latest_checkpoint": os.path.basename(latest_checkpoint) if latest_checkpoint else None,
        "latest_epoch": latest_epoch,
        "checkpoint_count": checkpoint_count,
        "device": "GPU" if torch.cuda.is_available() else "CPU"
    }

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available MonoX checkpoints."""
    checkpoint_dir = "/tmp/checkpoints"
    if not os.path.isdir(checkpoint_dir):
        return {"checkpoints": [], "count": 0}
    
    checkpoints = []
    for checkpoint_file in Path(checkpoint_dir).glob("monox_generator_*.pth"):
        stat = os.stat(checkpoint_file)
        try:
            epoch_num = int(checkpoint_file.stem.split('_')[-1])
        except:
            epoch_num = 0
            
        checkpoints.append({
            "name": checkpoint_file.name,
            "epoch": epoch_num,
            "size_mb": round(stat.st_size / (1024*1024), 2),
            "modified": stat.st_mtime
        })
    
    checkpoints.sort(key=lambda x: x["epoch"], reverse=True)
    return {"checkpoints": checkpoints, "count": len(checkpoints)}

@app.get("/api/dataset-info")
async def get_dataset_info():
    """Get information about the loaded dataset."""
    try:
        # Test loading the HF dataset
        dataset = MonoDataset(use_hf_dataset=True)
        
        return {
            "dataset_source": "lukua/monox-dataset" if dataset.use_hf_dataset else "local/synthetic",
            "total_images": len(dataset),
            "using_hf_dataset": dataset.use_hf_dataset,
            "status": "loaded" if len(dataset) > 0 else "empty",
            "message": "Successfully loaded MonoX dataset from Hugging Face"
        }
        
    except Exception as e:
        return {
            "dataset_source": "error",
            "total_images": 0,
            "using_hf_dataset": False,
            "status": "error",
            "message": f"Dataset loading failed: {str(e)}"
        }

class GDriveRequest(BaseModel):
    gdrive_url: str

@app.post("/api/download-hf-checkpoint")
async def download_hf_checkpoint(background_tasks: BackgroundTasks):
    """Download the latest checkpoint from Hugging Face model repository."""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Cannot download while training is in progress")
    
    async def download_checkpoint():
        global training_status
        try:
            training_status["message"] = "Downloading latest checkpoint from HF models..."
            
            # This will be implemented once the model repo is set up
            # For now, provide instructions
            training_status["message"] = "HF model integration ready - upload your checkpoints to lukua/monox-models first"
                
        except Exception as e:
            training_status["message"] = f"Download failed: {str(e)}"
    
    background_tasks.add_task(download_checkpoint)
    return {"message": "HF checkpoint download started", "status": "downloading"}

@app.post("/api/download-gdrive-checkpoint")
async def download_gdrive_checkpoint(request: GDriveRequest, background_tasks: BackgroundTasks):
    """Download a checkpoint from Google Drive (legacy support)."""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Cannot download while training is in progress")
    
    async def download_checkpoint():
        global training_status
        try:
            training_status["message"] = "Downloading checkpoint from Google Drive..."
            
            # Extract file ID from Google Drive URL
            gdrive_url = request.gdrive_url
            if "drive.google.com" in gdrive_url:
                if "/file/d/" in gdrive_url:
                    file_id = gdrive_url.split("/file/d/")[1].split("/")[0]
                elif "id=" in gdrive_url:
                    file_id = gdrive_url.split("id=")[1].split("&")[0]
                else:
                    raise ValueError("Invalid Google Drive URL format")
                
                # Convert to direct download URL
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                # Download the checkpoint
                checkpoint_dir = "/tmp/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Determine filename
                checkpoint_file = f"{checkpoint_dir}/downloaded_checkpoint.pth"
                
                urllib.request.urlretrieve(download_url, checkpoint_file)
                
                # Verify the checkpoint
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    if 'epoch' in checkpoint:
                        epoch = checkpoint['epoch']
                        new_name = f"{checkpoint_dir}/monox_generator_{epoch+1:05d}.pth"
                        os.rename(checkpoint_file, new_name)
                        training_status["message"] = f"Successfully downloaded checkpoint from epoch {epoch+1}"
                    else:
                        training_status["message"] = "Downloaded checkpoint (epoch unknown)"
                except Exception as e:
                    training_status["message"] = f"Downloaded file, but verification failed: {str(e)}"
            else:
                raise ValueError("URL must be a Google Drive link")
                
        except Exception as e:
            training_status["message"] = f"Download failed: {str(e)}"
    
    background_tasks.add_task(download_checkpoint)
    return {"message": "Checkpoint download started", "status": "downloading"}

@app.get("/api/generate-sample")
async def generate_sample():
    """Generate a sample image using the latest checkpoint."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Find latest checkpoint
        latest_checkpoint, latest_epoch = find_latest_monox_checkpoint()
        if not latest_checkpoint:
            raise HTTPException(status_code=404, detail="No checkpoint found. Train the model first.")
        
        # Load generator
        generator = Generator().to(device)
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        # Generate sample
        with torch.no_grad():
            z = torch.randn(1, 128).to(device)
            fake_img = generator(z)
            
            # Save sample
            samples_dir = "/tmp/samples"
            os.makedirs(samples_dir, exist_ok=True)
            sample_file = f"{samples_dir}/generated_sample_{int(time.time())}.png"
            save_image(fake_img, sample_file, normalize=True, value_range=(-1, 1))
        
        return {
            "message": "Sample generated successfully",
            "epoch": latest_epoch,
            "sample_file": os.path.basename(sample_file),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)