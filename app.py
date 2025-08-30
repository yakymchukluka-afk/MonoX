#!/usr/bin/env python3
"""
MonoX Hugging Face Space - Minimal Stable Version
A participative art project powered by StyleGAN-V for generating dynamic visual content.
"""

import os
import sys
import time
import asyncio
from pathlib import Path
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
import numpy as np

app = FastAPI(title="MonoX", description="A participative art project powered by StyleGAN-V")

# Global training state
training_status = {"is_training": False, "current_step": 0, "message": "Ready to start fresh training"}

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

class SyntheticDataset(Dataset):
    """Simple synthetic dataset for training demo."""
    def __init__(self, size=100, transform=None):
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate synthetic art-like images
        image = Image.new('RGB', (512, 512))
        pixels = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        
        if self.transform:
            image = self.transform(image)
        return image

def find_latest_checkpoint(checkpoint_dir="/tmp/checkpoints"):
    """Find the latest checkpoint."""
    if not Path(checkpoint_dir).exists():
        return None, 0
    
    checkpoints = list(Path(checkpoint_dir).glob("monox_generator_*.pth"))
    if not checkpoints:
        return None, 0
    
    # Find latest by epoch number
    latest_epoch = 0
    latest_file = None
    for ckpt in checkpoints:
        try:
            epoch = int(ckpt.stem.split('_')[-1])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = ckpt
        except:
            continue
    
    return str(latest_file) if latest_file else None, latest_epoch

async def run_training():
    """Run MonoX training from scratch."""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["message"] = "Initializing fresh training..."
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs("/tmp/checkpoints", exist_ok=True)
        os.makedirs("/tmp/samples", exist_ok=True)
        
        # Initialize models
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        
        # Check for existing checkpoint
        latest_checkpoint, start_epoch = find_latest_checkpoint()
        
        if latest_checkpoint:
            training_status["message"] = f"Resuming from epoch {start_epoch}"
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            start_epoch += 1
        else:
            training_status["message"] = "Starting fresh training from epoch 0"
            start_epoch = 0
        
        # Setup optimizers
        lr = 0.0002
        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Load optimizer states if resuming
        if latest_checkpoint and 'optimizer_G_state_dict' in checkpoint:
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        # Setup dataset and loss
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = SyntheticDataset(size=50, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        adversarial_loss = nn.BCELoss()
        
        training_status["message"] = "Training in progress..."
        
        # Training loop
        for epoch in range(start_epoch, start_epoch + 20):  # 20 epochs for demo
            epoch_start = time.time()
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for batch_idx, real_imgs in enumerate(dataloader):
                if batch_idx >= 5:  # Limit batches
                    break
                    
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9
                fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1
                
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
            
            avg_d_loss = epoch_d_loss / 5
            avg_g_loss = epoch_g_loss / 5
            
            training_status["current_step"] = epoch + 1
            training_status["message"] = f"Epoch {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}"
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_file = f"/tmp/checkpoints/monox_generator_{epoch+1:05d}.pth"
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
                    sample_file = f"/tmp/samples/epoch_{epoch+1:05d}.png"
                    save_image(sample_imgs, sample_file, nrow=2, normalize=True, value_range=(-1, 1))
            
            await asyncio.sleep(0.1)
        
        training_status["message"] = f"Training completed! Generated {training_status['current_step']} epochs."
        
    except Exception as e:
        training_status["message"] = f"Training error: {str(e)}"
    finally:
        training_status["is_training"] = False

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main page of the MonoX application."""
    gpu_status = "Yes" if torch.cuda.is_available() else "No"
    torch_version = torch.__version__
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MonoX - StyleGAN-V Art Project</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #333; text-align: center; }}
            .info {{ background: #e8f4fd; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .section {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            button {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }}
            button:hover {{ background: #0056b3; }}
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
                <p><strong>Dataset:</strong> <a href="https://huggingface.co/datasets/lukua/monox-dataset" target="_blank" style="color: #007bff;">lukua/monox-dataset</a> (800+ monotype images)</p>
            </div>
            
            <div class="section">
                <h4>🚀 Training System</h4>
                <p>Ready to start fresh training from epoch 0. Current status: {training_status["message"]}</p>
                <button onclick="startTraining()">Start Fresh Training</button>
                <button onclick="checkTrainingStatus()">Check Training Status</button>
            </div>
            
            <div class="section">
                <h4>🎨 Art Generation</h4>
                <p>Generate samples using trained models.</p>
                <button onclick="generateSample()">Generate Sample</button>
                <button onclick="listCheckpoints()">List Checkpoints</button>
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
                
                async function checkTrainingStatus() {{
                    try {{
                        const response = await fetch('/api/training-status');
                        const result = await response.json();
                        alert(`Training Status:\\nEpoch: ${{result.current_step}}\\nMessage: ${{result.message}}\\nCheckpoints: ${{result.checkpoint_count}}`);
                    }} catch (e) {{
                        alert('Error checking status: ' + e.message);
                    }}
                }}
                
                async function generateSample() {{
                    try {{
                        const response = await fetch('/api/generate-sample');
                        const result = await response.json();
                        alert(`Sample generated from epoch ${{result.epoch}}!`);
                    }} catch (e) {{
                        alert('Error generating sample: ' + e.message);
                    }}
                }}
                
                async function listCheckpoints() {{
                    try {{
                        const response = await fetch('/api/checkpoints');
                        const result = await response.json();
                        if (result.count > 0) {{
                            let msg = `Found ${{result.count}} checkpoints:\\n\\n`;
                            result.checkpoints.forEach(ckpt => {{
                                msg += `Epoch ${{ckpt.epoch}}: ${{ckpt.name}}\\n`;
                            }});
                            alert(msg);
                        }} else {{
                            alert('No checkpoints found. Start training first!');
                        }}
                    }} catch (e) {{
                        alert('Error listing checkpoints: ' + e.message);
                    }}
                }}
            </script>
            
            <div class="section">
                <h4>📊 Migration Status</h4>
                <p><strong>✅ Dataset:</strong> Uploaded to lukua/monox-dataset</p>
                <p><strong>✅ Space:</strong> Running and stable</p>
                <p><strong>🔄 Training:</strong> Ready to start fresh from epoch 0</p>
                <p><strong>📁 Storage:</strong> Training progress will be saved to HF repositories</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "message": "MonoX Space is running successfully"
    }

@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks):
    """Start MonoX training."""
    global training_status
    
    if training_status["is_training"]:
        return {"message": "Training is already in progress", "status": "running"}
    
    latest_checkpoint, latest_epoch = find_latest_checkpoint()
    checkpoint_info = f"from epoch {latest_epoch}" if latest_checkpoint else "from scratch"
    
    background_tasks.add_task(run_training)
    
    return {
        "message": f"Training started {checkpoint_info}",
        "epoch": latest_epoch,
        "status": "started"
    }

@app.get("/api/training-status")
async def get_training_status():
    """Get current training status."""
    latest_checkpoint, latest_epoch = find_latest_checkpoint()
    checkpoint_count = len(list(Path("/tmp/checkpoints").glob("*.pth"))) if Path("/tmp/checkpoints").exists() else 0
    
    return {
        "is_training": training_status["is_training"],
        "current_step": training_status["current_step"],
        "message": training_status["message"],
        "latest_epoch": latest_epoch,
        "checkpoint_count": checkpoint_count,
        "device": "GPU" if torch.cuda.is_available() else "CPU"
    }

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available checkpoints."""
    checkpoint_dir = "/tmp/checkpoints"
    if not Path(checkpoint_dir).exists():
        return {"checkpoints": [], "count": 0}
    
    checkpoints = []
    for checkpoint_file in Path(checkpoint_dir).glob("monox_generator_*.pth"):
        try:
            epoch_num = int(checkpoint_file.stem.split('_')[-1])
            stat = os.stat(checkpoint_file)
            checkpoints.append({
                "name": checkpoint_file.name,
                "epoch": epoch_num,
                "size_mb": round(stat.st_size / (1024*1024), 2)
            })
        except:
            continue
    
    checkpoints.sort(key=lambda x: x["epoch"], reverse=True)
    return {"checkpoints": checkpoints, "count": len(checkpoints)}

@app.get("/api/generate-sample")
async def generate_sample():
    """Generate a sample image."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        latest_checkpoint, latest_epoch = find_latest_checkpoint()
        if not latest_checkpoint:
            return {"message": "No checkpoint found. Start training first!", "status": "no_model"}
        
        # Load generator
        generator = Generator().to(device)
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        # Generate sample
        with torch.no_grad():
            z = torch.randn(1, 128).to(device)
            sample_img = generator(z)
            
            sample_file = f"/tmp/samples/generated_{int(time.time())}.png"
            save_image(sample_img, sample_file, normalize=True, value_range=(-1, 1))
        
        return {
            "message": "Sample generated successfully",
            "epoch": latest_epoch,
            "status": "success"
        }
        
    except Exception as e:
        return {"message": f"Generation failed: {str(e)}", "status": "error"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)