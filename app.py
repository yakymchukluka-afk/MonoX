#!/usr/bin/env python3
"""
MonoX Training Interface - FastAPI Backend
Designed for HF Spaces with Docker deployment
"""

from fastapi import FastAPI
import os
import subprocess
import time
from pathlib import Path
import torch

app = FastAPI(title="MonoX Backend", description="StyleGAN-V based generative art platform")

def setup_environment():
    """Minimal environment setup."""
    # Create directories
    Path("previews").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    return "✅ Environment ready"

def check_training_status():
    """Check if training is running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'gan_training'], capture_output=True, text=True)
        if result.stdout.strip():
            return "🟢 Training is RUNNING"
        else:
            return "🔴 Training is STOPPED"
    except:
        return "❓ Status unknown"

def get_progress_info():
    """Get current progress information."""
    preview_dir = Path("previews")
    checkpoint_dir = Path("checkpoints")
    
    previews = len(list(preview_dir.glob("*.png"))) if preview_dir.exists() else 0
    checkpoints = len(list(checkpoint_dir.glob("*.pth"))) if checkpoint_dir.exists() else 0
    
    return {
        "status": check_training_status(),
        "previews": previews,
        "checkpoints": checkpoints,
        "progress_percent": previews * 2,
        "hardware_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

@app.get("/")
def greet():
    """Root endpoint - confirms MonoX backend is active."""
    return {"status": "MonoX backend active"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    setup_result = setup_environment()
    return {
        "status": "healthy",
        "setup": setup_result,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/training/status")
def training_status():
    """Get training status and progress."""
    return get_progress_info()

@app.post("/training/start/cpu")
def start_cpu_training():
    """Start CPU training."""
    try:
        subprocess.Popen(['python3', 'simple_gan_training.py'])
        return {"status": "success", "message": "🚀 CPU training started! Check back in 15 minutes for next sample."}
    except Exception as e:
        return {"status": "error", "message": f"❌ Failed to start training: {e}"}

@app.post("/training/start/gpu")
def start_gpu_training():
    """Start GPU training if available."""
    try:
        if torch.cuda.is_available():
            subprocess.Popen(['python3', 'gpu_gan_training.py'])
            return {"status": "success", "message": "🚀 GPU training started! Much faster - check back in 30 seconds!"}
        else:
            return {"status": "warning", "message": "⚠️ No GPU detected. Upgrade hardware in Space settings first."}
    except Exception as e:
        return {"status": "error", "message": f"❌ Failed to start GPU training: {e}"}

@app.get("/samples/latest")
def get_latest_sample():
    """Get the latest generated sample."""
    preview_dir = Path("previews")
    if not preview_dir.exists():
        return {"status": "no_samples", "message": "No samples generated yet"}
    
    samples = sorted(preview_dir.glob("samples_epoch_*.png"), key=lambda x: x.stat().st_mtime)
    if not samples:
        return {"status": "no_samples", "message": "No samples found"}
    
    latest = samples[-1]
    epoch_num = int(latest.stem.split('_')[-1])
    size_mb = latest.stat().st_size / (1024*1024)
    
    return {
        "status": "success",
        "file_path": str(latest),
        "epoch": epoch_num,
        "size_mb": round(size_mb, 1)
    }

# Initialize environment on startup
setup_environment()