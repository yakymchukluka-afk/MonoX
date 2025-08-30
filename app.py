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
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch

app = FastAPI(title="MonoX", description="A participative art project powered by StyleGAN-V")

# Global training state
training_status = {"is_training": False, "current_step": 0, "message": "Ready to start training"}

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

async def run_training_process():
    """Run the training process in the background."""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["message"] = "Starting training..."
        
        # Check for existing checkpoint
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            training_status["message"] = f"Resuming from checkpoint: {os.path.basename(latest_checkpoint)}"
        else:
            training_status["message"] = "Starting training from scratch"
        
        # Create directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("previews", exist_ok=True)
        
        # Run training command
        cmd = [
            sys.executable, "train.py",
            "dataset=ffs",
            "training.steps=1000",  # Quick training for demo
            "training.batch=2",
            "training.num_workers=0",
            "training.fp16=false",
            "launcher=local"  # Use local mode for Hugging Face Space
        ]
        
        if latest_checkpoint:
            cmd.append(f"training.resume={latest_checkpoint}")
        
        training_status["message"] = f"Running: {' '.join(cmd)}"
        
        # Run the training process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app"
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            training_status["message"] = "Training completed successfully!"
        else:
            training_status["message"] = f"Training failed: {stderr.decode()[:200]}"
            
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
                            alert('Training Status: ' + result.message);
                        }} catch (e) {{
                            alert('Error checking status: ' + e.message);
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
    """Start training from the latest checkpoint."""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")
    
    # Check for latest checkpoint
    latest_checkpoint = find_latest_checkpoint()
    checkpoint_info = f"from checkpoint {os.path.basename(latest_checkpoint)}" if latest_checkpoint else "from scratch"
    
    # Start training in background
    background_tasks.add_task(run_training_process)
    
    return {
        "message": f"Training started {checkpoint_info}",
        "checkpoint": os.path.basename(latest_checkpoint) if latest_checkpoint else None,
        "status": "started"
    }

@app.get("/api/training-status")
async def get_training_status():
    """Get current training status."""
    latest_checkpoint = find_latest_checkpoint()
    
    return {
        "is_training": training_status["is_training"],
        "current_step": training_status["current_step"],
        "message": training_status["message"],
        "latest_checkpoint": os.path.basename(latest_checkpoint) if latest_checkpoint else None,
        "checkpoint_count": len(glob.glob("checkpoints/*.pkl")) + len(glob.glob("checkpoints/*.pt"))
    }

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available checkpoints."""
    checkpoint_dir = "checkpoints"
    if not os.path.isdir(checkpoint_dir):
        return {"checkpoints": [], "count": 0}
    
    patterns = ["*.pkl", "*.pt", "*.ckpt", "*.pth"]
    checkpoints = []
    for pattern in patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        for file in files:
            stat = os.stat(file)
            checkpoints.append({
                "name": os.path.basename(file),
                "size_mb": round(stat.st_size / (1024*1024), 2),
                "modified": stat.st_mtime
            })
    
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)
    return {"checkpoints": checkpoints, "count": len(checkpoints)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)