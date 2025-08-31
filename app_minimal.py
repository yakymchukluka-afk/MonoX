#!/usr/bin/env python3
"""
MonoX Hugging Face Space - Minimal Stable Version
A participative art project powered by StyleGAN-V for generating dynamic visual content.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import torch

app = FastAPI(title="MonoX", description="A participative art project powered by StyleGAN-V")

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
                <p>Ready to start fresh training from epoch 0 using your monotype dataset.</p>
                <button onclick="alert('Training system will be activated once Space is stable!')">Start Training (Coming Soon)</button>
            </div>
            
            <div class="section">
                <h4>🎨 Art Generation</h4>
                <p>Generate samples and latent walks once training begins.</p>
                <button onclick="alert('Art generation will be available after training!')">Generate Art (Coming Soon)</button>
            </div>
            
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)