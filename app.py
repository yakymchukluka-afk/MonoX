#!/usr/bin/env python3
"""
MonoX Hugging Face Space - FastAPI Server
A participative art project powered by StyleGAN-V for generating dynamic visual content.
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch

app = FastAPI(title="MonoX", description="A participative art project powered by StyleGAN-V")

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
            </div>
        </div>
    </body>
    </html>
    """.format(
        gpu_status="Yes" if torch.cuda.is_available() else "No",
        torch_version=torch.__version__
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)