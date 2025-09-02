#!/usr/bin/env python3
"""
MonoX - Clean FastAPI Application for Latent Walk Generation
Restored to clean state - August 26, 2025 baseline
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

app = FastAPI(title="MonoX Latent Walk Generator", version="1.0.0")

# Data models
class LatentWalkRequest(BaseModel):
    steps: int = 10
    interpolation_method: str = "spherical"
    seed_start: Optional[int] = None
    seed_end: Optional[int] = None

class LatentWalkResponse(BaseModel):
    walk_id: str
    steps: int
    method: str
    created_at: str
    status: str

# Ensure directories exist
def ensure_directories():
    """Create necessary directories for the application."""
    Path("walks").mkdir(exist_ok=True)
    Path("previews").mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    ensure_directories()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MonoX Latent Walk Generator",
        "version": "1.0.0",
        "description": "Clean FastAPI service for generating latent walks",
        "reset_from": "commit 4e42c0a (Aug 26, 2025)",
        "endpoints": [
            "/walks/generate - Generate a new latent walk",
            "/walks/{walk_id} - Get walk details",
            "/walks - List all walks"
        ]
    }

@app.post("/walks/generate", response_model=LatentWalkResponse)
async def generate_latent_walk(request: LatentWalkRequest):
    """Generate a new latent walk."""
    walk_id = str(uuid.uuid4())
    
    # Create walk metadata
    walk_data = {
        "walk_id": walk_id,
        "steps": request.steps,
        "interpolation_method": request.interpolation_method,
        "seed_start": request.seed_start,
        "seed_end": request.seed_end,
        "created_at": datetime.now().isoformat(),
        "status": "generated"
    }
    
    # Save walk data as JSON
    walks_dir = Path("walks")
    walk_file = walks_dir / f"{walk_id}.json"
    
    with open(walk_file, "w") as f:
        json.dump(walk_data, f, indent=2)
    
    return LatentWalkResponse(**walk_data)

@app.get("/walks/{walk_id}")
async def get_walk(walk_id: str):
    """Get details of a specific latent walk."""
    walk_file = Path("walks") / f"{walk_id}.json"
    
    if not walk_file.exists():
        raise HTTPException(status_code=404, detail="Walk not found")
    
    with open(walk_file, "r") as f:
        walk_data = json.load(f)
    
    return walk_data

@app.get("/walks")
async def list_walks():
    """List all generated latent walks."""
    walks_dir = Path("walks")
    walks = []
    
    for walk_file in walks_dir.glob("*.json"):
        try:
            with open(walk_file, "r") as f:
                walk_data = json.load(f)
                walks.append(walk_data)
        except Exception:
            continue  # Skip corrupted files
    
    # Sort by creation time, newest first
    walks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return {"walks": walks, "count": len(walks)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "walks_directory": os.path.exists("walks"),
        "previews_directory": os.path.exists("previews")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)