#!/usr/bin/env python3
"""
Download the MonoX dataset from Hugging Face datasets repo (public access only).
"""
import os
from huggingface_hub import snapshot_download

def main():
    print("[dataset] Downloading MonoX dataset from Hugging Face (public access)...")
    
    # Ensure data directory exists
    os.makedirs("/workspace/data", exist_ok=True)
    
    # Download dataset using public access (no token needed)
    snapshot_download(
        repo_id="lukua/monox-dataset", 
        repo_type="dataset", 
        local_dir="/workspace/data/monox-dataset", 
        local_dir_use_symlinks=False, 
        max_workers=8,
        token=None  # Explicitly no token for public access
    )
    
    print("[dataset] Dataset downloaded successfully to /workspace/data/monox-dataset")

if __name__ == "__main__":
    main()