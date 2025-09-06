#!/usr/bin/env python3
"""
Download the MonoX dataset from Hugging Face model repo.
"""
import os
from huggingface_hub import snapshot_download

def main():
    print("[dataset] Downloading MonoX dataset from Hugging Face...")
    
    # Ensure data directory exists
    os.makedirs("/workspace/data", exist_ok=True)
    
    # Download dataset using model-repo semantics
    snapshot_download(
        repo_id="lukua/monox-dataset", 
        repo_type="model", 
        local_dir="/workspace/data/monox-dataset", 
        local_dir_use_symlinks=False, 
        max_workers=8
    )
    
    print("[dataset] Dataset downloaded successfully to /workspace/data/monox-dataset")

if __name__ == "__main__":
    main()