#!/usr/bin/env python3
"""
Upload checkpoints and samples to Hugging Face model repo.
"""
import os
import time
from pathlib import Path
from huggingface_hub import upload_folder

def main():
    print("[upload] Starting batch upload to Hugging Face...")
    
    # Ensure output directory exists
    out_dir = Path("/workspace/out")
    if not out_dir.exists():
        print("[upload] No output directory found, waiting...")
        time.sleep(60)  # Wait a bit for training to start
    
    # Upload using model-repo semantics
    upload_folder(
        folder_path="/workspace/out", 
        repo_id="lukua/monox-model", 
        repo_type="model", 
        allow_patterns=["checkpoints/**", "samples/**", "logs/**", "*.md"], 
        commit_message="batch: checkpoints + samples"
    )
    
    print("[upload] Upload completed successfully")

if __name__ == "__main__":
    main()