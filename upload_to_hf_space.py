#!/usr/bin/env python3
"""
Direct upload to HuggingFace Space using API
Bypasses git push timeout issues
"""

from huggingface_hub import HfApi, login
import os
from pathlib import Path

def upload_files():
    """Upload all necessary files to HF Space."""
    
    # Authenticate
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found!")
        return False
    
    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        print("✅ HuggingFace authentication successful")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    repo_id = "lukua/monox"
    
    # Files to upload
    files_to_upload = [
        "hf_dataset_training.py",
        "minimal_working_gan.py", 
        "app.py",
        "requirements.txt",
        "monitor_training_progress.py",
        "trigger_hf_training.py",
        "create_sample_dataset.py"
    ]
    
    print("🚀 Uploading files to HF Space...")
    
    for file_path in files_to_upload:
        if Path(file_path).exists():
            try:
                print(f"📤 Uploading: {file_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="space"
                )
                print(f"✅ Uploaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to upload {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    # Create a simple README for the Space
    readme_content = """---
title: MonoX Training
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# MonoX StyleGAN Training

AI-generated monotype artwork using StyleGAN-V architecture.

## Features
- Real-time training monitoring
- GPU-accelerated training
- Preview generation
- Checkpoint saving

## Usage
1. Click "Start GPU Training" for fastest results
2. Monitor progress in real-time
3. View generated samples in the preview panel

Training uses the `lukua/monox-dataset` with 868 monotype artwork samples.
"""
    
    try:
        print("📤 Creating README.md...")
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space"
        )
        print("✅ README.md uploaded")
    except Exception as e:
        print(f"❌ Failed to upload README: {e}")
    
    print("🎉 Upload completed!")
    return True

if __name__ == "__main__":
    success = upload_files()
    if success:
        print("\n🚀 Space should rebuild automatically")
        print("🔗 Check: https://huggingface.co/spaces/lukua/monox")
    else:
        print("\n❌ Upload failed")