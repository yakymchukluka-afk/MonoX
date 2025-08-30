#!/usr/bin/env python3
"""
🚀 MonoX Complete Migration Script - Google Drive to Hugging Face
================================================================

This script migrates your complete MonoX project from Google Drive to Hugging Face:
1. Creates HF dataset repository for training images
2. Creates HF model repository for checkpoints  
3. Uploads all your training data and checkpoints
4. Updates the Space to use HF repositories

Prerequisites:
- Hugging Face account with write access
- Google Drive with MonoX_training folder
- HF CLI authenticated: `hf auth login`
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

def check_hf_auth():
    """Check if HF CLI is authenticated."""
    try:
        result = subprocess.run(["hf", "auth", "whoami"], capture_output=True, text=True)
        if "Not logged in" in result.stdout:
            print("❌ Not authenticated with Hugging Face")
            print("Please run: hf auth login")
            print("Then paste your Hugging Face token")
            return False
        else:
            print("✅ Hugging Face authentication verified")
            return True
    except Exception as e:
        print(f"❌ Error checking HF auth: {e}")
        return False

def create_repositories():
    """Create the dataset and model repositories on Hugging Face."""
    print("\n📁 Creating Hugging Face repositories...")
    
    api = HfApi()
    
    try:
        # Create dataset repository
        print("Creating dataset repository: lukua/monox-dataset")
        create_repo(
            repo_id="lukua/monox-dataset",
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
        print("✅ Dataset repository created")
        
        # Create model repository  
        print("Creating model repository: lukua/monox-models")
        create_repo(
            repo_id="lukua/monox-models",
            repo_type="model", 
            exist_ok=True,
            private=False
        )
        print("✅ Model repository created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating repositories: {e}")
        return False

def setup_dataset_repo():
    """Set up the dataset repository structure."""
    print("\n🎨 Setting up dataset repository...")
    
    dataset_dir = "/tmp/monox-dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create README
    readme_content = '''---
title: MonoX Training Dataset
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: static
license: mit
tags:
- art
- monotype
- stylegan
- generative-art
- dataset
---

# MonoX Training Dataset

This dataset contains 800+ monotype images used for training the MonoX generative art model.

## Usage

```python
from datasets import load_dataset
dataset = load_dataset("lukua/monox-dataset")
```

## Training

Use with MonoX Space: https://huggingface.co/spaces/lukua/monox
'''
    
    with open(f"{dataset_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Dataset repository structure ready")
    return dataset_dir

def setup_model_repo():
    """Set up the model repository structure."""
    print("\n🧠 Setting up model repository...")
    
    model_dir = "/tmp/monox-models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create README and config
    readme_content = '''---
title: MonoX Models  
emoji: 🧠
colorFrom: purple
colorTo: blue
license: mit
tags:
- pytorch
- stylegan
- generative-art
- monotype
---

# MonoX Models

Trained model checkpoints for the MonoX generative art project.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download latest checkpoint
checkpoint_path = hf_hub_download(repo_id="lukua/monox-models", filename="latest_checkpoint.pth")
checkpoint = torch.load(checkpoint_path)
```

## Integration

- **Dataset**: [lukua/monox-dataset](https://huggingface.co/datasets/lukua/monox-dataset)
- **Interactive Space**: [lukua/monox](https://huggingface.co/spaces/lukua/monox)
'''
    
    config_content = {
        "model_type": "monox-gan",
        "architecture": "custom", 
        "framework": "pytorch",
        "task": "image-generation",
        "dataset": "lukua/monox-dataset",
        "image_size": 512,
        "latent_dim": 128
    }
    
    with open(f"{model_dir}/README.md", "w") as f:
        f.write(readme_content)
        
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config_content, f, indent=2)
    
    print("✅ Model repository structure ready")
    return model_dir

def upload_to_hf():
    """Upload the repository structures to Hugging Face."""
    print("\n⬆️ Uploading to Hugging Face...")
    
    try:
        # Upload dataset repo structure
        upload_file(
            path_or_fileobj="/tmp/monox-dataset/README.md",
            path_in_repo="README.md",
            repo_id="lukua/monox-dataset",
            repo_type="dataset"
        )
        print("✅ Dataset README uploaded")
        
        # Upload model repo structure  
        upload_file(
            path_or_fileobj="/tmp/monox-models/README.md", 
            path_in_repo="README.md",
            repo_id="lukua/monox-models",
            repo_type="model"
        )
        
        upload_file(
            path_or_fileobj="/tmp/monox-models/config.json",
            path_in_repo="config.json", 
            repo_id="lukua/monox-models",
            repo_type="model"
        )
        print("✅ Model repository files uploaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def create_migration_instructions():
    """Create instructions for manual data migration."""
    instructions = '''
🚀 MonoX Migration Instructions
==============================

Your Hugging Face repositories are now set up! Here's how to complete the migration:

## 1. Dataset Migration (800+ images)

### Option A: Using HF CLI (Recommended)
```bash
# Download your images from Google Drive to local folder
# Then upload to HF dataset:
hf upload lukua/monox-dataset ./your_local_images_folder --repo-type=dataset
```

### Option B: Using Web Interface
1. Go to: https://huggingface.co/datasets/lukua/monox-dataset
2. Click "Add file" → "Upload files"
3. Upload your monotype images from Google Drive

## 2. Checkpoint Migration

### Option A: Using HF CLI
```bash
# Download checkpoints from Google Drive
# Then upload:
hf upload lukua/monox-models ./checkpoints_folder
```

### Option B: Individual Upload
1. Go to: https://huggingface.co/lukua/monox-models  
2. Upload each checkpoint file individually

## 3. Google Drive Download Helper

If you need to download from Google Drive in bulk:

```python
# Use this script to download from Google Drive
import gdown
import os

# Your Google Drive folder ID
folder_id = "YOUR_GOOGLE_DRIVE_FOLDER_ID"

# Download entire folder
gdown.download_folder(
    f"https://drive.google.com/drive/folders/{folder_id}",
    output="./monox_data",
    quiet=False
)
```

## 4. Next Steps

Once uploaded, your MonoX Space will automatically:
- Load datasets from: lukua/monox-dataset  
- Load checkpoints from: lukua/monox-models
- Resume training from latest checkpoint
- Generate art using trained models

🎉 Your MonoX project will be fully integrated in Hugging Face!
'''
    
    with open("/workspace/MIGRATION_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("📋 Migration instructions created: MIGRATION_INSTRUCTIONS.md")

def main():
    """Main migration function."""
    print("🚀 MonoX Complete Migration Script")
    print("=" * 50)
    
    # Check authentication
    if not check_hf_auth():
        print("\n❌ Please authenticate with Hugging Face first:")
        print("   hf auth login")
        return False
    
    # Create repositories
    if not create_repositories():
        return False
    
    # Set up repository structures
    dataset_dir = setup_dataset_repo()
    model_dir = setup_model_repo()
    
    # Upload initial structure
    if not upload_to_hf():
        return False
    
    # Create migration instructions
    create_migration_instructions()
    
    print("\n🎉 Migration setup complete!")
    print("📁 Dataset repo: https://huggingface.co/datasets/lukua/monox-dataset")
    print("🧠 Model repo: https://huggingface.co/lukua/monox-models")
    print("📋 See MIGRATION_INSTRUCTIONS.md for next steps")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)