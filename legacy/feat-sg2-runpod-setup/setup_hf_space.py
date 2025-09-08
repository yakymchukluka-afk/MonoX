#!/usr/bin/env python3
"""
Complete HF Space Setup for MonoX Training
This script sets up everything needed for proper HF Space operation.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the complete environment for HF Space."""
    print("🚀 Setting up MonoX HF Space Environment")
    print("=" * 50)
    
    # Create necessary directories
    directories = [
        "samples",
        "checkpoints", 
        "logs",
        "previews",
        ".huggingface"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Set environment variables
    os.environ['HF_HOME'] = '/workspace/.huggingface'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/.huggingface/hub'
    
    print("✅ Environment variables set")
    
    return True

def check_hf_authentication():
    """Check and setup HF authentication."""
    print("\\n🔧 Checking HF Authentication...")
    
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found!")
        print("📝 To fix this:")
        print("   1. Go to your HF Space settings")
        print("   2. Add secret: HF_TOKEN = your_hf_token")
        print("   3. Restart the space")
        return False
    
    print(f"✅ HF_TOKEN found: {hf_token[:8]}...")
    
    # Save token to file
    token_file = Path('.huggingface/token')
    with open(token_file, 'w') as f:
        f.write(hf_token)
    
    print("✅ Token saved to .huggingface/token")
    return True

def create_hf_config():
    """Create proper HF configuration."""
    print("\\n📝 Creating HF configuration...")
    
    config_content = """# HF Spaces Configuration
# Proper authentication setup

build:
  skip_git_config: true

# Repository settings
repository: lukua/monox-model
upload_paths:
  samples: samples/
  checkpoints: checkpoints/
  logs: logs/
"""
    
    config_file = Path('.huggingface/config.yaml')
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ HF config created")
    return True

def test_upload():
    """Test upload functionality."""
    print("\\n🧪 Testing upload functionality...")
    
    try:
        from huggingface_hub import HfApi, upload_file
        
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print("❌ No token available for testing")
            return False
        
        # Create test file
        test_file = Path('test_upload.txt')
        with open(test_file, 'w') as f:
            f.write("Test upload from HF Space\\n")
            f.write(f"Timestamp: {time.time()}\\n")
        
        # Test upload
        upload_file(
            path_or_fileobj=str(test_file),
            path_in_repo="test_upload.txt",
            repo_id="lukua/monox-model",
            token=hf_token,
            repo_type="model"
        )
        
        print("✅ Upload test successful!")
        test_file.unlink()  # Clean up
        return True
        
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return False

def create_space_readme():
    """Create README for the HF Space."""
    print("\\n📖 Creating Space README...")
    
    readme_content = """---
title: MonoX Training
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: AI-powered monotype art generation
---

# MonoX Training Space

This space runs MonoX StyleGAN training with automatic uploads to the model repository.

## Features

- 🚀 GPU-accelerated training
- 📤 Automatic uploads to lukua/monox-model
- 📊 Real-time progress monitoring
- 🎨 High-quality monotype art generation

## Setup

1. Add your HF_TOKEN as a secret in Space settings
2. The training will automatically start and upload results
3. Check the model repository for generated samples

## Repository

All outputs are uploaded to: https://huggingface.co/lukua/monox-model
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ README created")
    return True

def main():
    """Main setup function."""
    print("🎯 MonoX HF Space Complete Setup")
    print("=" * 40)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    # Check authentication
    auth_ok = check_hf_authentication()
    
    # Create config
    create_hf_config()
    
    # Create README
    create_space_readme()
    
    # Test upload if authenticated
    if auth_ok:
        test_upload()
    
    print("\\n✅ Setup completed!")
    print("\\n📋 Next steps:")
    if not auth_ok:
        print("   1. Add HF_TOKEN as secret in Space settings")
        print("   2. Restart the Space")
    print("   3. Run: python fixed_training_script.py")
    print("   4. Check lukua/monox-model for uploaded files")
    
    return True

if __name__ == "__main__":
    import time
    main()