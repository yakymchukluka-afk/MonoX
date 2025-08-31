#!/usr/bin/env python3
"""
🚀 Create MonoX Training Repository on Hugging Face
==================================================

This script creates the lukua/monox-training repository for storing
training progress, checkpoints, logs, and latent walks.
"""

import subprocess
import sys
from huggingface_hub import HfApi, create_repo

def check_hf_auth():
    """Check if HF CLI is authenticated."""
    try:
        result = subprocess.run(["hf", "auth", "whoami"], capture_output=True, text=True)
        if "Not logged in" in result.stdout:
            print("❌ Not authenticated with Hugging Face")
            print("Please run: hf auth login")
            return False
        else:
            print("✅ Hugging Face authentication verified")
            print(f"Logged in as: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ Error checking HF auth: {e}")
        return False

def create_training_repository():
    """Create the training progress repository."""
    print("\n📊 Creating MonoX Training Repository...")
    
    try:
        # Create the repository
        repo_url = create_repo(
            repo_id="lukua/monox-training",
            repo_type="model",  # Using model type for better LFS support
            exist_ok=True,
            private=False
        )
        
        print(f"✅ Training repository created: {repo_url}")
        print("📁 Repository: https://huggingface.co/lukua/monox-training")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating training repository: {e}")
        return False

def main():
    """Main function."""
    print("🚀 MonoX Training Repository Setup")
    print("=" * 50)
    
    # Check authentication
    if not check_hf_auth():
        print("\n❌ Please authenticate first:")
        print("   hf auth login")
        return False
    
    # Create repository
    if not create_training_repository():
        return False
    
    print("\n🎉 Training repository setup complete!")
    print("\n📋 Next steps:")
    print("1. Your Space will now save training progress to /tmp/")
    print("2. Periodically upload progress to lukua/monox-training")
    print("3. Start training from epoch 0 with fresh models")
    print("4. Generate latent walks and monitor progress")
    
    print("\n🔗 Your MonoX Ecosystem:")
    print("🎨 Dataset: https://huggingface.co/datasets/lukua/monox-dataset")
    print("📊 Training: https://huggingface.co/lukua/monox-training")
    print("🌐 Space: https://huggingface.co/spaces/lukua/monox")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)