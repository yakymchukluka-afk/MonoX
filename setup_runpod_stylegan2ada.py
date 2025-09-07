#!/usr/bin/env python3
"""
RunPod StyleGAN2-ADA Setup Script
This script sets up the complete environment for StyleGAN2-ADA training on RunPod
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, check=True, shell=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result

def check_gpu():
    """Check GPU availability and status"""
    print("🔍 Checking GPU status...")
    try:
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            print("✅ GPU detected:")
            print(result.stdout)
        else:
            print("❌ No GPU detected or nvidia-smi not available")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False
    return True

def install_dependencies():
    """Install required Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    dependencies = [
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "click>=8.0.0",
        "pillow>=8.0.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "huggingface_hub>=0.10.0",
        "datasets>=2.0.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        run_command(f"pip install {dep}")

def setup_stylegan2ada():
    """Set up StyleGAN2-ADA submodule"""
    print("🔧 Setting up StyleGAN2-ADA...")
    
    # Ensure we're in the right directory
    os.chdir("/workspace")
    
    # Initialize and update submodules
    run_command("git submodule update --init --recursive")
    
    # Checkout the compatibility branch
    os.chdir("/workspace/train/runpod-hf/vendor/stylegan2ada")
    run_command("git checkout pytorch-2.0-compatibility")
    
    print("✅ StyleGAN2-ADA setup complete")

def download_dataset():
    """Download the custom dataset from Hugging Face"""
    print("📥 Downloading dataset...")
    
    dataset_script = """
import os
from huggingface_hub import hf_hub_download
from datasets import load_dataset

# Create data directory
os.makedirs("/workspace/data", exist_ok=True)

try:
    # Download the dataset
    print("Downloading lukua/monox-dataset...")
    dataset = load_dataset("lukua/monox-dataset")
    
    # Save as zip file for StyleGAN2-ADA
    print("Preparing dataset for StyleGAN2-ADA...")
    import zipfile
    from PIL import Image
    import io
    
    with zipfile.ZipFile("/workspace/data/monox-dataset.zip", "w") as zipf:
        for i, item in enumerate(dataset["train"]):
            # Convert to PIL Image and resize to 1024x1024
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # Save to zip
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            zipf.writestr(f"{i:06d}.png", img_buffer.getvalue())
    
    print(f"✅ Dataset prepared: /workspace/data/monox-dataset.zip")
    print(f"📊 Total images: {len(dataset['train'])}")
    
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("Please ensure you have access to the lukua/monox-dataset")
    raise
"""
    
    with open("/tmp/download_dataset.py", "w") as f:
        f.write(dataset_script)
    
    run_command("python /tmp/download_dataset.py")

def test_compatibility():
    """Test PyTorch compatibility with StyleGAN2-ADA"""
    print("🧪 Testing PyTorch compatibility...")
    
    result = run_command("python /workspace/test_pytorch_compatibility.py", check=False)
    if result.returncode == 0:
        print("✅ Compatibility test passed!")
        print(result.stdout)
    else:
        print("❌ Compatibility test failed!")
        print(result.stdout)
        print(result.stderr)
        return False
    
    return True

def create_training_config():
    """Create training configuration files"""
    print("⚙️ Creating training configuration...")
    
    # Create output directory
    os.makedirs("/workspace/output", exist_ok=True)
    
    # Create a more detailed training script
    training_script = """#!/bin/bash

# StyleGAN2-ADA Training Script for RunPod
# Optimized for A100 80GB GPU

set -e

# Configuration
DATASET_PATH="/workspace/data/monox-dataset.zip"
OUTPUT_DIR="/workspace/output"
RESOLUTION=1024
BATCH_SIZE=8
GAMMA=10
MIRROR=1
KIMG=25000

# GPU optimization
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to StyleGAN2-ADA directory
cd /workspace/train/runpod-hf/vendor/stylegan2ada

echo "🚀 Starting StyleGAN2-ADA training..."
echo "📊 Dataset: $DATASET_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "🎯 Resolution: $RESOLUTION"
echo "📦 Batch size: $BATCH_SIZE"
echo "⚡ Gamma: $GAMMA"
echo "🔄 Kimg: $KIMG"

# Run training with PyTorch compatibility
python train.py \\
    --outdir="$OUTPUT_DIR" \\
    --data="$DATASET_PATH" \\
    --gpus=1 \\
    --batch="$BATCH_SIZE" \\
    --gamma="$GAMMA" \\
    --mirror="$MIRROR" \\
    --kimg="$KIMG" \\
    --snap=50 \\
    --metrics=fid50k_full \\
    --resume=ffhq1024 \\
    --cfg=auto \\
    --aug=ada \\
    --p=0.2 \\
    --target=0.6 \\
    --augpipe=blit,geom,color,filter,noise,cutout

echo "✅ Training completed successfully!"
"""
    
    with open("/workspace/train_stylegan2ada.sh", "w") as f:
        f.write(training_script)
    
    run_command("chmod +x /workspace/train_stylegan2ada.sh")
    print("✅ Training script created: /workspace/train_stylegan2ada.sh")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup StyleGAN2-ADA on RunPod")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-test", action="store_true", help="Skip compatibility test")
    args = parser.parse_args()
    
    print("🚀 Setting up StyleGAN2-ADA on RunPod")
    print("=" * 50)
    
    # Check GPU
    if not check_gpu():
        print("❌ GPU check failed. Please ensure you're running on a GPU instance.")
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Setup StyleGAN2-ADA
    setup_stylegan2ada()
    
    # Download dataset (unless skipped)
    if not args.skip_dataset:
        download_dataset()
    else:
        print("⏭️ Skipping dataset download")
    
    # Test compatibility (unless skipped)
    if not args.skip_test:
        if not test_compatibility():
            print("❌ Compatibility test failed. Please check the errors above.")
            sys.exit(1)
    else:
        print("⏭️ Skipping compatibility test")
    
    # Create training configuration
    create_training_config()
    
    print("\n🎉 Setup complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Verify your dataset is ready: ls -la /workspace/data/")
    print("2. Run training: ./train_stylegan2ada.sh")
    print("3. Monitor progress: tail -f /workspace/output/training_log.txt")
    print("\nFor troubleshooting, run: python /workspace/test_pytorch_compatibility.py")

if __name__ == "__main__":
    main()