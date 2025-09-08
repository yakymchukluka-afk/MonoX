#!/usr/bin/env python3
"""
Simple one-command setup for MonoX in Google Colab
Handles all the pip version compatibility issues
"""

import subprocess
import sys

def run_cmd(cmd):
    """Run command and return success status"""
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except:
        return False

print("🚀 Installing MonoX dependencies for Colab...")

# Method 1: Try with pip downgrade
print("📦 Method 1: Using compatible pip version...")
if run_cmd("pip install 'pip<24.1'"):
    if run_cmd("pip install omegaconf==2.0.4"):
        if run_cmd("pip install hydra-core==1.0.7"):
            print("✅ Method 1 successful!")
        else:
            print("❌ Method 1 failed on hydra")
    else:
        print("❌ Method 1 failed on omegaconf")
else:
    print("❌ Method 1 failed on pip downgrade")

# Method 2: Try newer compatible versions
print("\n📦 Method 2: Using newer compatible versions...")
if run_cmd("pip install omegaconf>=2.1.0"):
    if run_cmd("pip install hydra-core>=1.1.0"):
        print("✅ Method 2 successful!")
    else:
        print("❌ Method 2 failed on hydra")
else:
    print("❌ Method 2 failed on omegaconf")

# Method 3: Force install with --force-reinstall
print("\n📦 Method 3: Force reinstall...")
if run_cmd("pip install --force-reinstall --no-deps omegaconf==2.0.4"):
    if run_cmd("pip install --force-reinstall --no-deps hydra-core==1.0.7"):
        if run_cmd("pip install pyyaml>=5.1"):  # Install PyYAML separately
            print("✅ Method 3 successful!")
        else:
            print("❌ Method 3 failed on PyYAML")
    else:
        print("❌ Method 3 failed on hydra")
else:
    print("❌ Method 3 failed on omegaconf")

# Install other dependencies
print("\n📦 Installing other dependencies...")
other_deps = [
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
    "'pytorch-lightning>=1.5.0,<1.8.0'",
    "numpy pillow scipy tqdm tensorboard matplotlib opencv-python",
    "imageio imageio-ffmpeg ninja psutil"
]

for dep in other_deps:
    run_cmd(f"pip install {dep}")

# Test import
print("\n🔍 Testing imports...")
try:
    import hydra
    import omegaconf
    import torch
    print("✅ All core packages imported successfully!")
    print(f"🔥 Hydra: {hydra.__version__}")
    print(f"🔥 OmegaConf: {omegaconf.__version__}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print("🎉 Setup complete! Ready for training.")
except Exception as e:
    print(f"❌ Import test failed: {e}")
    print("⚠️  You may need to restart the runtime and try again.")