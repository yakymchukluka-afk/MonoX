#!/usr/bin/env python3
"""
Python 3.12 compatible installer for MonoX in Google Colab
Uses newer Hydra version that's compatible with Python 3.12
"""

import subprocess
import sys

def run_cmd(cmd, description=""):
    """Run command and return success status"""
    if description:
        print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        if description:
            print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        if description:
            print(f"❌ {description} - Failed")
        return False

print("🚀 Installing MonoX dependencies for Colab (Python 3.12 compatible)...")

# Strategy: Use newer Hydra that's compatible with Python 3.12
print("\n📦 Installing Python 3.12 compatible versions...")

# Install newer compatible versions
success = True
success &= run_cmd("pip install 'hydra-core>=1.2.0'", "Installing newer Hydra (Python 3.12 compatible)")
success &= run_cmd("pip install 'omegaconf>=2.2.0'", "Installing newer OmegaConf (Python 3.12 compatible)")

if not success:
    print("\n🔄 Trying alternative approach...")
    # Alternative: Use specific known working versions
    run_cmd("pip install hydra-core==1.3.2", "Installing Hydra 1.3.2")
    run_cmd("pip install omegaconf==2.3.0", "Installing OmegaConf 2.3.0")

# Install other dependencies
print("\n📦 Installing PyTorch and other dependencies...")
run_cmd("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch with CUDA")
run_cmd("pip install 'pytorch-lightning>=1.5.0,<2.0.0'", "Installing PyTorch Lightning")

# Install remaining ML dependencies
deps = [
    "numpy", "pillow", "scipy", "tqdm", "tensorboard", "matplotlib",
    "opencv-python", "imageio", "imageio-ffmpeg", "ninja", "psutil"
]

for dep in deps:
    run_cmd(f"pip install {dep}", f"Installing {dep}")

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
    print(f"🔥 Python: {sys.version}")
    print("🎉 Setup complete! Ready for training.")
    
    # Check if we need to update configs for newer Hydra
    import packaging.version
    hydra_version = packaging.version.parse(hydra.__version__)
    if hydra_version >= packaging.version.parse("1.2.0"):
        print("\n⚠️  NOTE: Using newer Hydra version.")
        print("   The configs may need minor updates for compatibility.")
        print("   If you see any config errors, this is expected and we can fix them.")
        
except Exception as e:
    print(f"❌ Import test failed: {e}")
    print("\n🔄 Trying to fix import issues...")
    
    # Try to install specific working combination
    run_cmd("pip install hydra-core==1.3.2 omegaconf==2.3.0 --force-reinstall", "Force reinstalling compatible versions")
    
    try:
        import hydra
        import omegaconf
        print("✅ Fixed! Ready for training.")
    except Exception as e2:
        print(f"❌ Still failing: {e2}")
        print("⚠️  You may need to restart the runtime and try a different approach.")