#!/usr/bin/env python3
"""
Clean Colab Environment Setup for MonoX + StyleGAN-V Training
===========================================================

This script properly sets up the environment for training without assumptions
about virtual environments or existing installations.

Run this ONCE in Colab before training:
    !python colab_environment_setup.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

# Configuration
MONOX_ROOT = "/content/MonoX"
STYLEGAN_V_URL = "https://github.com/yakymchukluka-afk/stylegan-v.git" 
RESULTS_DIR = "/content/MonoX/results"
DATASET_DIR = "/content/MonoX/dataset"

def run_command(cmd, check=True, cwd=None):
    """Run command with proper error handling"""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=isinstance(cmd, str),
            check=check,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        if check:
            raise
        return e

def check_gpu():
    """Verify GPU is available and show details"""
    print("\n=== GPU Check ===")
    try:
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            print("✅ GPU detected successfully!")
            # Check CUDA availability in Python
            import torch
            if torch.cuda.is_available():
                print(f"✅ PyTorch CUDA available: {torch.version.cuda}")
                print(f"✅ GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                return True
            else:
                print("❌ PyTorch CUDA not available")
                return False
        else:
            print("❌ nvidia-smi failed - no GPU available")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n=== Setting up directories ===")
    
    dirs_to_create = [
        MONOX_ROOT,
        f"{MONOX_ROOT}/.external",
        RESULTS_DIR,
        f"{RESULTS_DIR}/logs",
        f"{RESULTS_DIR}/previews", 
        f"{RESULTS_DIR}/checkpoints",
        DATASET_DIR
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")

def install_dependencies():
    """Install required Python packages"""
    print("\n=== Installing dependencies ===")
    
    # First ensure pip is up to date
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install PyTorch with CUDA support
    run_command([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    
    # Install other core dependencies
    packages = [
        "hydra-core==1.3.2",  # Updated version
        "omegaconf==2.3.0",   # Updated version  
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "matplotlib>=3.4.0",
        "opencv-python>=4.5.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.5",
        "ninja>=1.10.0",
        "psutil>=5.8.0",
        "pytorch-lightning>=1.8.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0"
    ]
    
    for package in packages:
        run_command([sys.executable, "-m", "pip", "install", package])

def clone_stylegan_v():
    """Properly clone StyleGAN-V repository"""
    print("\n=== Cloning StyleGAN-V ===")
    
    stylegan_dir = f"{MONOX_ROOT}/.external/stylegan-v"
    
    # Remove existing directory if it exists but is incomplete
    if os.path.exists(stylegan_dir):
        if not os.path.exists(f"{stylegan_dir}/.git"):
            print(f"Removing incomplete StyleGAN-V directory: {stylegan_dir}")
            shutil.rmtree(stylegan_dir)
        else:
            print(f"StyleGAN-V already exists at {stylegan_dir}")
            return stylegan_dir
    
    # Clone the repository
    run_command([
        "git", "clone", "--recursive", 
        STYLEGAN_V_URL,
        stylegan_dir
    ])
    
    # Initialize and update submodules
    run_command(["git", "submodule", "update", "--init", "--recursive"], 
                cwd=stylegan_dir)
    
    print(f"✅ StyleGAN-V cloned to: {stylegan_dir}")
    return stylegan_dir

def setup_environment_vars():
    """Set up environment variables"""
    print("\n=== Setting up environment variables ===")
    
    env_vars = {
        "MONOX_ROOT": MONOX_ROOT,
        "DATASET_DIR": DATASET_DIR,
        "LOGS_DIR": f"{RESULTS_DIR}/logs",
        "PREVIEWS_DIR": f"{RESULTS_DIR}/previews", 
        "CKPT_DIR": f"{RESULTS_DIR}/checkpoints",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

def create_sample_dataset():
    """Create a minimal sample dataset for testing"""
    print("\n=== Creating sample dataset ===")
    
    # Create sample images directory
    sample_dir = f"{DATASET_DIR}/sample_images"
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a few simple test images using PIL
    try:
        from PIL import Image
        import numpy as np
        
        for i in range(10):
            # Create simple colored squares
            img_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f"{sample_dir}/sample_{i:03d}.png")
        
        print(f"✅ Created 10 sample images in {sample_dir}")
        return sample_dir
        
    except Exception as e:
        print(f"⚠️  Could not create sample dataset: {e}")
        return None

def verify_setup():
    """Verify the setup is complete and working"""
    print("\n=== Verifying setup ===")
    
    checks = []
    
    # Check directories
    for dir_path in [MONOX_ROOT, f"{MONOX_ROOT}/.external/stylegan-v", RESULTS_DIR]:
        if os.path.exists(dir_path):
            checks.append(f"✅ Directory exists: {dir_path}")
        else:
            checks.append(f"❌ Missing directory: {dir_path}")
    
    # Check Python path
    stylegan_path = f"{MONOX_ROOT}/.external/stylegan-v"
    if stylegan_path in sys.path:
        checks.append("✅ StyleGAN-V in Python path")
    else:
        sys.path.insert(0, stylegan_path)
        checks.append(f"✅ Added StyleGAN-V to Python path: {stylegan_path}")
    
    # Check imports
    try:
        import torch
        checks.append(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        checks.append("❌ PyTorch not available")
    
    try:
        import hydra
        checks.append(f"✅ Hydra: {hydra.__version__}")
    except ImportError:
        checks.append("❌ Hydra not available")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            checks.append("❌ CUDA not available")
    except:
        checks.append("❌ Could not check CUDA")
    
    for check in checks:
        print(check)
    
    return all("✅" in check for check in checks)

def main():
    """Main setup function"""
    print("🚀 MonoX + StyleGAN-V Colab Environment Setup")
    print("=" * 50)
    
    start_time = time.time()
    
    # Ensure we're in the right location
    if not os.path.exists(MONOX_ROOT):
        print(f"Creating MonoX directory: {MONOX_ROOT}")
        os.makedirs(MONOX_ROOT, exist_ok=True)
    
    # Change to MonoX directory
    os.chdir(MONOX_ROOT)
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # 1. Check GPU
        gpu_ok = check_gpu()
        if not gpu_ok:
            print("⚠️  Warning: No GPU detected. Training will be very slow!")
        
        # 2. Setup directories
        setup_directories()
        
        # 3. Install dependencies  
        install_dependencies()
        
        # 4. Clone StyleGAN-V
        stylegan_dir = clone_stylegan_v()
        
        # 5. Setup environment variables
        setup_environment_vars()
        
        # 6. Create sample dataset
        create_sample_dataset()
        
        # 7. Verify setup
        setup_ok = verify_setup()
        
        elapsed = time.time() - start_time
        
        if setup_ok:
            print(f"\n🎉 Setup completed successfully in {elapsed:.1f}s!")
            print(f"\nNext steps:")
            print(f"1. Upload your dataset to: {DATASET_DIR}")
            print(f"2. Run training with: !python colab_training_launcher.py")
            print(f"3. Monitor results in: {RESULTS_DIR}")
        else:
            print(f"\n❌ Setup completed with errors in {elapsed:.1f}s")
            print("Please check the output above for issues.")
            
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()