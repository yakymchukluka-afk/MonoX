#!/usr/bin/env python3
"""
Verification script for StyleGAN-V training environment
Checks all dependencies, configurations, and directory structure
"""

import os
import sys
import importlib
from pathlib import Path

def check_import(package_name, description=""):
    """Check if a package can be imported"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} {description}")
        return True
    except ImportError:
        print(f"❌ {package_name} {description} - NOT FOUND")
        return False

def check_directory(path, description=""):
    """Check if a directory exists"""
    if os.path.exists(path):
        print(f"✅ {path} {description}")
        return True
    else:
        print(f"❌ {path} {description} - NOT FOUND")
        return False

def check_file(path, description=""):
    """Check if a file exists"""
    if os.path.isfile(path):
        print(f"✅ {path} {description}")
        return True
    else:
        print(f"❌ {path} {description} - NOT FOUND")
        return False

def main():
    print("🔍 StyleGAN-V Environment Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check Python packages
    print("\n📦 Checking Python Dependencies:")
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("cv2", "OpenCV"),
        ("hydra", "Hydra"),
        ("scipy", "SciPy"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
        ("tensorboard", "TensorBoard"),
        ("av", "PyAV")
    ]
    
    for pkg, desc in packages:
        if not check_import(pkg, desc):
            all_good = False
    
    # Check PyTorch/CUDA setup
    print("\n🔥 Checking PyTorch/CUDA Setup:")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA not available - training will be very slow on CPU")
    except Exception as e:
        print(f"❌ PyTorch setup error: {e}")
        all_good = False
    
    # Check directory structure
    print("\n📁 Checking Directory Structure:")
    directories = [
        ("/workspace/stylegan-stack/data/originals", "Dataset directory"),
        ("/workspace/stylegan-stack/models/checkpoints", "Checkpoints directory"),
        ("/workspace/stylegan-stack/logs", "Logs directory"),
        ("/workspace/stylegan-stack/generated_previews", "Previews directory"),
        ("/workspace/stylegan-stack/stylegan-v", "StyleGAN-V source"),
        ("/workspace/stylegan-stack/training", "Training documentation")
    ]
    
    for dir_path, desc in directories:
        if not check_directory(dir_path, desc):
            all_good = False
    
    # Check configuration files
    print("\n⚙️  Checking Configuration Files:")
    config_files = [
        ("/workspace/stylegan-stack/training_config.yaml", "Main training config"),
        ("/workspace/stylegan-stack/stylegan-v/configs/dataset/mono_project.yaml", "Dataset config"),
        ("/workspace/stylegan-stack/stylegan-v/configs/training/mono_training.yaml", "Training config"),
        ("/workspace/stylegan-stack/train_styleganv.sh", "Training script"),
        ("/workspace/stylegan-stack/training/README.md", "Documentation")
    ]
    
    for file_path, desc in config_files:
        if not check_file(file_path, desc):
            all_good = False
    
    # Check StyleGAN-V installation
    print("\n🎨 Checking StyleGAN-V Installation:")
    try:
        # Check if we can access StyleGAN-V source
        stylegan_path = "/workspace/stylegan-stack/stylegan-v"
        if os.path.exists(os.path.join(stylegan_path, "src")):
            sys.path.insert(0, stylegan_path)
            import src
            print("✅ StyleGAN-V source code accessible")
        else:
            print("❌ StyleGAN-V source not found")
            all_good = False
    except ImportError as e:
        print(f"❌ StyleGAN-V import error: {e}")
        all_good = False
    
    # Dataset check
    print("\n📊 Checking Dataset:")
    dataset_path = "/workspace/stylegan-stack/data/originals"
    if os.path.exists(dataset_path):
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if subdirs:
            print(f"✅ Found {len(subdirs)} video directories in dataset")
        else:
            print("⚠️  Dataset directory exists but no video subdirectories found")
            print("   Upload your dataset before starting training")
    else:
        print("⚠️  Dataset directory not found - will be created when needed")
    
    # Final status
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! Environment is ready for training.")
        print("\nNext steps:")
        print("1. Upload your dataset to /workspace/stylegan-stack/data/originals/")
        print("2. Run: ./train_styleganv.sh")
    else:
        print("❌ Some issues found. Please fix them before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()