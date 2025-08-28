#!/usr/bin/env python3
"""
Standalone Debug Checklist for MonoX + StyleGAN-V
================================================

Run this to diagnose issues without starting training.

Usage:
    !python colab_debug_checklist.py
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import time

# Configuration
MONOX_ROOT = "/content/MonoX"
STYLEGAN_V_DIR = "/content/MonoX/.external/stylegan-v"
RESULTS_DIR = "/content/MonoX/results"

def check_directories():
    """Check if all required directories exist"""
    print("📁 Checking Directories")
    print("-" * 30)
    
    required_dirs = [
        (MONOX_ROOT, "MonoX root"),
        (f"{MONOX_ROOT}/.external", "External dependencies"),
        (STYLEGAN_V_DIR, "StyleGAN-V"),
        (f"{STYLEGAN_V_DIR}/.git", "StyleGAN-V git repo"),
        (f"{RESULTS_DIR}/logs", "Logs directory"),
        (f"{RESULTS_DIR}/previews", "Previews directory"),
        (f"{RESULTS_DIR}/checkpoints", "Checkpoints directory"),
        (f"{MONOX_ROOT}/configs", "Configs directory"),
        (f"{MONOX_ROOT}/src", "Source directory")
    ]
    
    all_good = True
    for dir_path, description in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {description}: {dir_path}")
        else:
            print(f"❌ {description}: {dir_path} (MISSING)")
            all_good = False
    
    return all_good

def check_python_environment():
    """Check Python environment and packages"""
    print("\n🐍 Checking Python Environment")
    print("-" * 30)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    print(f"PYTHONPATH: {pythonpath}")
    
    if STYLEGAN_V_DIR in pythonpath:
        print("✅ StyleGAN-V in PYTHONPATH")
    else:
        print("❌ StyleGAN-V NOT in PYTHONPATH")
        print(f"   Expected: {STYLEGAN_V_DIR}")
    
    # Check sys.path
    if STYLEGAN_V_DIR in sys.path:
        print("✅ StyleGAN-V in sys.path")
    else:
        print("⚠️  StyleGAN-V NOT in sys.path (will be added)")
        sys.path.insert(0, STYLEGAN_V_DIR)
    
    return True

def check_required_packages():
    """Check if all required packages are installed"""
    print("\n📦 Checking Required Packages")
    print("-" * 30)
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("tqdm", "TQDM")
    ]
    
    all_installed = True
    for package, description in required_packages:
        try:
            if package == "hydra":
                import hydra
                print(f"✅ {description}: {hydra.__version__}")
            elif package == "omegaconf":
                import omegaconf
                print(f"✅ {description}: {omegaconf.__version__}")
            elif package == "torch":
                import torch
                print(f"✅ {description}: {torch.__version__}")
            elif package == "torchvision":
                import torchvision
                print(f"✅ {description}: {torchvision.__version__}")
            elif package == "numpy":
                import numpy
                print(f"✅ {description}: {numpy.__version__}")
            elif package == "PIL":
                import PIL
                print(f"✅ {description}: {PIL.__version__}")
            elif package == "cv2":
                import cv2
                print(f"✅ {description}: {cv2.__version__}")
            elif package == "tqdm":
                import tqdm
                print(f"✅ {description}: {tqdm.__version__}")
            else:
                __import__(package)
                print(f"✅ {description}: installed")
        except ImportError:
            print(f"❌ {description}: NOT INSTALLED")
            all_installed = False
        except Exception as e:
            print(f"⚠️  {description}: {e}")
    
    return all_installed

def check_gpu_cuda():
    """Check GPU and CUDA availability"""
    print("\n🖥️  Checking GPU and CUDA")
    print("-" * 30)
    
    # Check nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi available")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Tesla' in line or 'RTX' in line or 'GTX' in line or 'A100' in line or 'V100' in line or 'L4' in line:
                    print(f"   GPU: {line.strip()}")
        else:
            print("❌ nvidia-smi failed")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
        return False
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # Test GPU memory
            try:
                device = torch.device('cuda:0')
                x = torch.randn(100, 100).to(device)
                print(f"✅ GPU memory test passed")
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️  GPU memory test failed: {e}")
                
            return True
        else:
            print("❌ PyTorch CUDA not available")
            return False
    except Exception as e:
        print(f"❌ PyTorch CUDA check failed: {e}")
        return False

def check_stylegan_v_modules():
    """Check if StyleGAN-V modules can be imported"""
    print("\n🎨 Checking StyleGAN-V Modules")
    print("-" * 30)
    
    # Ensure StyleGAN-V is in path
    if STYLEGAN_V_DIR not in sys.path:
        sys.path.insert(0, STYLEGAN_V_DIR)
    
    modules_to_check = [
        "src",
        "src.infra",
        "src.infra.launch"
    ]
    
    all_good = True
    for module_name in modules_to_check:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"✅ {module_name}: found at {spec.origin}")
            else:
                print(f"❌ {module_name}: not found")
                all_good = False
        except Exception as e:
            print(f"❌ {module_name}: error {e}")
            all_good = False
    
    # Try to actually import the launch module
    try:
        import src.infra.launch
        print("✅ src.infra.launch: successfully imported")
    except Exception as e:
        print(f"❌ src.infra.launch import failed: {e}")
        all_good = False
    
    return all_good

def check_config_files():
    """Check if config files are valid"""
    print("\n⚙️  Checking Configuration Files")
    print("-" * 30)
    
    config_files = [
        f"{MONOX_ROOT}/configs/config.yaml",
        f"{MONOX_ROOT}/configs/config_clean.yaml",
        f"{MONOX_ROOT}/configs/dataset/base.yaml",
        f"{MONOX_ROOT}/configs/training/base.yaml",
        f"{MONOX_ROOT}/configs/visualizer/base.yaml"
    ]
    
    all_good = True
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                # Try to parse with OmegaConf
                from omegaconf import OmegaConf
                conf = OmegaConf.load(config_file)
                print(f"✅ {os.path.basename(config_file)}: valid YAML")
            except Exception as e:
                print(f"❌ {os.path.basename(config_file)}: invalid YAML - {e}")
                all_good = False
        else:
            print(f"❌ {os.path.basename(config_file)}: missing")
            all_good = False
    
    return all_good

def check_dataset():
    """Check dataset availability"""
    print("\n📊 Checking Dataset")
    print("-" * 30)
    
    dataset_dir = os.environ.get("DATASET_DIR", f"{MONOX_ROOT}/dataset")
    print(f"Dataset directory: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory does not exist: {dataset_dir}")
        return False
    
    # Count image files
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(dataset_dir).rglob(f"*{ext}")))
        image_files.extend(list(Path(dataset_dir).rglob(f"*{ext.upper()}")))
    
    if image_files:
        print(f"✅ Found {len(image_files)} image files")
        
        # Check a few sample images
        for i, img_file in enumerate(image_files[:3]):
            try:
                from PIL import Image
                img = Image.open(img_file)
                print(f"   Sample {i+1}: {img_file.name} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                print(f"   Sample {i+1}: {img_file.name} (ERROR: {e})")
        
        return True
    else:
        print(f"❌ No image files found in {dataset_dir}")
        return False

def check_environment_variables():
    """Check required environment variables"""
    print("\n🌍 Checking Environment Variables")
    print("-" * 30)
    
    required_vars = [
        ("DATASET_DIR", "Dataset directory"),
        ("LOGS_DIR", "Logs directory"),
        ("PREVIEWS_DIR", "Previews directory"),
        ("CKPT_DIR", "Checkpoints directory"),
        ("CUDA_VISIBLE_DEVICES", "CUDA devices")
    ]
    
    all_set = True
    for var_name, description in required_vars:
        value = os.environ.get(var_name)
        if value:
            print(f"✅ {var_name}: {value}")
        else:
            print(f"⚠️  {var_name}: not set")
            all_set = False
    
    return all_set

def test_minimal_training_command():
    """Test if the training command can be constructed"""
    print("\n🧪 Testing Training Command")
    print("-" * 30)
    
    try:
        cmd = [
            sys.executable, "-m", "src.infra.launch",
            "--config-path", f"{MONOX_ROOT}/configs",
            "--config-name", "config_clean",
            "--help"
        ]
        
        print(f"Test command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=STYLEGAN_V_DIR,
            env=dict(os.environ, PYTHONPATH=f"{STYLEGAN_V_DIR}:{os.environ.get('PYTHONPATH', '')}")
        )
        
        if result.returncode == 0:
            print("✅ Training command help works")
            return True
        else:
            print(f"❌ Training command failed:")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training command timed out")
        return False
    except Exception as e:
        print(f"❌ Training command error: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("🔍 MonoX StyleGAN-V Diagnostic Checklist")
    print("=" * 50)
    
    start_time = time.time()
    
    checks = [
        ("Directories", check_directories),
        ("Python Environment", check_python_environment),
        ("Required Packages", check_required_packages),
        ("GPU and CUDA", check_gpu_cuda),
        ("StyleGAN-V Modules", check_stylegan_v_modules),
        ("Configuration Files", check_config_files),
        ("Dataset", check_dataset),
        ("Environment Variables", check_environment_variables),
        ("Training Command", test_minimal_training_command)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{check_name.upper()}")
        print("=" * len(check_name))
        
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"💥 {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n📋 DIAGNOSTIC SUMMARY ({elapsed:.1f}s)")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Ready for training.")
        return True
    else:
        print("⚠️  Some checks failed. Review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)