#!/usr/bin/env python3
"""
Quick Verification Script for MonoX Setup
=========================================

Run this to verify that MonoX is properly set up and ready for training.

Usage:
    !python /content/MonoX/verify_monox_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

def check_directories():
    """Check if all required directories exist"""
    print("📁 Checking directories...")
    
    required_dirs = [
        "/content/MonoX",
        "/content/MonoX/.external/stylegan-v",
        "/content/MonoX/configs",
        "/content/MonoX/results/logs",
        "/content/MonoX/results/previews", 
        "/content/MonoX/results/checkpoints",
        "/content/MonoX/dataset"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (MISSING)")
            all_good = False
    
    return all_good

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU...")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi works")
            # Extract GPU name
            for line in result.stdout.split('\n'):
                if any(gpu in line for gpu in ['Tesla', 'RTX', 'GTX', 'A100', 'V100', 'L4']):
                    print(f"   {line.strip()}")
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
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   GPU name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("❌ PyTorch CUDA not available")
            return False
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        return False

def check_python_modules():
    """Check if required Python modules can be imported"""
    print("\n🐍 Checking Python modules...")
    
    modules = [
        ("torch", "PyTorch"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy")
    ]
    
    all_good = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (NOT INSTALLED)")
            all_good = False
    
    return all_good

def check_stylegan_v():
    """Check if StyleGAN-V can be imported"""
    print("\n🎨 Checking StyleGAN-V...")
    
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    
    # Check if directory exists and has content
    if not os.path.exists(stylegan_dir):
        print("❌ StyleGAN-V directory missing")
        return False
    
    if not os.path.exists(f"{stylegan_dir}/.git"):
        print("❌ StyleGAN-V not properly cloned (no .git)")
        return False
    
    print("✅ StyleGAN-V directory exists")
    
    # Add to Python path and try import
    if stylegan_dir not in sys.path:
        sys.path.insert(0, stylegan_dir)
    
    try:
        import src
        print("✅ src module importable")
    except ImportError as e:
        print(f"❌ src module import failed: {e}")
        return False
    
    try:
        import src.infra.launch
        print("✅ src.infra.launch importable")
        return True
    except ImportError as e:
        print(f"❌ src.infra.launch import failed: {e}")
        return False

def check_configs():
    """Check if config files exist and are valid"""
    print("\n⚙️  Checking configuration files...")
    
    config_files = [
        "/content/MonoX/configs/config.yaml",
        "/content/MonoX/configs/dataset/base.yaml",
        "/content/MonoX/configs/training/base.yaml"
    ]
    
    all_good = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {os.path.basename(config_file)}")
        else:
            print(f"❌ {os.path.basename(config_file)} (MISSING)")
            all_good = False
    
    return all_good

def test_training_command():
    """Test if the training command can be constructed"""
    print("\n🧪 Testing training command...")
    
    # Setup environment
    os.environ.update({
        "DATASET_DIR": "/content/MonoX/dataset",
        "LOGS_DIR": "/content/MonoX/results/logs",
        "PYTHONPATH": "/content/MonoX/.external/stylegan-v:/content/MonoX"
    })
    
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        "--config-path", "/content/MonoX/configs",
        "--config-name", "config",
        "--help"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/content/MonoX/.external/stylegan-v"
        )
        
        if result.returncode == 0:
            print("✅ Training command works")
            return True
        else:
            print(f"❌ Training command failed:")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training command timed out")
        return False
    except Exception as e:
        print(f"❌ Training command error: {e}")
        return False

def quick_gpu_test():
    """Quick GPU memory test"""
    print("\n⚡ Quick GPU test...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            x = torch.randn(100, 100).to(device)
            print("✅ GPU memory allocation works")
            del x
            torch.cuda.empty_cache()
            return True
        else:
            print("❌ CUDA not available for testing")
            return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 MonoX Setup Verification")
    print("=" * 40)
    
    checks = [
        ("Directories", check_directories),
        ("GPU", check_gpu),
        ("Python Modules", check_python_modules),
        ("StyleGAN-V", check_stylegan_v),
        ("Config Files", check_configs),
        ("Training Command", test_training_command),
        ("GPU Test", quick_gpu_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name.upper()}")
        print("-" * len(check_name))
        
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"💥 {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print(f"\n📋 VERIFICATION SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready to train.")
        print("\nTo start training run:")
        print("   !python /content/MonoX/launch_training.py")
        return True
    else:
        print("\n⚠️  Some checks failed. Try running setup again:")
        print("   !python /content/MonoX/setup_monox_colab.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)