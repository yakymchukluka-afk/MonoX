#!/usr/bin/env python3
"""
🔍💥🔥 TRAINING DIAGNOSTICS - FIND OUT WHY TRAINING ISN'T RUNNING! 🔥💥🔍
================================================================================
This script diagnoses training issues and shows exactly what's happening!
"""

import os
import subprocess
import sys
import time

def training_diagnostics():
    """Comprehensive training diagnostics."""
    print("🔍💥🔥 TRAINING DIAGNOSTICS - FIND OUT WHY TRAINING ISN'T RUNNING!")
    print("=" * 80)
    
    # Step 1: Check environment
    print("\n📊 STEP 1: CHECK ENVIRONMENT")
    os.chdir("/content")
    
    # Check directories
    dirs_to_check = [
        "/content/MonoX",
        "/content/MonoX/.external/stylegan-v",
        "/content/MonoX/.external/stylegan-v/src",
        "/content/MonoX/.external/stylegan-v/configs",
        "/content/drive/MyDrive/MonoX/dataset"
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            try:
                files = os.listdir(dir_path)
                print(f"✅ {dir_path} - {len(files)} items")
            except Exception as e:
                print(f"⚠️ {dir_path} - exists but can't list: {e}")
        else:
            print(f"❌ {dir_path} - NOT FOUND")
    
    # Step 2: Check GPU
    print("\n🔥 STEP 2: CHECK GPU STATUS")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("❌ CUDA not available!")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
    
    # Step 3: Check dataset
    print("\n📁 STEP 3: CHECK DATASET")
    dataset_path = "/content/drive/MyDrive/MonoX/dataset"
    if os.path.exists(dataset_path):
        try:
            items = os.listdir(dataset_path)
            print(f"✅ Dataset found: {len(items)} items")
            print(f"📂 Contents: {items[:10]}...")  # Show first 10 items
        except Exception as e:
            print(f"⚠️ Dataset exists but can't access: {e}")
    else:
        print("❌ Dataset NOT FOUND!")
        print("🔍 Available drives:")
        if os.path.exists("/content/drive"):
            for item in os.listdir("/content/drive"):
                print(f"   - /content/drive/{item}")
    
    # Step 4: Try to run training with verbose output
    print("\n🚀 STEP 4: RUN TRAINING WITH VERBOSE OUTPUT")
    
    # Set environment
    env_vars = {
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTHONPATH': '/content/MonoX/.external/stylegan-v'
    }
    
    for var, val in env_vars.items():
        os.environ[var] = val
        print(f"✅ {var}={val}")
    
    # Change to StyleGAN-V directory
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    if not os.path.exists(stylegan_dir):
        print(f"❌ StyleGAN-V directory not found: {stylegan_dir}")
        return False
    
    os.chdir(stylegan_dir)
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Try a simple test first
    print("\n🧪 TESTING: Simple Python import")
    test_cmd = ['python3', '-c', 'import src; print("✅ StyleGAN-V imports work!")']
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ StyleGAN-V imports successfully!")
            print(result.stdout)
        else:
            print("❌ StyleGAN-V import failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"❌ Import test failed: {e}")
    
    # Try training with full output capture
    print("\n🚀 TESTING: Full training command with output")
    
    training_cmd = [
        'python3', '-m', 'src.infra.launch',
        'hydra.run.dir=logs',
        'exp_suffix=diagnostic_test',
        'dataset.path=/content/drive/MyDrive/MonoX/dataset',
        'dataset.resolution=256',
        'training.kimg=2',  # Very short test
        'training.snap=1',
        'num_gpus=1',
        '++training.gpus=1',
        '++training.batch_size=1',  # Minimal batch
        '++training.outdir=/content/MonoX/results',
        '--config-path=/content/MonoX/.external/stylegan-v/configs',
        '--config-name=config'
    ]
    
    print(f"🔥 Command: {' '.join(training_cmd)}")
    print("=" * 40)
    
    try:
        # Run with full output
        process = subprocess.Popen(
            training_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy()
        )
        
        line_count = 0
        for line in iter(process.stdout.readline, ''):
            line_count += 1
            print(f"{line_count:3}: {line.rstrip()}")
            
            # Look for specific error patterns
            if any(error in line.lower() for error in ['error', 'failed', 'not found', 'traceback']):
                print(f"    ❌ *** ERROR DETECTED: {line.strip()} ***")
            
            if any(success in line.lower() for success in ['loading', 'training', 'gpu', 'cuda']):
                print(f"    ✅ *** SUCCESS INDICATOR: {line.strip()} ***")
            
            # Stop after reasonable output
            if line_count > 100:
                print("⏹️ Stopping diagnostic after 100 lines...")
                process.terminate()
                break
        
        # Wait for process to finish
        return_code = process.wait()
        print(f"\n📊 Process completed with return code: {return_code}")
        
        if return_code == 0:
            print("✅ Command completed successfully!")
        else:
            print("❌ Command failed!")
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    # Step 5: Check for any created files
    print("\n📁 STEP 5: CHECK FOR CREATED FILES")
    
    check_paths = [
        "/content/MonoX/results",
        "/content/MonoX/logs", 
        "/content/MonoX/.external/stylegan-v/experiments",
        "/content/MonoX/.external/stylegan-v/logs"
    ]
    
    for path in check_paths:
        if os.path.exists(path):
            try:
                items = os.listdir(path)
                if items:
                    print(f"✅ {path}: {items}")
                else:
                    print(f"📂 {path}: empty")
            except Exception as e:
                print(f"⚠️ {path}: {e}")
        else:
            print(f"❌ {path}: not found")
    
    return True

def check_running_processes():
    """Check for any running training processes."""
    print("\n🔍 CHECKING FOR RUNNING PROCESSES")
    
    try:
        # Check for StyleGAN processes
        result = subprocess.run(['pgrep', '-f', 'stylegan'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ StyleGAN processes found:")
            for pid in result.stdout.strip().split('\n'):
                print(f"   PID: {pid}")
        else:
            print("ℹ️ No StyleGAN processes running")
        
        # Check for Python processes
        result = subprocess.run(['pgrep', '-f', 'python.*train'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training processes found:")
            for pid in result.stdout.strip().split('\n'):
                print(f"   PID: {pid}")
        else:
            print("ℹ️ No training processes running")
            
    except Exception as e:
        print(f"⚠️ Process check failed: {e}")

if __name__ == "__main__":
    print("🔍💥🔥 TRAINING DIAGNOSTICS - FIND OUT WHY TRAINING ISN'T RUNNING!")
    print("=" * 80)
    
    check_running_processes()
    
    success = training_diagnostics()
    
    if success:
        print("\n🔍💥🔥 DIAGNOSTICS COMPLETE!")
        print("✅ Check the output above for specific error messages")
        print("🔥 Focus on any lines marked with ❌ ERROR DETECTED")
        print("📊 The diagnostic should reveal exactly why training isn't starting")
    else:
        print("\n⚠️ Diagnostics encountered issues")
        print("🔍 Check the basic setup first")
    
    print("=" * 80)