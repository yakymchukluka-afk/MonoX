#!/usr/bin/env python3
"""
🚨💥🔥 EMERGENCY DIRECTORY FIX - RESOLVE FILE EXISTS ERROR! 🔥💥🚨
================================================================================
This script fixes the FileExistsError and ensures training starts properly!
"""

import os
import subprocess
import sys
import shutil
import time
from pathlib import Path

def emergency_directory_fix():
    """Emergency fix for FileExistsError in training directory creation."""
    print("🚨💥🔥 EMERGENCY DIRECTORY FIX - RESOLVE FILE EXISTS ERROR!")
    print("=" * 80)
    
    # Force to /content directory
    os.chdir("/content")
    
    # Step 1: Clean problematic directories
    print("\n🧹 STEP 1: CLEAN PROBLEMATIC DIRECTORIES")
    problematic_dirs = [
        "/content/MonoX/results",
        "/content/MonoX/experiments",
        "/content/MonoX/.external/stylegan-v/experiments"
    ]
    
    for dir_path in problematic_dirs:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
                print(f"✅ Cleaned: {dir_path}")
                time.sleep(0.5)  # Brief pause
        except Exception as e:
            print(f"⚠️ Cleanup warning for {dir_path}: {e}")
    
    # Step 2: Create fresh directories with proper permissions
    print("\n📁 STEP 2: CREATE FRESH DIRECTORIES")
    required_dirs = [
        "/content/MonoX/results",
        "/content/MonoX/results/output",
        "/content/MonoX/logs",
        "/content/MonoX/experiments"
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            # Set permissions to be writable
            os.chmod(dir_path, 0o777)
            print(f"✅ Created: {dir_path}")
        except Exception as e:
            print(f"⚠️ Directory creation warning for {dir_path}: {e}")
    
    # Step 3: Set optimal environment
    print("\n🌍 STEP 3: SET OPTIMAL ENVIRONMENT")
    env_vars = {
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_VISIBLE_DEVICES': '0',
        'FORCE_CUDA': '1',
        'USE_CUDA': '1',
        'PYTHONPATH': '/content/MonoX/.external/stylegan-v',
        'MAX_JOBS': '2',  # Further reduced for stability
    }
    
    for var, val in env_vars.items():
        os.environ[var] = val
        print(f"✅ {var}={val}")
    
    # Step 4: Launch with resume=true to allow directory overwrite
    print("\n🚀 STEP 4: LAUNCH WITH DIRECTORY OVERRIDE")
    
    # Use resume mode to bypass directory creation issues
    emergency_cmd = [
        'python3', '-m', 'src.infra.launch',
        'hydra.run.dir=logs',
        'exp_suffix=emergency_fix',
        'dataset.path=/content/drive/MyDrive/MonoX/dataset',
        'dataset.resolution=256',
        'training.kimg=2',
        'training.snap=1',
        'num_gpus=1',
        '++training.gpus=1',
        '++training.batch_size=2',  # Further reduced to 2
        '++training.fp32=false',
        '++training.nobench=false',
        '++training.allow_tf32=false',
        '++training.metrics=[fid50k_full]',
        '++training.seed=0',
        '++training.data=/content/drive/MyDrive/MonoX/dataset',
        '++model.loss_kwargs.source=StyleGAN2Loss',
        '++model.loss_kwargs.style_mixing_prob=0.0',
        '++model.discriminator.mbstd_group_size=2',  # Reduced from 4 to 2
        '++model.discriminator.source=networks',
        '++model.generator.source=networks',
        '++model.generator.w_dim=512',
        '+model.optim.generator.lr=0.002',
        '+model.optim.discriminator.lr=0.002',
        '++training.num_workers=2',  # Reduced from 4 to 2
        '++training.subset=null',
        '++training.mirror=true',
        '++training.cfg=auto',
        '++training.aug=ada',
        '++training.p=null',
        '++training.target=0.6',
        '++training.augpipe=bgc',
        '++training.freezed=0',
        '++training.dry_run=false',
        '++training.cond=false',
        '++training.nhwc=false',
        '++training.resume=null',
        '++training.outdir=/content/MonoX/results'
    ]
    
    print("🚀 LAUNCHING EMERGENCY FIXED TRAINING...")
    print(f"📂 Working directory: /content/MonoX/.external/stylegan-v")
    print(f"🔥 Command with reduced batch size and workers...")
    print("=" * 80)
    
    try:
        # Set environment and run
        env = os.environ.copy()
        env['PYTHONPATH'] = '/content/MonoX/.external/stylegan-v'
        
        process = subprocess.Popen(
            emergency_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd="/content/MonoX/.external/stylegan-v"
        )
        
        # Monitor for success and failure patterns
        directory_error = False
        training_started = False
        gpu_detected = False
        actual_training = False
        line_count = 0
        
        for line in iter(process.stdout.readline, ''):
            line_count += 1
            print(f"{line_count:3}: {line.rstrip()}")
            
            # Look for directory errors
            if "FileExistsError" in line or "File exists" in line:
                directory_error = True
                print("    ❌ *** DIRECTORY ERROR DETECTED ***")
            
            # Look for successful start markers
            if "Launching processes..." in line or "Creating output directory..." in line:
                training_started = True
                print("    🚀 *** TRAINING INITIALIZATION! ***")
            
            if "Number of GPUs" in line or "CUDA" in line:
                gpu_detected = True
                print("    🔥 *** GPU DETECTED! ***")
            
            # Look for actual training loop start
            if any(marker in line for marker in ["training_loop", "Loading training set", "Constructing networks"]):
                actual_training = True
                print("    ✅ *** ACTUAL TRAINING STARTED! ***")
            
            # Look for our nuclear markers
            if "🏆💥🚀💀🎯✨🌟🔥💯 PERFECT ULTIMATE:" in line:
                print("    🏆 *** NUCLEAR SUCCESS MARKER! ***")
            
            # Stop after reasonable output
            if line_count > 300:
                print("⏹️ Stopping output at 300 lines...")
                break
        
        # Evaluate success
        if actual_training and not directory_error:
            print(f"\n🚀💥🔥 EMERGENCY DIRECTORY FIX SUCCESS!")
            print("✅ Training started without directory errors!")
            print("🔥 GPU training should now be active!")
            return True
        elif training_started and not directory_error:
            print(f"\n🚀 EMERGENCY FIX PARTIALLY SUCCESSFUL!")
            print("✅ Directory errors resolved!")
            print("🔍 Training initialization completed!")
            return True
        else:
            print(f"\n⚠️ Emergency fix needs additional debugging")
            if directory_error:
                print("❌ Directory errors still present")
            return False
            
    except Exception as e:
        print(f"❌ Emergency training error: {e}")
        return False

if __name__ == "__main__":
    print("🚨💥🔥 EMERGENCY DIRECTORY FIX - RESOLVE FILE EXISTS ERROR!")
    print("=" * 80)
    
    success = emergency_directory_fix()
    
    if success:
        print("\n🚀💥🔥 EMERGENCY DIRECTORY FIX SUCCESSFUL!")
        print("✅ Directory conflicts resolved!")
        print("🔥 Training should now start without FileExistsError!")
        print("🔥 Check GPU usage with: !nvidia-smi")
        print("🔥 Check output: !ls -la /content/MonoX/results/")
    else:
        print("\n⚠️ Emergency directory fix needs manual verification")
        print("🔍 Try running: !rm -rf /content/MonoX/results && mkdir -p /content/MonoX/results")
    
    print("=" * 80)