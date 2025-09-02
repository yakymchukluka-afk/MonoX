#!/usr/bin/env python3
"""
🔥💥💀🚀 DIRECT TRAINING LAUNCHER - NO PATCHING NEEDED! 🚀💀💥🔥
================================================================================
This script launches training directly using the working configuration!
"""

import os
import subprocess
import sys
import time

def direct_training_launcher():
    """Launch training directly with the working configuration."""
    print("🔥💥💀🚀 DIRECT TRAINING LAUNCHER - NO PATCHING NEEDED!")
    print("=" * 80)
    
    # Force to /content directory
    os.chdir("/content")
    
    # Step 1: Set optimal environment
    print("\n🌍 STEP 1: SET OPTIMAL ENVIRONMENT")
    env_vars = {
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_VISIBLE_DEVICES': '0',
        'FORCE_CUDA': '1',
        'USE_CUDA': '1',
        'PYTHONPATH': '/content/MonoX/.external/stylegan-v',
        'MAX_JOBS': '4',  # Optimized
    }
    
    for var, val in env_vars.items():
        os.environ[var] = val
        print(f"✅ {var}={val}")
    
    # Step 2: Ensure directories exist
    print("\n📁 STEP 2: ENSURE DIRECTORIES EXIST")
    required_dirs = [
        "/content/MonoX/results",
        "/content/MonoX/results/output",
        "/content/MonoX/logs"
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Directory ready: {dir_path}")
        except Exception as e:
            print(f"⚠️ Directory warning: {e}")
    
    # Step 3: Check StyleGAN-V setup
    print("\n🔍 STEP 3: CHECK STYLEGAN-V SETUP")
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    if os.path.exists(stylegan_dir):
        print(f"✅ StyleGAN-V directory found: {stylegan_dir}")
        
        # Check for key files
        key_files = [
            "src/infra/launch.py",
            "src/train.py",
            "configs/config.yaml"
        ]
        
        for file_path in key_files:
            full_path = os.path.join(stylegan_dir, file_path)
            if os.path.exists(full_path):
                print(f"✅ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
    else:
        print(f"❌ StyleGAN-V directory not found: {stylegan_dir}")
        return False
    
    # Step 4: Launch optimized training
    print("\n🚀 STEP 4: LAUNCH OPTIMIZED TRAINING")
    
    # High-performance command (no patching needed!)
    training_cmd = [
        'python3', '-m', 'src.infra.launch',
        'hydra.run.dir=logs',
        'exp_suffix=direct_optimized',
        'dataset.path=/content/drive/MyDrive/MonoX/dataset',
        'dataset.resolution=256',
        'training.kimg=25000',  # Full training
        'training.snap=200',    # Save every 200 kimg
        'num_gpus=1',
        '++training.gpus=1',
        '++training.batch_size=4',  # Optimized batch size
        '++training.fp32=false',
        '++training.nobench=false',
        '++training.allow_tf32=false',
        '++training.metrics=[fid50k_full]',
        '++training.seed=0',
        '++training.data=/content/drive/MyDrive/MonoX/dataset',
        '++model.loss_kwargs.source=StyleGAN2Loss',
        '++model.loss_kwargs.style_mixing_prob=0.0',
        '++model.discriminator.mbstd_group_size=4',  # Standard size
        '++model.discriminator.source=networks',
        '++model.generator.source=networks',
        '++model.generator.w_dim=512',
        '+model.optim.generator.lr=0.002',
        '+model.optim.discriminator.lr=0.002',
        '++training.num_workers=4',  # Optimized workers
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
    
    print("🚀 LAUNCHING DIRECT OPTIMIZED TRAINING...")
    print(f"📂 Working directory: {stylegan_dir}")
    print("🔥 Optimized settings: batch_size=4, workers=4, full training...")
    print("=" * 80)
    
    try:
        # Set environment and run
        env = os.environ.copy()
        env['PYTHONPATH'] = '/content/MonoX/.external/stylegan-v'
        
        process = subprocess.Popen(
            training_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=stylegan_dir
        )
        
        # Monitor for success patterns
        training_started = False
        gpu_detected = False
        actual_training = False
        networks_ready = False
        line_count = 0
        
        for line in iter(process.stdout.readline, ''):
            line_count += 1
            print(f"{line_count:3}: {line.rstrip()}")
            
            # Look for success markers
            if "Creating output directory..." in line:
                training_started = True
                print("    🚀 *** TRAINING INITIALIZATION! ***")
            
            if "Number of GPUs" in line or "CUDA" in line:
                gpu_detected = True
                print("    🔥 *** GPU DETECTED! ***")
            
            # Look for actual training loop start
            if any(marker in line for marker in ["Loading training set", "Constructing networks", "Launching processes"]):
                actual_training = True
                print("    ✅ *** ACTUAL TRAINING STARTED! ***")
            
            # Look for successful network construction
            if "Generator" in line and "Parameters" in line:
                networks_ready = True
                print("    🏗️ *** NETWORKS READY! ***")
            
            # Look for our nuclear markers (if they appear)
            if "🏆💥🚀💀🎯✨🌟🔥💯" in line:
                print("    🏆 *** NUCLEAR SUCCESS MARKER! ***")
            
            # Look for training tick (kimg progress)
            if "tick" in line.lower() and "kimg" in line.lower():
                print("    📈 *** TRAINING PROGRESS! ***")
            
            # Stop after reasonable output or if training is clearly progressing
            if line_count > 500 or (networks_ready and actual_training):
                print("⏹️ Training launched successfully - continuing in background...")
                break
        
        # Evaluate success
        if actual_training:
            print(f"\n🔥💥💀🚀 DIRECT TRAINING LAUNCHER SUCCESS!")
            print("✅ Training launched successfully!")
            print("🔥 GPU training should now be active!")
            print("📊 Check progress with: !nvidia-smi")
            return True
        elif training_started:
            print(f"\n🔥 TRAINING INITIALIZATION SUCCESS!")
            print("✅ Training started - may be loading data...")
            print("🔍 Monitor progress in the background!")
            return True
        else:
            print(f"\n⚠️ Training launch needs verification")
            print("🔍 Check the output above for any errors")
            return False
            
    except Exception as e:
        print(f"❌ Training launch error: {e}")
        return False

def check_current_training():
    """Check if training is already running."""
    print("\n🔍 CHECKING FOR EXISTING TRAINING...")
    
    try:
        # Check for python processes running StyleGAN
        result = subprocess.run(['pgrep', '-f', 'stylegan'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training process found running!")
            print("PID(s):", result.stdout.strip())
            return True
        else:
            print("ℹ️ No training process currently running")
            return False
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
        return False

if __name__ == "__main__":
    print("🔥💥💀🚀 DIRECT TRAINING LAUNCHER - NO PATCHING NEEDED!")
    print("=" * 80)
    
    # Check if training is already running
    if check_current_training():
        print("\n🎉 TRAINING IS ALREADY RUNNING!")
        print("🔥 Check GPU usage: !nvidia-smi")
        print("📊 Check results: !ls -la /content/MonoX/results/")
        print("🔍 To start new training, kill existing process first")
    else:
        success = direct_training_launcher()
        
        if success:
            print("\n🔥💥💀🚀 DIRECT TRAINING LAUNCHER SUCCESSFUL!")
            print("✅ Training launched without any patching needed!")
            print("🔥 GPU should now be actively training!")
            print("🔥 Check: !nvidia-smi")
            print("🔥 Check: !ls -la /content/MonoX/results/")
        else:
            print("\n⚠️ Training launch may need manual verification")
            print("🔍 Try running the command manually in StyleGAN-V directory")
    
    print("=" * 80)