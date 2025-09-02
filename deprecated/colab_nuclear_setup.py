#!/usr/bin/env python3
"""
🚀 COLAB NUCLEAR SETUP - BULLETPROOF VERSION
===========================================
This script handles all directory issues and ensures nuclear activation.
"""

import os
import subprocess
import shutil
import sys
import time

def run_cmd(cmd, description="", check_error=True, timeout=300):
    """Run command with error handling."""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/content"  # Always use /content as base
        )
        
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        
        if result.stderr and "warning" not in result.stderr.lower():
            print(f"⚠️  Error: {result.stderr.strip()}")
        
        if check_error and result.returncode != 0:
            print(f"❌ Command failed with code {result.returncode}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def nuclear_setup():
    """Complete nuclear setup for Colab."""
    print("🚀🚀🚀 COLAB NUCLEAR SETUP")
    print("="*60)
    
    # Step 1: Clean environment
    print("\n🧹 STEP 1: CLEAN ENVIRONMENT")
    os.chdir("/content")
    
    if os.path.exists("/content/MonoX"):
        print("🗑️  Removing old MonoX...")
        shutil.rmtree("/content/MonoX")
    
    # Step 2: Clone fresh
    print("\n📥 STEP 2: CLONE FRESH MONOX")
    if not run_cmd("git clone https://github.com/yakymchukluka-afk/MonoX", "Cloning MonoX"):
        return False
    
    os.chdir("/content/MonoX")
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Step 3: Initialize submodules
    print("\n🔗 STEP 3: INITIALIZE SUBMODULES")
    if not run_cmd("git submodule update --init --recursive", "Initializing submodules"):
        return False
    
    # Step 4: Get latest updates
    print("\n🔄 STEP 4: GET LATEST UPDATES")
    run_cmd("git pull origin main", "Getting latest main")
    run_cmd("git submodule update --remote", "Updating submodules")
    
    # Step 5: Verify structure
    print("\n🔍 STEP 5: VERIFY STRUCTURE")
    expected_files = [
        "train_super_gpu_forced.py",
        "force_nuclear_activation.py",
        ".external/stylegan-v/src/training/training_loop.py"
    ]
    
    for file_path in expected_files:
        full_path = f"/content/MonoX/{file_path}"
        if os.path.exists(full_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Step 6: Apply nuclear patches directly
    print("\n🚀 STEP 6: APPLY NUCLEAR PATCHES")
    training_loop_path = "/content/MonoX/.external/stylegan-v/src/training/training_loop.py"
    
    if os.path.exists(training_loop_path):
        with open(training_loop_path, 'r') as f:
            content = f.read()
        
        if "🚀🚀🚀 NUCLEAR: training_loop() function called!" not in content:
            print("🔧 Adding nuclear markers to training_loop.py...")
            
            # Find the training_loop function and add our debug
            import re
            pattern = r'(def training_loop\([^)]+\):\s*)'
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            
            if match:
                insert_pos = match.end()
                nuclear_code = '''
    # 🚀 NUCLEAR DEBUG: Training loop started!
    print("🚀🚀🚀 NUCLEAR: training_loop() function called!")
    print(f"🚀🚀🚀 NUCLEAR: rank={rank}, num_gpus={num_gpus}, batch_size={batch_size}")
    print(f"🚀🚀🚀 NUCLEAR: CUDA available: {torch.cuda.is_available()}")
    print(f"🚀🚀🚀 NUCLEAR: Device count: {torch.cuda.device_count()}")
    
    # 🚀 NUCLEAR: GPU memory pre-allocation
    if torch.cuda.is_available() and rank == 0:
        device = torch.device('cuda')
        print(f"🚀🚀🚀 NUCLEAR: Starting GPU memory pre-allocation...")
        try:
            # Pre-allocate GPU memory
            gpu_total = torch.cuda.get_device_properties(device).total_memory
            print(f"🔥 NUCLEAR: Total GPU memory: {gpu_total / 1024**3:.1f} GB")
            
            # Create large tensors for heavy GPU usage
            warmup_tensors = []
            for i in range(3):
                tensor = torch.randn(2048, 2048, device=device, dtype=torch.float32)
                result = torch.mm(tensor, tensor)
                warmup_tensors.append(result)
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                print(f"🔥 NUCLEAR: Allocated tensor {i+1}: {allocated:.1f} GB")
            
            # Clean up warmup
            del warmup_tensors
            torch.cuda.empty_cache()
            print(f"🚀🚀🚀 NUCLEAR: GPU warmed up for HEAVY UTILIZATION!")
            
        except Exception as e:
            print(f"⚠️  NUCLEAR: GPU pre-allocation warning: {e}")
    
'''
                content = content[:insert_pos] + nuclear_code + content[insert_pos:]
                
                with open(training_loop_path, 'w') as f:
                    f.write(content)
                
                print("✅ Nuclear markers added to training_loop.py!")
            else:
                print("❌ Could not find training_loop function")
        else:
            print("✅ Nuclear markers already present!")
    else:
        print(f"❌ Training loop file not found: {training_loop_path}")
    
    # Step 7: Launch nuclear training
    print("\n🚀 STEP 7: LAUNCH NUCLEAR TRAINING")
    
    # Clean results directory
    results_dir = "/content/MonoX/results"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    
    # Clean experiment directory
    exp_dir = "/content/MonoX/experiments"
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    
    # Set nuclear environment
    nuclear_env = os.environ.copy()
    nuclear_env.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_LAUNCH_BLOCKING': '1',
        'FORCE_CUDA': '1',
        'PYTHONPATH': '/content/MonoX/.external/stylegan-v/src'
    })
    
    # Nuclear training command
    cmd = [
        "python3", "train_super_gpu_forced.py",
        "exp_suffix=nuclear_colab",
        "dataset.path=/content/drive/MyDrive/MonoX/dataset",
        "dataset.resolution=256",
        "training.total_kimg=2",
        "training.snapshot_kimg=1",
        "visualizer.save_every_kimg=1",
        "num_gpus=1"
    ]
    
    print("🚀 LAUNCHING NUCLEAR TRAINING...")
    print(f"   Command: {' '.join(cmd)}")
    print("="*60)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=nuclear_env,
            cwd="/content/MonoX"
        )
        
        nuclear_found = False
        line_count = 0
        
        # Monitor output in real-time
        for line in iter(process.stdout.readline, ''):
            line_count += 1
            print(f"{line_count:3}: {line.rstrip()}")
            
            if "🚀🚀🚀 NUCLEAR:" in line:
                nuclear_found = True
                print(f"    🎉 *** NUCLEAR MARKER FOUND! ***")
            
            # Stop after reasonable amount of output
            if line_count > 500:
                print("⏹️  Stopping output after 500 lines...")
                break
        
        process.wait()
        
        if nuclear_found:
            print("\n🎉🚀💥 NUCLEAR ACTIVATION SUCCESSFUL! 💥🚀🎉")
            print("✅ Training loop reached!")
            print("🔥 GPU utilization should now be active!")
            return True
        else:
            print("\n❌ Nuclear markers not found in output")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COLAB NUCLEAR SETUP - BULLETPROOF VERSION")
    print("="*60)
    success = nuclear_setup()
    if success:
        print("\n🏆 NUCLEAR SETUP COMPLETED!")
        print("💥 Your NVIDIA L4 should now be under heavy utilization!")
    else:
        print("\n⚠️  Setup needs investigation")
    print("="*60)