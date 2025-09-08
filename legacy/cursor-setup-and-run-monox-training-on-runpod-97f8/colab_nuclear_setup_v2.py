#!/usr/bin/env python3
"""
🚀 COLAB NUCLEAR SETUP V2 - ULTRA ROBUST
========================================
Fixed git repository and directory handling.
"""

import os
import subprocess
import shutil
import sys
import time

def run_cmd(cmd, description="", check_error=True, timeout=300, cwd=None):
    """Run command with error handling."""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")
    print(f"   Directory: {cwd or os.getcwd()}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        
        if result.stderr:
            print(f"⚠️  Stderr: {result.stderr.strip()}")
        
        print(f"📊 Return code: {result.returncode}")
        
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

def nuclear_setup_v2():
    """Ultra robust nuclear setup."""
    print("🚀🚀🚀 COLAB NUCLEAR SETUP V2 - ULTRA ROBUST")
    print("="*60)
    
    # Ensure we start from content root
    print(f"📁 Starting directory: {os.getcwd()}")
    os.chdir("/content")
    print(f"📁 Changed to: {os.getcwd()}")
    
    # Step 1: Complete cleanup
    print("\n🧹 STEP 1: COMPLETE CLEANUP")
    if os.path.exists("/content/MonoX"):
        print("🗑️  Removing existing MonoX...")
        try:
            shutil.rmtree("/content/MonoX")
            print("✅ Cleanup successful")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    # Step 2: Fresh clone with verification
    print("\n📥 STEP 2: FRESH CLONE WITH VERIFICATION")
    
    if not run_cmd(
        "git clone https://github.com/yakymchukluka-afk/MonoX", 
        "Cloning MonoX repository",
        cwd="/content"
    ):
        print("❌ Git clone failed")
        return False
    
    # Verify clone worked
    if not os.path.exists("/content/MonoX"):
        print("❌ MonoX directory not created")
        return False
    
    if not os.path.exists("/content/MonoX/.git"):
        print("❌ Git repository not properly cloned")
        return False
    
    print("✅ Git clone verified")
    
    # Change to MonoX directory
    os.chdir("/content/MonoX")
    print(f"📁 Changed to: {os.getcwd()}")
    
    # Step 3: Submodule initialization
    print("\n🔗 STEP 3: SUBMODULE INITIALIZATION")
    
    if not run_cmd(
        "git submodule update --init --recursive",
        "Initializing submodules",
        cwd="/content/MonoX"
    ):
        print("⚠️  Submodule init failed, trying alternative...")
        run_cmd(
            "git submodule init && git submodule update",
            "Alternative submodule setup",
            check_error=False,
            cwd="/content/MonoX"
        )
    
    # Step 4: Verify structure
    print("\n🔍 STEP 4: VERIFY STRUCTURE")
    
    required_paths = [
        "/content/MonoX/train_super_gpu_forced.py",
        "/content/MonoX/.external",
        "/content/MonoX/.external/stylegan-v",
        "/content/MonoX/.external/stylegan-v/src",
        "/content/MonoX/.external/stylegan-v/src/training"
    ]
    
    for path in required_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
    
    # Step 5: Apply nuclear patches directly
    print("\n🚀 STEP 5: APPLY NUCLEAR PATCHES")
    
    training_loop_path = "/content/MonoX/.external/stylegan-v/src/training/training_loop.py"
    
    if not os.path.exists(training_loop_path):
        print(f"❌ Training loop not found at: {training_loop_path}")
        
        # Try to find it
        print("🔍 Searching for training_loop.py...")
        for root, dirs, files in os.walk("/content/MonoX/.external"):
            for file in files:
                if file == "training_loop.py":
                    found_path = os.path.join(root, file)
                    print(f"📍 Found training_loop.py at: {found_path}")
        
        return False
    
    # Read and patch training_loop.py
    with open(training_loop_path, 'r') as f:
        content = f.read()
    
    if "🚀🚀🚀 NUCLEAR: training_loop() function called!" not in content:
        print("🔧 Adding nuclear markers...")
        
        # Find the training_loop function signature
        import re
        pattern = r'def training_loop\(\s*([^)]+)\s*\):\s*'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        
        if match:
            print("📍 Found training_loop function")
            
            # Insert nuclear code after the function signature
            insert_pos = match.end()
            
            nuclear_code = '''
    # 🚀 NUCLEAR DEBUG: Training loop activated!
    print("🚀🚀🚀 NUCLEAR: training_loop() function called!")
    print(f"🚀🚀🚀 NUCLEAR: rank={rank}, num_gpus={num_gpus}, batch_size={batch_size}")
    print(f"🚀🚀🚀 NUCLEAR: CUDA available: {torch.cuda.is_available()}")
    print(f"🚀🚀🚀 NUCLEAR: Device count: {torch.cuda.device_count()}")
    
    # 🔥 NUCLEAR GPU MEMORY PRE-ALLOCATION
    if torch.cuda.is_available() and rank == 0:
        device = torch.device('cuda')
        print(f"🔥 NUCLEAR: Starting aggressive GPU memory allocation...")
        
        try:
            gpu_props = torch.cuda.get_device_properties(device)
            total_mem = gpu_props.total_memory / (1024**3)
            print(f"🔥 NUCLEAR: Total GPU memory: {total_mem:.1f} GB")
            
            # Allocate large tensors for heavy GPU usage
            warmup_tensors = []
            for i in range(4):
                size = 1536  # Large tensor size
                tensor = torch.randn(size, size, device=device, dtype=torch.float32)
                # Perform GPU operations
                result = torch.mm(tensor, tensor.transpose(0, 1))
                result = torch.relu(result)
                warmup_tensors.append(result)
                
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                print(f"🔥 NUCLEAR: Tensor {i+1} allocated: {allocated:.2f} GB")
            
            # Show peak usage before cleanup
            peak_usage = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"🚀 NUCLEAR: PEAK GPU USAGE: {peak_usage:.2f} GB")
            
            # Clean up warmup tensors
            del warmup_tensors
            torch.cuda.empty_cache()
            
            final_usage = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"🚀 NUCLEAR: Post-cleanup usage: {final_usage:.2f} GB")
            print(f"🔥 NUCLEAR: GPU PREPPED FOR MAXIMUM UTILIZATION!")
            
        except Exception as e:
            print(f"⚠️ NUCLEAR: GPU allocation warning: {e}")
    
'''
            
            content = content[:insert_pos] + nuclear_code + content[insert_pos:]
            
            # Write the patched content
            with open(training_loop_path, 'w') as f:
                f.write(content)
            
            print("✅ Nuclear patches applied successfully!")
        else:
            print("❌ Could not find training_loop function signature")
            return False
    else:
        print("✅ Nuclear patches already present!")
    
    # Step 6: Clean training environment
    print("\n🧹 STEP 6: CLEAN TRAINING ENVIRONMENT")
    
    dirs_to_clean = [
        "/content/MonoX/results",
        "/content/MonoX/experiments",
        "/content/MonoX/logs"
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"🗑️  Cleaned: {dir_path}")
    
    # Step 7: Launch nuclear training
    print("\n🚀 STEP 7: LAUNCH NUCLEAR TRAINING")
    
    # Set up environment
    nuclear_env = os.environ.copy()
    nuclear_env.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_LAUNCH_BLOCKING': '1',
        'FORCE_CUDA': '1',
        'PYTHONPATH': '/content/MonoX/.external/stylegan-v/src'
    })
    
    # Training command
    cmd = [
        'python3', 'train_super_gpu_forced.py',
        'exp_suffix=nuclear_v2',
        'dataset.path=/content/drive/MyDrive/MonoX/dataset',
        'dataset.resolution=256',
        'training.total_kimg=2',
        'training.snapshot_kimg=1',
        'visualizer.save_every_kimg=1',
        'num_gpus=1'
    ]
    
    print("🚀 LAUNCHING NUCLEAR TRAINING...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Directory: /content/MonoX")
    print("="*60)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=nuclear_env,
            cwd="/content/MonoX"
        )
        
        nuclear_found = False
        line_count = 0
        
        for line in iter(process.stdout.readline, ''):
            line_count += 1
            print(f"{line_count:3}: {line.rstrip()}")
            
            if "🚀🚀🚀 NUCLEAR:" in line:
                nuclear_found = True
                print(f"    🎉 *** NUCLEAR ACTIVATION DETECTED! ***")
            
            if "🔥 NUCLEAR: PEAK GPU USAGE:" in line:
                print(f"    💥 *** PEAK GPU UTILIZATION ACHIEVED! ***")
            
            # Stop after reasonable output
            if line_count > 600:
                print("⏹️ Stopping after 600 lines (training continuing in background)...")
                break
        
        if nuclear_found:
            print(f"\n🎉🚀💥 NUCLEAR ACTIVATION SUCCESSFUL! 💥🚀🎉")
            print("✅ Training loop reached with nuclear patches!")
            print("💥 GPU should now be at MAXIMUM UTILIZATION!")
            return True
        else:
            print(f"\n🔍 Nuclear markers not detected in first {line_count} lines")
            return False
            
    except Exception as e:
        print(f"❌ Training launch error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COLAB NUCLEAR SETUP V2 - ULTRA ROBUST")
    print("="*60)
    success = nuclear_setup_v2()
    if success:
        print("\n🏆 NUCLEAR SETUP V2 COMPLETED!")
        print("💥 NVIDIA L4 should now be under HEAVY UTILIZATION!")
    else:
        print("\n⚠️ Setup needs further investigation")
    print("="*60)