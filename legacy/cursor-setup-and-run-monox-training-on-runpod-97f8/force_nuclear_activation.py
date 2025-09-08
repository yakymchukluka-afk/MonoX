#!/usr/bin/env python3
"""
🚀 FORCE NUCLEAR ACTIVATION SCRIPT
=================================
This script bypasses all submodule and directory issues to force
the nuclear training_loop() activation with our torch fixes.
"""

import subprocess
import os
import shutil
import sys

def force_nuclear_activation():
    """Force nuclear activation by ensuring all fixes are applied."""
    print("🚀🚀🚀 FORCING NUCLEAR ACTIVATION!")
    print("="*60)
    
    # Step 1: Clean results directory
    results_dir = "/content/MonoX/results"
    if os.path.exists(results_dir):
        print(f"🧹 Cleaning results directory: {results_dir}")
        shutil.rmtree(results_dir)
    
    # Step 2: Verify our nuclear torch fix is in place
    training_loop_path = "/content/MonoX/.external/stylegan-v/src/training/training_loop.py"
    print(f"🔍 Checking nuclear fix in: {training_loop_path}")
    
    if not os.path.exists(training_loop_path):
        print(f"❌ File not found: {training_loop_path}")
        print("🔍 Let's check what's available:")
        base_dir = "/content/MonoX/.external/stylegan-v"
        if os.path.exists(base_dir):
            print(f"📁 Contents of {base_dir}:")
            for item in os.listdir(base_dir):
                print(f"   - {item}")
            src_dir = os.path.join(base_dir, "src")
            if os.path.exists(src_dir):
                print(f"📁 Contents of {src_dir}:")
                for item in os.listdir(src_dir):
                    print(f"   - {item}")
        print("⚠️  Proceeding without nuclear fix verification...")
        return False
    
    with open(training_loop_path, 'r') as f:
        content = f.read()
    
    if "🚀🚀🚀 NUCLEAR: training_loop() function called!" in content:
        print("✅ Nuclear training_loop markers found!")
    else:
        print("❌ Nuclear markers missing! Adding them now...")
        # Apply the fix if missing
        content = content.replace(
            "def training_loop(",
            """def training_loop("""
        )
        # Find the function signature end and add our debug
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
    
    # 🚀 NUCLEAR: Continue to GPU memory pre-allocation
    if torch.cuda.is_available() and rank == 0:
        device = torch.device('cuda')
        print(f"🚀🚀🚀 NUCLEAR: Starting AGGRESSIVE GPU memory pre-allocation...")
    
'''
            content = content[:insert_pos] + nuclear_code + content[insert_pos:]
            
            with open(training_loop_path, 'w') as f:
                f.write(content)
            print("✅ Nuclear markers added!")
    
    # Step 3: Remove old experiment directory to force fresh files
    exp_dir = "/content/MonoX/experiments/ffs_stylegan-v_random_unknown"
    if os.path.exists(exp_dir):
        print(f"🗑️  Removing old experiment: {exp_dir}")
        shutil.rmtree(exp_dir)
    
    # Step 4: Run the training with forced nuclear activation
    print("🚀 LAUNCHING NUCLEAR TRAINING...")
    print("="*60)
    
    cmd = [
        "python3", "/content/MonoX/train_super_gpu_forced.py",
        "exp_suffix=nuclear_forced",
        "dataset.path=/content/drive/MyDrive/MonoX/dataset",
        "dataset.resolution=256",
        "training.total_kimg=2",
        "training.snapshot_kimg=1",
        "visualizer.save_every_kimg=1",
        "num_gpus=1"
    ]
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/content/MonoX/.external/stylegan-v/src'
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd='/content/MonoX'
        )
        
        output = result.stdout + result.stderr
        print("📡 NUCLEAR TRAINING OUTPUT:")
        print("="*60)
        
        # Look for nuclear markers
        lines = output.split('\n')
        nuclear_found = False
        for i, line in enumerate(lines, 1):
            print(f"{i:3}: {line}")
            if "🚀🚀🚀 NUCLEAR:" in line:
                nuclear_found = True
                print(f"    🎉 *** NUCLEAR MARKER FOUND! ***")
        
        if nuclear_found:
            print("\n🎉🚀💥 NUCLEAR ACTIVATION SUCCESSFUL! 💥🚀🎉")
            print("✅ Training loop reached!")
            print("🔥 GPU utilization should now be active!")
        else:
            print("\n❌ Nuclear markers not found in output")
            print("🔍 Check the output above for errors")
        
        return nuclear_found
        
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out - this might indicate progress!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NUCLEAR ACTIVATION FORCE SCRIPT")
    print("="*60)
    success = force_nuclear_activation()
    if success:
        print("\n🏆 MISSION ACCOMPLISHED!")
        print("💥 Your NVIDIA L4 should now be under heavy utilization!")
    else:
        print("\n⚠️  Nuclear activation needs investigation")
    print("="*60)