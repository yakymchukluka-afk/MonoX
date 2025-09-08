#!/usr/bin/env python3
"""
Force fresh training test by cleaning experiment directory
"""
import shutil
import os
import subprocess
import sys

def fresh_training_test():
    """Clean experiment directory and run fresh training"""
    
    exp_dir = "/content/MonoX/experiments/ffs_stylegan-v_random_unknown"
    
    print("🧹 FORCING FRESH TRAINING TEST...")
    
    # Remove old experiment directory if it exists
    if os.path.exists(exp_dir):
        print(f"🗑️  Removing old experiment directory: {exp_dir}")
        shutil.rmtree(exp_dir)
        print("✅ Old directory removed")
    else:
        print("📁 No old directory to remove")
    
    # Now run our training script to force fresh file copying
    print("🚀 Running fresh training with latest fixes...")
    
    cmd = [
        sys.executable, "train_super_gpu_forced.py",
        "exp_suffix=fresh_nuclear",
        "dataset.path=/content/drive/MyDrive/MonoX/dataset",
        "dataset.resolution=256", 
        "training.total_kimg=2",  # Very short for testing
        "training.snapshot_kimg=1",
        "visualizer.save_every_kimg=1",
        "num_gpus=1"
    ]
    
    print(f"🎯 Command: {' '.join(cmd)}")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd="/content/MonoX"
        )
        
        print("\n📡 FRESH TRAINING OUTPUT:")
        print("-" * 50)
        
        nuclear_found = False
        line_count = 0
        training_started = False
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line_count += 1
                print(f"{line_count:3d}: {output.strip()}")
                
                # Check for fresh file copying
                if "copying" in output:
                    print("    📂 *** FRESH FILE COPYING! ***")
                    
                # Check for training command execution  
                if "TRAINING COMMAND START" in output:
                    training_started = True
                    print("    🚀 *** TRAINING COMMAND STARTING! ***")
                
                # Check for our nuclear markers
                if "🚀🚀🚀 NUCLEAR:" in output:
                    nuclear_found = True
                    print("    🎉 *** NUCLEAR DEBUG MARKER FOUND! ***")
                elif "🔥 MONOX:" in output:
                    nuclear_found = True  
                    print("    🔥 *** MONOX GPU MARKER FOUND! ***")
                elif "Loading training set" in output:
                    print("    📊 *** TRAINING SET LOADING! ***")
                elif "Constructing networks" in output:
                    print("    🏗️ *** NETWORKS CONSTRUCTING! ***")
                elif "training.data" in output and "error" in output.lower():
                    print("    ❌ *** STILL SEEING DATA ERROR! ***")
        
        return_code = process.poll()
        print(f"\n⏱️  Process finished with return code: {return_code}")
        
        if nuclear_found:
            print("🎉 SUCCESS: Nuclear markers found!")
            return True
        elif training_started:
            print("🔍 Training started but no nuclear markers yet")
            return False
        else:
            print("❌ Training didn't start properly")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fresh_training_test()
    if success:
        print("\n🚀 NUCLEAR GPU CODE IS WORKING!")
        print("🔥 Check nvidia-smi for heavy GPU usage!")
    else:
        print("\n🔍 Need to check what's still preventing nuclear activation")