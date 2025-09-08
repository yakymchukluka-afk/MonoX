#!/usr/bin/env python3
"""
Wait for training to reach our nuclear code and show real-time output
"""
import subprocess
import sys
import time
import os

def monitor_training():
    """Monitor training with real-time output"""
    
    print("🚀 MONITORING TRAINING WITH PATIENCE...")
    print("🔍 Looking for nuclear debug markers...")
    
    # Run training with real-time output monitoring
    cmd = [
        sys.executable, "train_super_gpu_forced.py",
        "exp_suffix=patient_test",
        "dataset.path=/content/drive/MyDrive/MonoX/dataset",  
        "dataset.resolution=256",
        "training.total_kimg=5",  # Short but enough to start training
        "training.snapshot_kimg=2",
        "visualizer.save_every_kimg=1",
        "num_gpus=1"
    ]
    
    try:
        # Start process with real-time output
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True, 
                                 bufsize=1,
                                 universal_newlines=True,
                                 cwd="/content/MonoX")
        
        nuclear_markers_found = []
        start_time = time.time()
        
        print("\n📡 REAL-TIME OUTPUT:")
        print("-" * 50)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
                # Check for nuclear markers
                if "🚀🚀🚀 NUCLEAR:" in output:
                    nuclear_markers_found.append("NUCLEAR DEBUG FOUND!")
                    print("🎉 *** NUCLEAR DEBUG MARKER DETECTED! ***")
                    
                if "🔥 MONOX:" in output:
                    nuclear_markers_found.append("MONOX GPU FOUND!")
                    print("🔥 *** MONOX GPU MARKER DETECTED! ***")
                    
                if "🚀 MONOX: Pre-allocating GPU memory" in output:
                    nuclear_markers_found.append("GPU PRE-ALLOCATION FOUND!")
                    print("💥 *** GPU PRE-ALLOCATION STARTED! ***")
                    
                if "Loading training set" in output:
                    nuclear_markers_found.append("TRAINING SET LOADING!")
                    print("📊 *** TRAINING SET LOADING! ***")
                    
                if "Constructing networks" in output:
                    nuclear_markers_found.append("NETWORKS CONSTRUCTING!")
                    print("🏗️ *** NETWORKS BEING CONSTRUCTED! ***")
            
            # Stop after 5 minutes max
            if time.time() - start_time > 300:
                print("\n⏰ 5 minute timeout reached")
                process.terminate()
                break
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Total time: {elapsed:.1f} seconds")
        
        if nuclear_markers_found:
            print(f"\n🎉 SUCCESS! Found {len(nuclear_markers_found)} nuclear markers:")
            for marker in nuclear_markers_found:
                print(f"   ✅ {marker}")
            return True
        else:
            print(f"\n❌ No nuclear markers found in {elapsed:.1f} seconds")
            return False
            
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return False

if __name__ == "__main__":
    success = monitor_training()
    if success:
        print("\n🚀 NUCLEAR GPU CODE IS ACTIVATING!")
        print("🔥 Training should show heavy GPU usage")
    else:
        print("\n🔍 Need to investigate training startup process")