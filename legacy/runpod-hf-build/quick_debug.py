#!/usr/bin/env python3
"""
Quick debug: Test if our training_loop is being called at all
"""
import subprocess
import sys
import os

def test_training_debug():
    """Test if our training debug output appears"""
    
    print("🔬 TESTING IF TRAINING_LOOP IS CALLED...")
    
    # Run a very quick training test to see debug output
    cmd = [
        sys.executable, "train_super_gpu_forced.py",
        "exp_suffix=debug_test",
        "dataset.path=/content/drive/MyDrive/MonoX/dataset",  
        "dataset.resolution=256",
        "training.total_kimg=1",  # Very short training
        "training.snapshot_kimg=1",
        "visualizer.save_every_kimg=1",
        "num_gpus=1"
    ]
    
    try:
        # Run for maximum 30 seconds
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=30,
                              cwd="/content/MonoX")
        
        output = result.stdout + result.stderr
        
        print("🔍 LOOKING FOR DEBUG MARKERS...")
        
        # Check for our debug markers
        markers = [
            "🚀🚀🚀 NUCLEAR: training_loop() function called!",
            "🔥 MONOX: Forcing GPU device for rank",
            "🚀 MONOX: Pre-allocating GPU memory",
            "Loading training set...",
            "Constructing networks..."
        ]
        
        found_markers = []
        for marker in markers:
            if marker in output:
                found_markers.append(f"✅ {marker}")
            else:
                found_markers.append(f"❌ {marker}")
        
        print("\n📊 DEBUG MARKER RESULTS:")
        for marker in found_markers:
            print(f"   {marker}")
        
        # Check for any errors
        if "Traceback" in output or "Error" in output:
            print(f"\n❌ ERRORS FOUND:")
            error_lines = [line for line in output.split('\n') if 'Error' in line or 'Traceback' in line]
            for line in error_lines[:5]:  # Show first 5 error lines
                print(f"   {line}")
        
        # Show last 10 lines of output
        print(f"\n📝 LAST 10 LINES OF OUTPUT:")
        last_lines = output.split('\n')[-10:]
        for line in last_lines:
            if line.strip():
                print(f"   {line}")
                
        return "🚀🚀🚀 NUCLEAR: training_loop()" in output
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out - this might mean it's working but slow")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_training_debug()
    if success:
        print("\n🎉 TRAINING_LOOP IS BEING CALLED!")
        print("🚀 Nuclear GPU code should be activating")
    else:
        print("\n❌ TRAINING_LOOP NOT REACHED")
        print("🔍 Need to debug further")