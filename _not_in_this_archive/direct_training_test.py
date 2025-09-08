#!/usr/bin/env python3
"""
Run training command directly to see any errors
"""
import subprocess
import os
import sys

def run_training_directly():
    """Run the exact training command to see what happens"""
    
    print("🔬 RUNNING TRAINING COMMAND DIRECTLY...")
    
    # Change to experiment directory
    exp_dir = "/content/MonoX/experiments/ffs_stylegan-v_random_unknown"
    
    if not os.path.exists(exp_dir):
        print(f"❌ Experiment directory does not exist: {exp_dir}")
        return
    
    print(f"📁 Changing to: {exp_dir}")
    os.chdir(exp_dir)
    
    # Set environment
    env = os.environ.copy()
    env["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"
    
    # The exact command from the log
    cmd = [
        "python3", "src/train.py",
        "hydra.run.dir=.",
        "hydra.output_subdir=null", 
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print(f"📁 In directory: {os.getcwd()}")
    print(f"🌍 Environment: TORCH_EXTENSIONS_DIR={env.get('TORCH_EXTENSIONS_DIR')}")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"\n📡 REAL-TIME OUTPUT:")
        print("-" * 50)
        
        nuclear_found = False
        line_count = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line_count += 1
                print(f"{line_count:3d}: {output.strip()}")
                
                # Check for our nuclear markers
                if "🚀🚀🚀 NUCLEAR:" in output:
                    nuclear_found = True
                    print("*** 🎉 NUCLEAR DEBUG MARKER FOUND! ***")
                elif "🔥 MONOX:" in output:
                    nuclear_found = True  
                    print("*** 🔥 MONOX GPU MARKER FOUND! ***")
                elif "Loading training set" in output:
                    print("*** 📊 TRAINING SET LOADING! ***")
                elif "Constructing networks" in output:
                    print("*** 🏗️ NETWORKS CONSTRUCTING! ***")
        
        return_code = process.poll()
        print(f"\n⏱️  Process finished with return code: {return_code}")
        
        if nuclear_found:
            print("🎉 SUCCESS: Nuclear markers found!")
        else:
            print("❌ No nuclear markers found")
            
        return nuclear_found
        
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

if __name__ == "__main__":
    success = run_training_directly()
    if success:
        print("\n🚀 NUCLEAR GPU CODE IS WORKING!")
    else:
        print("\n🔍 Nuclear code not reached - check output above")