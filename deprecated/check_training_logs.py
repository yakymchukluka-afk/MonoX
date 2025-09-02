#!/usr/bin/env python3
"""
Check if training is actually running and what logs it's producing
"""
import os
import time
import subprocess

def check_training_status():
    """Check what's happening with training"""
    
    print("🔍 CHECKING TRAINING STATUS...")
    
    # Check if training process is running
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        train_processes = [line for line in result.stdout.split('\n') if 'train.py' in line or 'python' in line and 'stylegan' in line]
        
        if train_processes:
            print(f"✅ Found {len(train_processes)} training-related processes:")
            for proc in train_processes:
                print(f"   {proc}")
        else:
            print("❌ No training processes found")
    except:
        print("⚠️  Could not check processes")
    
    # Check experiment directory
    exp_dir = "/content/MonoX/experiments/ffs_stylegan-v_random_unknown"
    if os.path.exists(exp_dir):
        print(f"\n📁 Experiment directory exists: {exp_dir}")
        
        # Check for log files
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.log'):
                    log_path = os.path.join(root, file)
                    print(f"📝 Found log: {log_path}")
                    
                    # Show last few lines
                    try:
                        with open(log_path, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                print(f"   Last 5 lines of {file}:")
                                for line in lines[-5:]:
                                    print(f"      {line.strip()}")
                    except:
                        print(f"   Could not read {file}")
    else:
        print(f"\n❌ Experiment directory does not exist: {exp_dir}")
    
    # Check main logs directory  
    logs_dir = "/content/MonoX/logs"
    if os.path.exists(logs_dir):
        print(f"\n📁 Logs directory: {logs_dir}")
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
        
        if log_files:
            print(f"📝 Found {len(log_files)} log files:")
            # Get the most recent log
            latest_log = max([os.path.join(logs_dir, f) for f in log_files], key=os.path.getmtime)
            print(f"📊 Most recent log: {latest_log}")
            
            try:
                with open(latest_log, 'r') as f:
                    content = f.read()
                    if content:
                        print(f"📄 Log content (last 1000 chars):")
                        print("-" * 50)
                        print(content[-1000:])
                        print("-" * 50)
                        
                        # Check for our nuclear markers
                        nuclear_markers = [
                            "🚀🚀🚀 NUCLEAR:",
                            "🔥 MONOX:",
                            "Loading training set",
                            "Constructing networks"
                        ]
                        
                        found_markers = []
                        for marker in nuclear_markers:
                            if marker in content:
                                found_markers.append(marker)
                        
                        if found_markers:
                            print(f"\n🎉 FOUND NUCLEAR MARKERS:")
                            for marker in found_markers:
                                print(f"   ✅ {marker}")
                        else:
                            print(f"\n🔍 No nuclear markers found yet")
                            
                    else:
                        print("📄 Log file is empty")
            except Exception as e:
                print(f"❌ Could not read log: {e}")
        else:
            print("📝 No log files found")
    else:
        print(f"\n❌ Logs directory does not exist: {logs_dir}")

if __name__ == "__main__":
    check_training_status()