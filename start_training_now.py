#!/usr/bin/env python3
"""
Direct training starter for HF Space
This script will be executed to start training immediately
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import torch

def setup_environment():
    """Setup training environment."""
    print("🔧 Setting up training environment...")
    
    # Create necessary directories
    Path("previews").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    print("✅ Directories created")

def check_gpu():
    """Check GPU availability."""
    print("🔍 Checking GPU availability...")
    
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"CUDA Available: {cuda_available}")
    print(f"GPU Count: {device_count}")
    
    if cuda_available:
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
    
    return cuda_available

def start_training():
    """Start the best available training script."""
    print("🚀 Starting training...")
    
    # Check available training scripts in order of preference
    training_scripts = [
        "gpu_gan_training.py",
        "train_gpu_forced.py", 
        "simple_gan_training.py",
        "train.py"
    ]
    
    for script in training_scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"📝 Found training script: {script}")
            
            try:
                # Start training in background
                print(f"🎯 Executing: python3 {script}")
                process = subprocess.Popen([
                    'python3', script
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                print(f"✅ Training started with PID: {process.pid}")
                
                # Monitor for a few seconds to catch immediate errors
                time.sleep(5)
                
                if process.poll() is None:
                    print("🟢 Training is running successfully!")
                    return True
                else:
                    stdout, stderr = process.communicate()
                    print(f"❌ Training failed immediately:")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    
            except Exception as e:
                print(f"❌ Failed to start {script}: {e}")
                continue
    
    print("❌ No suitable training script found or all failed")
    return False

def monitor_initial_progress():
    """Monitor initial training progress."""
    print("📊 Monitoring initial progress...")
    
    for i in range(6):  # Check for 3 minutes
        preview_dir = Path("previews")
        checkpoint_dir = Path("checkpoints")
        
        previews = len(list(preview_dir.glob("*.png"))) if preview_dir.exists() else 0
        checkpoints = len(list(checkpoint_dir.glob("*.pth"))) if checkpoint_dir.exists() else 0
        
        print(f"   Check {i+1}: Previews: {previews}, Checkpoints: {checkpoints}")
        
        if previews > 0 or checkpoints > 0:
            print("🎉 Training is producing results!")
            break
            
        time.sleep(30)  # Wait 30 seconds

def main():
    """Main execution."""
    print("🎨 MonoX Training Launcher")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Start training
    if start_training():
        print("\n✅ Training launched successfully!")
        monitor_initial_progress()
        print("\n🎯 Training is now running in background")
        print("📱 Check the FastAPI endpoints for status updates")
        print("🌐 Visit your HF Space to see the progress")
    else:
        print("\n❌ Failed to start training")
        print("🔍 Check the available training scripts and dependencies")

if __name__ == "__main__":
    main()