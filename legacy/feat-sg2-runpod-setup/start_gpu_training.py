#!/usr/bin/env python3
"""
MonoX GPU Training Starter
Optimized for HF Spaces GPU compute
"""

import torch
import os
import sys
from pathlib import Path

def check_gpu_setup():
    """Check GPU availability and setup."""
    print("🔍 GPU Setup Analysis:")
    print("=" * 30)
    
    # Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1e9:.1f} GB")
            print(f"  Compute: {gpu_props.major}.{gpu_props.minor}")
        
        # Test GPU
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✅ GPU Test: PASSED")
            return True
        except Exception as e:
            print(f"❌ GPU Test: FAILED - {e}")
            return False
    else:
        print("📱 Running on CPU")
        return False

def estimate_training_time(has_gpu=False, gpu_type="T4"):
    """Estimate training time based on hardware."""
    total_epochs = 50
    
    if has_gpu:
        if gpu_type == "T4":
            time_per_epoch = 0.5  # 30 seconds
            cost_per_hour = 0.60
        elif gpu_type == "A10G":
            time_per_epoch = 0.25  # 15 seconds  
            cost_per_hour = 1.50
        elif gpu_type == "A100":
            time_per_epoch = 0.1   # 6 seconds
            cost_per_hour = 3.00
        else:
            time_per_epoch = 0.5
            cost_per_hour = 0.60
    else:
        time_per_epoch = 15  # 15 minutes on CPU
        cost_per_hour = 0
    
    total_time_hours = (total_epochs * time_per_epoch) / 60
    total_cost = total_time_hours * cost_per_hour
    
    print(f"\n⏱️  Training Time Estimates:")
    print(f"   Hardware: {'GPU ' + gpu_type if has_gpu else 'CPU'}")
    print(f"   Per Epoch: {time_per_epoch * 60:.0f} seconds")
    print(f"   Total Time: {total_time_hours:.1f} hours")
    print(f"   Total Cost: ${total_cost:.2f}")
    
    return total_time_hours, total_cost

def main():
    """Main function."""
    print("🚀 MonoX GPU Training Setup")
    print("=" * 40)
    
    # Check current progress
    preview_dir = Path('previews')
    checkpoint_dir = Path('checkpoints')
    
    if preview_dir.exists():
        samples = list(preview_dir.glob('samples_epoch_*.png'))
        print(f"📊 Current Progress: {len(samples)} epochs completed")
    else:
        print("📊 Current Progress: No samples found")
    
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        print(f"💾 Checkpoints: {len(checkpoints)} saved")
    else:
        print("💾 Checkpoints: None")
    
    # Check GPU
    has_gpu = check_gpu_setup()
    
    # Show estimates for different hardware
    print(f"\n💰 Hardware Comparison:")
    estimate_training_time(False, "CPU")
    estimate_training_time(True, "T4")
    estimate_training_time(True, "A10G") 
    estimate_training_time(True, "A100")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    if has_gpu:
        print("✅ GPU detected - ready for fast training!")
        print("   Run: python3 gpu_gan_training.py")
    else:
        print("💡 To upgrade to GPU:")
        print("   1. Go to your HF Space")
        print("   2. Click 'Settings' → 'Hardware'")
        print("   3. Select GPU tier (T4 recommended)")
        print("   4. Restart Space")
        print("   5. Run: python3 gpu_gan_training.py")
    
    print(f"\n🎨 Your MonoX training will generate stunning results either way!")

if __name__ == "__main__":
    main()