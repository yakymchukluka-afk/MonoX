#!/usr/bin/env python3
"""
MonoX Training Launcher for HF Space
====================================

Simple launcher designed to run in lukua/monox HF Space.
Will automatically use your authentication to access lukua/monox-dataset.

Usage in your HF Space:
    python3 launch_training_in_space.py

Features:
- Automatic authentication (uses your HF Space credentials)
- 1024x1024 resolution training
- Connects to lukua/monox-dataset 
- Syncs outputs to lukua/monox-model
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_space_environment():
    """Check if we're running in the correct HF Space environment."""
    print("🏠 Checking HF Space Environment...")
    
    # Check if this looks like an HF Space
    space_indicators = [
        Path("/home/user").exists(),  # HF Spaces user directory
        os.environ.get("SPACE_ID"),   # HF Space ID
        os.environ.get("SPACE_AUTHOR_NAME"),  # Space author
    ]
    
    if any(space_indicators):
        print("✅ Running in HF Space environment")
        return True
    else:
        print("⚠️  Not in HF Space - authentication may be required")
        return True  # Allow anyway

def test_dataset_access():
    """Test access to lukua/monox-dataset in HF Space."""
    print("\n🔒 Testing Dataset Access...")
    
    try:
        from datasets import load_dataset
        
        # Test loading a small sample
        print("🔗 Connecting to lukua/monox-dataset...")
        dataset = load_dataset("lukua/monox-dataset", split="train[:1]")
        
        print("✅ Dataset access successful!")
        print(f"📊 Sample loaded: {len(dataset)} items")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📝 Sample keys: {list(sample.keys())}")
            
            # Find image field
            image_field = None
            for key in ['image', 'img', 'picture', 'photo']:
                if key in sample:
                    image_field = key
                    break
            
            if image_field:
                image = sample[image_field]
                print(f"🖼️  Image field: {image_field}")
                print(f"📐 Image size: {image.size if hasattr(image, 'size') else 'unknown'}")
                print("✅ Dataset structure validated!")
                return True
            else:
                print(f"❌ No image field found in: {list(sample.keys())}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        
        if "authentication" in str(e).lower() or "401" in str(e):
            print("🔑 Authentication issue detected")
            print("💡 This should work automatically in your HF Space")
            print("🔧 If this fails in your Space, check Space permissions")
        
        return False

def launch_stylegan_training():
    """Launch StyleGAN-V training with MonoX configuration."""
    print("\n🚀 Launching StyleGAN-V Training...")
    
    try:
        # Ensure directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True) 
        os.makedirs("samples", exist_ok=True)
        
        # Build training command
        cmd = [
            sys.executable,
            "src/infra/launch.py",
            "-cn", "monox_1024_strict",
            f"exp_suffix=monox_training_{int(time.time())}"
        ]
        
        print(f"📋 Training command:")
        print(f"   {' '.join(cmd)}")
        
        # Set environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        print(f"\n🎬 Starting training...")
        print("=" * 50)
        
        # Launch training
        result = subprocess.run(cmd, env=env, cwd=str(Path(__file__).parent))
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Training launch failed: {e}")
        return False

def main():
    """Main launcher function."""
    print("🎨 MonoX StyleGAN-V Training Launcher")
    print("=" * 50)
    print("🏠 Environment: HF Space (lukua/monox)")
    print("🔒 Dataset: lukua/monox-dataset (private)")
    print("🎯 Resolution: 1024x1024 pixels")
    print("=" * 50)
    
    try:
        # Step 1: Check environment
        check_space_environment()
        
        # Step 2: Test dataset access
        if test_dataset_access():
            print("\n✅ Dataset access confirmed!")
            
            # Step 3: Launch training
            if launch_stylegan_training():
                print("\n🎉 MonoX training launched successfully!")
                print("📊 Monitor progress in the Space interface")
            else:
                print("\n❌ Training launch failed")
                sys.exit(1)
        else:
            print("\n❌ Dataset access failed")
            print("🔧 Ensure you're running this in lukua/monox HF Space")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Launch interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()