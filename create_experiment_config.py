#!/usr/bin/env python3
"""
Create proper experiment configuration for StyleGAN-V training.
"""

import os
import yaml
import subprocess
from pathlib import Path

def create_stylegan_experiment_config():
    """Create the experiment_config.yaml required by StyleGAN-V."""
    
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    except ImportError:
        gpu_available = False
        gpu_count = 0
    
    # Create experiment configuration
    config = {
        # Dataset configuration
        "data": "/workspace/dataset",
        "resolution": 1024,
        "cond": False,  # Unconditional generation
        "subset": None,  # Use all images
        
        # Training configuration
        "outdir": "/workspace/training_output",
        "cfg": "auto",  # Auto-configure based on dataset
        "gpus": max(gpu_count, 1),  # Use CPU if no GPU
        "kimg": 100,  # Short training for testing (100k images)
        "snap": 25,   # Save checkpoint every 25 kimg
        "batch_size": 4 if not gpu_available else 16,
        "resume": None,  # Fresh training
        
        # Augmentation
        "aug": "ada",  # Adaptive augmentation
        "mirror": True,  # Enable x-flips
        "augpipe": "bgc",  # Augmentation pipeline
        "target": 0.6,  # ADA target
        
        # Performance
        "fp32": not gpu_available,  # Use FP32 for CPU
        "nhwc": False,
        "nobench": False,
        "allow_tf32": False,
        "num_workers": 2,  # Reduced for stability
        
        # Metrics (disabled for faster training)
        "metrics": [],
        
        # Random seed
        "seed": 42,
        
        # Misc
        "dry_run": False,
        "freezed": 0
    }
    
    return config

def create_configs_for_stylegan():
    """Create all necessary config files for StyleGAN-V training."""
    
    # Create output directories
    output_dirs = [
        "/workspace/training_output",
        "/workspace/logs", 
        "/workspace/checkpoints",
        "/workspace/previews"
    ]
    
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create experiment config for StyleGAN-V directory
    experiment_config = create_stylegan_experiment_config()
    
    # Save to StyleGAN-V directory (where train.py expects it)
    stylegan_config_path = "/workspace/.external/stylegan-v/experiment_config.yaml"
    with open(stylegan_config_path, "w") as f:
        yaml.dump(experiment_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created experiment config: {stylegan_config_path}")
    
    # Also save a copy in our workspace
    workspace_config_path = "/workspace/experiment_config.yaml"
    with open(workspace_config_path, "w") as f:
        yaml.dump(experiment_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created workspace config: {workspace_config_path}")
    
    # Print config summary
    print("\\n📋 Training Configuration:")
    print(f"  Dataset: {experiment_config['data']}")
    print(f"  Resolution: {experiment_config['resolution']}")
    print(f"  Total KImg: {experiment_config['kimg']}")
    print(f"  Checkpoint Interval: {experiment_config['snap']} kimg")
    print(f"  Batch Size: {experiment_config['batch_size']}")
    print(f"  GPUs: {experiment_config['gpus']}")
    print(f"  Mixed Precision: {not experiment_config['fp32']}")
    print(f"  Augmentation: {experiment_config['aug']}")
    
    return experiment_config

def test_stylegan_training():
    """Test StyleGAN-V training with the created config."""
    
    # Create configs
    config = create_configs_for_stylegan()
    
    # Test the training command
    cmd = [
        "python3", "src/train.py",
        "--dry_run", "true"  # Just test the configuration
    ]
    
    print("\\n🧪 Testing StyleGAN-V configuration...")
    
    try:
        os.chdir("/workspace/.external/stylegan-v")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ StyleGAN-V configuration test passed!")
            print("📝 Output preview:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("❌ StyleGAN-V configuration test failed!")
            print("Error output:")
            print(result.stderr)
            return False
    
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False
    finally:
        os.chdir("/workspace")

def main():
    """Main function."""
    print("🔧 Creating StyleGAN-V Experiment Configuration")
    print("=" * 60)
    
    success = test_stylegan_training()
    
    if success:
        print("\\n✅ Configuration created and validated!")
        print("🚀 Ready to start training with:")
        print("   python3 direct_stylegan_training.py")
        return 0
    else:
        print("\\n❌ Configuration validation failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())