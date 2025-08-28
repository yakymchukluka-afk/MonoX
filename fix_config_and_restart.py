#!/usr/bin/env python3
"""
Fix Configuration and Restart Training
=====================================

This fixes the missing 'env' section in the config and restarts training.
"""

import os
import sys
from pathlib import Path

def fix_config():
    """Fix the configuration file with the missing env section"""
    print("🔧 Fixing configuration...")
    
    # Change to MonoX directory
    os.chdir("/content/MonoX")
    
    # Create the fixed config
    fixed_config = """# Fixed MonoX + StyleGAN-V Configuration
defaults:
  - dataset: base
  - training: base
  - visualizer: base
  - _self_

# Environment configuration (required by StyleGAN-V)
env:
  project_path: ${hydra:runtime.cwd}
  experiment_name: monox_stylegan_v
  run_name: ${now:%Y-%m-%d_%H-%M-%S}

# Experiment configuration
exp_suffix: "monox"
num_gpus: 1

# Dataset configuration
dataset:
  path: /content/MonoX/dataset
  resolution: 1024
  c_dim: 0

# Training configuration
training:
  total_kimg: 3000
  snapshot_kimg: 250
  batch_size: 4
  fp16: true
  num_gpus: 1
  log_dir: /content/MonoX/results/logs
  preview_dir: /content/MonoX/results/previews
  checkpoint_dir: /content/MonoX/results/checkpoints
  resume: ""

# Sampling configuration
sampling:
  truncation_psi: 1.0
  num_samples: 16

# Visualization configuration
visualizer:
  save_every_kimg: 50
  output_dir: /content/MonoX/results/previews
  grid_size: 4

# Hydra configuration
hydra:
  run:
    dir: /content/MonoX/results/logs
  job:
    chdir: false
  output_subdir: null  # Disable .hydra folder creation
"""
    
    # Write the fixed config
    with open("configs/config.yaml", "w") as f:
        f.write(fixed_config)
    
    print("✅ Config fixed with env section")
    
    # Also check if we need to update the dataset config
    dataset_config = """# Dataset configuration
path: /content/MonoX/dataset
resolution: 1024
c_dim: 0
num_channels: 3
"""
    
    with open("configs/dataset/base.yaml", "w") as f:
        f.write(dataset_config)
    
    # Update training config
    training_config = """# Training configuration
total_kimg: 3000
snapshot_kimg: 250
batch_size: 4
fp16: true
num_gpus: 1
learning_rate: 0.002
beta1: 0.0
beta2: 0.99
log_dir: /content/MonoX/results/logs
preview_dir: /content/MonoX/results/previews
checkpoint_dir: /content/MonoX/results/checkpoints
resume: ""
"""
    
    with open("configs/training/base.yaml", "w") as f:
        f.write(training_config)
    
    # Update visualizer config
    visualizer_config = """# Visualizer configuration
save_every_kimg: 50
output_dir: /content/MonoX/results/previews
grid_size: 4
num_samples: 16
"""
    
    with open("configs/visualizer/base.yaml", "w") as f:
        f.write(visualizer_config)
    
    print("✅ All configs updated")

def create_improved_training_script():
    """Create an improved training script with better error handling"""
    print("🚀 Creating improved training script...")
    
    training_script = '''#!/usr/bin/env python3
"""
Improved MonoX Training Launcher
==============================
"""

import os
import sys
import subprocess
import time

def setup_environment():
    """Setup environment variables"""
    print("🌍 Setting up environment...")
    
    env_vars = {
        "DATASET_DIR": "/content/MonoX/dataset",
        "LOGS_DIR": "/content/MonoX/results/logs",
        "PREVIEWS_DIR": "/content/MonoX/results/previews",
        "CKPT_DIR": "/content/MonoX/results/checkpoints",
        "PYTHONPATH": "/content/MonoX/.external/stylegan-v:/content/MonoX",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1"  # Show full error traces
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

def check_gpu():
    """Check GPU status"""
    print("\\n🖥️ Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {device_name}")
            print(f"✅ Memory: {memory_total:.1f} GB")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def verify_stylegan_v():
    """Verify StyleGAN-V can be imported"""
    print("\\n🎨 Verifying StyleGAN-V...")
    
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    if stylegan_dir not in sys.path:
        sys.path.insert(0, stylegan_dir)
    
    try:
        import src
        print("✅ src module importable")
        
        import src.infra.launch
        print("✅ src.infra.launch importable")
        
        return True
    except Exception as e:
        print(f"❌ StyleGAN-V import failed: {e}")
        return False

def test_config():
    """Test if config can be loaded"""
    print("\\n⚙️ Testing configuration...")
    
    try:
        from omegaconf import OmegaConf
        
        config_path = "/content/MonoX/configs/config.yaml"
        cfg = OmegaConf.load(config_path)
        
        print("✅ Config loads successfully")
        print(f"✅ Dataset path: {cfg.dataset.path}")
        print(f"✅ Total kimg: {cfg.training.total_kimg}")
        print(f"✅ Env section: {cfg.env}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def launch_training():
    """Launch training with comprehensive checks"""
    print("🚀 MonoX Training Launcher v2")
    print("=" * 50)
    
    # Pre-flight checks
    setup_environment()
    
    if not check_gpu():
        print("⚠️ GPU issues detected, but continuing...")
    
    if not verify_stylegan_v():
        print("❌ StyleGAN-V verification failed!")
        return False
    
    if not test_config():
        print("❌ Configuration test failed!")
        return False
    
    print("\\n🎯 All checks passed! Starting training...")
    print("=" * 50)
    
    # Launch training
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        "--config-path", "/content/MonoX/configs",
        "--config-name", "config"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working dir: {stylegan_dir}")
    print("=" * 50)
    
    try:
        # Change to StyleGAN-V directory
        os.chdir(stylegan_dir)
        
        # Launch process
        process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print("📊 Training output:")
        print("-" * 50)
        
        # Stream output
        for line in process.stdout:
            print(line.rstrip())
            
            # Look for success indicators
            if "Loading training set" in line:
                print("🎯 Dataset loading started!")
            elif "Starting training" in line:
                print("🔥 Training loop started!")
            elif "tick" in line.lower() and "kimg" in line.lower():
                print("💪 Training progress detected!")
            
        return_code = process.wait()
        
        if return_code == 0:
            print("\\n🎉 Training completed successfully!")
        else:
            print(f"\\n❌ Training failed with code: {return_code}")
        
        return return_code == 0
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\\n💥 Training failed: {e}")
        return False

if __name__ == "__main__":
    success = launch_training()
    
    if success:
        print("\\n📁 Check results:")
        print("   Logs: /content/MonoX/results/logs/")
        print("   Previews: /content/MonoX/results/previews/")
        print("   Checkpoints: /content/MonoX/results/checkpoints/")
    else:
        print("\\n🔧 Try debugging with:")
        print("   !ls -la /content/MonoX/.external/stylegan-v/")
        print("   !python -c 'import sys; sys.path.insert(0, \\\"/content/MonoX/.external/stylegan-v\\\"); import src.infra.launch'")
'''
    
    with open("/content/MonoX/start_training_v2.py", "w") as f:
        f.write(training_script)
    
    print("✅ Improved training script created")

def main():
    """Main function"""
    print("🔧 Fixing MonoX Configuration Issues")
    print("=" * 50)
    
    # Fix the config
    fix_config()
    
    # Create improved training script
    create_improved_training_script()
    
    print("\n🎉 Configuration fixed!")
    print("=" * 50)
    print("✅ Added missing 'env' section to config")
    print("✅ Updated all config files")
    print("✅ Created improved training script")
    
    print("\n🚀 To restart training with fixes:")
    print("!python /content/MonoX/start_training_v2.py")

if __name__ == "__main__":
    main()