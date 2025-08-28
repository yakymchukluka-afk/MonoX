#!/usr/bin/env python3
"""
Single-File MonoX Colab Setup
============================

This file contains everything needed to set up MonoX + StyleGAN-V in Google Colab.
Just run this one file and it will handle everything.

Usage in Colab:
    !wget https://raw.githubusercontent.com/your-repo/MonoX/main/setup_monox_colab.py
    !python setup_monox_colab.py

Or if you have the repo cloned:
    !cd /content && git clone https://github.com/your-repo/MonoX.git
    !cd /content/MonoX && python setup_monox_colab.py
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

# Configuration
MONOX_ROOT = "/content/MonoX"
STYLEGAN_V_URL = "https://github.com/yakymchukluka-afk/stylegan-v.git"
RESULTS_DIR = "/content/MonoX/results"
DATASET_DIR = "/content/MonoX/dataset"

def ensure_directory():
    """Ensure we're in the right directory and create structure"""
    print("📁 Setting up directory structure...")
    
    # Create MonoX directory if it doesn't exist
    Path(MONOX_ROOT).mkdir(parents=True, exist_ok=True)
    
    # Change to MonoX directory
    os.chdir(MONOX_ROOT)
    print(f"✅ Working directory: {os.getcwd()}")
    
    # Create all necessary subdirectories
    dirs_to_create = [
        ".external",
        "configs",
        "configs/dataset", 
        "configs/training",
        "configs/visualizer",
        "results",
        "results/logs",
        "results/previews",
        "results/checkpoints",
        "dataset",
        "dataset/sample_images",
        "src",
        "src/infra"
    ]
    
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_name}")

def run_command(cmd, check=True, cwd=None):
    """Run command with proper error handling"""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            check=check,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        if check:
            raise
        return e

def check_gpu():
    """Verify GPU is available"""
    print("\n🖥️  Checking GPU...")
    try:
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            print("✅ GPU detected!")
            return True
        else:
            print("❌ No GPU detected")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def install_dependencies():
    """Install all required packages"""
    print("\n📦 Installing dependencies...")
    
    # Update pip
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install PyTorch with CUDA
    run_command([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    
    # Install other packages
    packages = [
        "hydra-core==1.3.2",
        "omegaconf==2.3.0", 
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "ninja>=1.10.0"
    ]
    
    for package in packages:
        run_command([sys.executable, "-m", "pip", "install", package])

def clone_stylegan_v():
    """Clone StyleGAN-V repository"""
    print("\n🎨 Cloning StyleGAN-V...")
    
    stylegan_dir = ".external/stylegan-v"
    
    if os.path.exists(stylegan_dir):
        if os.path.exists(f"{stylegan_dir}/.git"):
            print("StyleGAN-V already cloned")
            return stylegan_dir
        else:
            shutil.rmtree(stylegan_dir)
    
    run_command([
        "git", "clone", "--recursive",
        STYLEGAN_V_URL,
        stylegan_dir
    ])
    
    print(f"✅ StyleGAN-V cloned to {stylegan_dir}")
    return stylegan_dir

def create_config_files():
    """Create necessary configuration files"""
    print("\n⚙️  Creating configuration files...")
    
    # Main config
    config_content = """# Clean Hydra configuration for MonoX StyleGAN-V training
defaults:
  - dataset: base
  - training: base
  - visualizer: base
  - _self_

exp_suffix: "monox"
num_gpus: 1

dataset:
  path: /content/MonoX/dataset
  resolution: 1024
  c_dim: 0

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

sampling:
  truncation_psi: 1.0

visualizer:
  save_every_kimg: 50
  output_dir: /content/MonoX/results/previews
  grid_size: 4

hydra:
  run:
    dir: /content/MonoX/results/logs
  job:
    chdir: false
"""
    
    with open("configs/config.yaml", "w") as f:
        f.write(config_content)
    
    # Dataset config
    dataset_config = """# Base dataset configuration
path: /content/MonoX/dataset
resolution: 1024
c_dim: 0
"""
    
    with open("configs/dataset/base.yaml", "w") as f:
        f.write(dataset_config)
    
    # Training config
    training_config = """# Base training configuration
total_kimg: 3000
snapshot_kimg: 250
batch_size: 4
fp16: true
log_dir: /content/MonoX/results/logs
preview_dir: /content/MonoX/results/previews
checkpoint_dir: /content/MonoX/results/checkpoints
"""
    
    with open("configs/training/base.yaml", "w") as f:
        f.write(training_config)
    
    # Visualizer config
    visualizer_config = """# Base visualizer configuration
save_every_kimg: 50
output_dir: /content/MonoX/results/previews
grid_size: 4
"""
    
    with open("configs/visualizer/base.yaml", "w") as f:
        f.write(visualizer_config)
    
    print("✅ Configuration files created")

def create_sample_dataset():
    """Create sample dataset for testing"""
    print("\n🎨 Creating sample dataset...")
    
    try:
        from PIL import Image
        import numpy as np
        
        for i in range(10):
            # Create simple test images
            img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f"dataset/sample_images/sample_{i:03d}.png")
        
        print("✅ Created 10 sample images")
    except Exception as e:
        print(f"⚠️  Could not create sample dataset: {e}")

def create_launch_script():
    """Create the training launch script"""
    print("\n🚀 Creating launch script...")
    
    launch_script = '''#!/usr/bin/env python3
"""Training Launcher for MonoX + StyleGAN-V"""

import os
import sys
import subprocess
from pathlib import Path

# Setup environment
MONOX_ROOT = "/content/MonoX"
STYLEGAN_V_DIR = "/content/MonoX/.external/stylegan-v"

def setup_environment():
    """Setup environment variables and paths"""
    os.environ.update({
        "DATASET_DIR": "/content/MonoX/dataset",
        "LOGS_DIR": "/content/MonoX/results/logs", 
        "PREVIEWS_DIR": "/content/MonoX/results/previews",
        "CKPT_DIR": "/content/MonoX/results/checkpoints",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1"
    })
    
    # Add StyleGAN-V to Python path
    if STYLEGAN_V_DIR not in sys.path:
        sys.path.insert(0, STYLEGAN_V_DIR)
    
    pythonpath = f"{STYLEGAN_V_DIR}:{MONOX_ROOT}"
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        pythonpath += f":{existing}"
    os.environ["PYTHONPATH"] = pythonpath

def launch_training():
    """Launch training process"""
    setup_environment()
    
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        "--config-path", f"{MONOX_ROOT}/configs",
        "--config-name", "config"
    ]
    
    print(f"Launching: {' '.join(cmd)}")
    print(f"Working directory: {STYLEGAN_V_DIR}")
    
    env = os.environ.copy()
    
    try:
        process = subprocess.Popen(
            cmd, cwd=STYLEGAN_V_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        for line in process.stdout:
            print(line.rstrip())
            
        return_code = process.wait()
        print(f"Training finished with code: {return_code}")
        
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    launch_training()
'''
    
    with open("launch_training.py", "w") as f:
        f.write(launch_script)
    
    print("✅ Launch script created")

def setup_environment_vars():
    """Set up environment variables"""
    print("\n🌍 Setting up environment variables...")
    
    env_vars = {
        "MONOX_ROOT": MONOX_ROOT,
        "DATASET_DIR": DATASET_DIR,
        "LOGS_DIR": f"{RESULTS_DIR}/logs",
        "PREVIEWS_DIR": f"{RESULTS_DIR}/previews",
        "CKPT_DIR": f"{RESULTS_DIR}/checkpoints",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

def verify_setup():
    """Verify the setup is working"""
    print("\n🔍 Verifying setup...")
    
    checks = []
    
    # Check directories
    required_dirs = [
        MONOX_ROOT,
        f"{MONOX_ROOT}/.external/stylegan-v",
        f"{RESULTS_DIR}/logs",
        "configs"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            checks.append(f"✅ {dir_path}")
        else:
            checks.append(f"❌ {dir_path} (missing)")
    
    # Check Python packages
    try:
        import torch
        checks.append(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            checks.append(f"✅ CUDA available")
        else:
            checks.append("❌ CUDA not available")
    except ImportError:
        checks.append("❌ PyTorch not installed")
    
    try:
        import hydra
        checks.append(f"✅ Hydra {hydra.__version__}")
    except ImportError:
        checks.append("❌ Hydra not installed")
    
    # Check StyleGAN-V import
    stylegan_path = f"{MONOX_ROOT}/.external/stylegan-v"
    if stylegan_path not in sys.path:
        sys.path.insert(0, stylegan_path)
    
    try:
        import src.infra.launch
        checks.append("✅ StyleGAN-V src module importable")
    except ImportError as e:
        checks.append(f"❌ StyleGAN-V import failed: {e}")
    
    for check in checks:
        print(check)
    
    return all("✅" in check for check in checks)

def main():
    """Main setup function"""
    print("🚀 MonoX + StyleGAN-V Colab Setup")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. Setup directory structure
        ensure_directory()
        
        # 2. Check GPU
        gpu_ok = check_gpu()
        
        # 3. Install dependencies
        install_dependencies()
        
        # 4. Clone StyleGAN-V
        clone_stylegan_v()
        
        # 5. Create config files
        create_config_files()
        
        # 6. Create sample dataset
        create_sample_dataset()
        
        # 7. Create launch script
        create_launch_script()
        
        # 8. Setup environment
        setup_environment_vars()
        
        # 9. Verify everything
        setup_ok = verify_setup()
        
        elapsed = time.time() - start_time
        
        if setup_ok:
            print(f"\n🎉 Setup completed successfully in {elapsed:.1f}s!")
            print("\nNext steps:")
            print("1. Run training: !python /content/MonoX/launch_training.py")
            print("2. Check GPU: !nvidia-smi")
            print("3. Monitor logs: !tail -f /content/MonoX/results/logs/*.log")
        else:
            print(f"\n⚠️  Setup completed with warnings in {elapsed:.1f}s")
            
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()