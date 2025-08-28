#!/usr/bin/env python3
"""
Direct Colab Solution - Copy and Paste This Entire Cell
=======================================================

This is a complete, self-contained solution that creates all files and sets up MonoX.
Just copy this entire code block into a single Colab cell and run it.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

def run_cmd(cmd, check=True, cwd=None):
    """Run command safely"""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check, 
                              capture_output=True, text=True, cwd=cwd)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        if check:
            raise
        return e

def create_all_files():
    """Create all necessary files"""
    print("📝 Creating all setup files...")
    
    # Ensure MonoX directory exists
    MONOX_ROOT = "/content/MonoX"
    Path(MONOX_ROOT).mkdir(parents=True, exist_ok=True)
    
    # Create main setup script
    setup_script = f"""#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

MONOX_ROOT = "/content/MonoX"
STYLEGAN_V_URL = "https://github.com/yakymchukluka-afk/stylegan-v.git"

def run_cmd(cmd, check=True, cwd=None):
    print(f"Running: {{' '.join(cmd) if isinstance(cmd, list) else cmd}}")
    try:
        result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check, 
                              capture_output=True, text=True, cwd=cwd)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {{e}}")
        if e.stderr:
            print(f"STDERR: {{e.stderr}}")
        if check:
            raise
        return e

def setup_directories():
    print("📁 Creating directories...")
    Path(MONOX_ROOT).mkdir(parents=True, exist_ok=True)
    os.chdir(MONOX_ROOT)
    
    dirs = [
        ".external", "configs", "configs/dataset", "configs/training", 
        "configs/visualizer", "results", "results/logs", "results/previews",
        "results/checkpoints", "dataset", "dataset/sample_images", "src", "src/infra"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✅ {{d}}")

def install_packages():
    print("\\n📦 Installing packages...")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_cmd([sys.executable, "-m", "pip", "install", "torch", "torchvision", 
             "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    
    packages = [
        "hydra-core==1.3.2", "omegaconf==2.3.0", "numpy>=1.21.0",
        "pillow>=8.0.0", "opencv-python>=4.5.0", "tqdm>=4.62.0",
        "matplotlib>=3.4.0", "ninja>=1.10.0", "psutil>=5.8.0"
    ]
    
    for pkg in packages:
        run_cmd([sys.executable, "-m", "pip", "install", pkg])

def clone_stylegan():
    print("\\n🎨 Cloning StyleGAN-V...")
    stylegan_dir = ".external/stylegan-v"
    
    if os.path.exists(stylegan_dir):
        if os.path.exists(f"{{stylegan_dir}}/.git"):
            print("StyleGAN-V already exists")
            return stylegan_dir
        else:
            shutil.rmtree(stylegan_dir)
    
    run_cmd(["git", "clone", "--recursive", STYLEGAN_V_URL, stylegan_dir])
    print(f"✅ StyleGAN-V cloned")
    return stylegan_dir

def create_configs():
    print("\\n⚙️ Creating configs...")
    
    main_config = '''defaults:
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
'''
    
    with open("configs/config.yaml", "w") as f:
        f.write(main_config)
    
    with open("configs/dataset/base.yaml", "w") as f:
        f.write("path: /content/MonoX/dataset\\nresolution: 1024\\nc_dim: 0\\n")
    
    with open("configs/training/base.yaml", "w") as f:
        f.write("total_kimg: 3000\\nsnapshot_kimg: 250\\nbatch_size: 4\\nfp16: true\\n")
    
    with open("configs/visualizer/base.yaml", "w") as f:
        f.write("save_every_kimg: 50\\ngrid_size: 4\\n")
    
    print("✅ Configs created")

def create_sample_data():
    print("\\n🎨 Creating sample data...")
    try:
        from PIL import Image
        import numpy as np
        
        for i in range(10):
            img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f"dataset/sample_images/sample_{{i:03d}}.png")
        
        print("✅ Sample dataset created")
    except Exception as e:
        print(f"⚠️ Could not create sample data: {{e}}")

def setup_environment():
    print("\\n🌍 Setting environment...")
    env_vars = {{
        "DATASET_DIR": "/content/MonoX/dataset",
        "LOGS_DIR": "/content/MonoX/results/logs",
        "PREVIEWS_DIR": "/content/MonoX/results/previews",
        "CKPT_DIR": "/content/MonoX/results/checkpoints",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/content/MonoX/.external/stylegan-v:/content/MonoX"
    }}
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {{key}}={{value}}")

def verify_setup():
    print("\\n🔍 Verifying setup...")
    checks = []
    
    dirs = [MONOX_ROOT, f"{{MONOX_ROOT}}/.external/stylegan-v", "configs"]
    for d in dirs:
        if os.path.exists(d):
            checks.append(f"✅ {{d}}")
        else:
            checks.append(f"❌ {{d}}")
    
    try:
        import torch
        checks.append(f"✅ PyTorch {{torch.__version__}}")
        if torch.cuda.is_available():
            checks.append("✅ CUDA available")
        else:
            checks.append("❌ CUDA not available")
    except:
        checks.append("❌ PyTorch not available")
    
    try:
        import hydra
        checks.append(f"✅ Hydra {{hydra.__version__}}")
    except:
        checks.append("❌ Hydra not available")
    
    if "/content/MonoX/.external/stylegan-v" not in sys.path:
        sys.path.insert(0, "/content/MonoX/.external/stylegan-v")
    
    try:
        import src.infra.launch
        checks.append("✅ StyleGAN-V importable")
    except Exception as e:
        checks.append(f"❌ StyleGAN-V import failed: {{e}}")
    
    for check in checks:
        print(check)
    
    return all("✅" in check for check in checks)

def main():
    print("🚀 MonoX + StyleGAN-V Setup")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        setup_directories()
        install_packages()
        clone_stylegan()
        create_configs()
        create_sample_data()
        setup_environment()
        success = verify_setup()
        
        elapsed = time.time() - start_time
        
        if success:
            print(f"\\n🎉 Setup completed in {{elapsed:.1f}}s!")
            print("\\nNext: Run training with !python /content/MonoX/launch_training.py")
        else:
            print(f"\\n⚠️ Setup completed with warnings in {{elapsed:.1f}}s")
        
    except Exception as e:
        print(f"\\n💥 Setup failed: {{e}}")
        raise

if __name__ == "__main__":
    main()
"""
    
    with open(f"{MONOX_ROOT}/setup_monox_colab.py", "w") as f:
        f.write(setup_script)
    
    # Create training launcher
    launcher_script = f"""#!/usr/bin/env python3
import os
import sys
import subprocess

def launch_training():
    print("🚀 Launching training...")
    
    os.environ.update({{
        "DATASET_DIR": "/content/MonoX/dataset",
        "LOGS_DIR": "/content/MonoX/results/logs",
        "PREVIEWS_DIR": "/content/MonoX/results/previews",
        "CKPT_DIR": "/content/MonoX/results/checkpoints",
        "PYTHONPATH": "/content/MonoX/.external/stylegan-v:/content/MonoX",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1"
    }})
    
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    if stylegan_dir not in sys.path:
        sys.path.insert(0, stylegan_dir)
    
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        "--config-path", "/content/MonoX/configs",
        "--config-name", "config"
    ]
    
    print(f"Command: {{' '.join(cmd)}}")
    print(f"Working dir: {{stylegan_dir}}")
    
    try:
        process = subprocess.Popen(
            cmd, cwd=stylegan_dir, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        for line in process.stdout:
            print(line.rstrip())
            
        return_code = process.wait()
        print(f"Training finished with code: {{return_code}}")
        
    except Exception as e:
        print(f"Training failed: {{e}}")

if __name__ == "__main__":
    launch_training()
"""
    
    with open(f"{MONOX_ROOT}/launch_training.py", "w") as f:
        f.write(launcher_script)
    
    # Create verification script
    verify_script = f"""#!/usr/bin/env python3
import os
import sys
import subprocess

def verify():
    print("🔍 Verification Checks")
    print("=" * 30)
    
    checks = []
    
    dirs = [
        "/content/MonoX",
        "/content/MonoX/.external/stylegan-v",
        "/content/MonoX/configs",
        "/content/MonoX/results"
    ]
    
    for d in dirs:
        if os.path.exists(d):
            checks.append(f"✅ {{d}}")
        else:
            checks.append(f"❌ {{d}}")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if result.returncode == 0:
            checks.append("✅ GPU available")
        else:
            checks.append("❌ GPU not available")
    except:
        checks.append("❌ GPU check failed")
    
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(f"✅ PyTorch CUDA {{torch.version.cuda}}")
        else:
            checks.append("❌ PyTorch CUDA not available")
    except:
        checks.append("❌ PyTorch not available")
    
    try:
        import hydra
        checks.append(f"✅ Hydra {{hydra.__version__}}")
    except:
        checks.append("❌ Hydra not available")
    
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    if stylegan_dir not in sys.path:
        sys.path.insert(0, stylegan_dir)
    
    try:
        import src.infra.launch
        checks.append("✅ StyleGAN-V importable")
    except Exception as e:
        checks.append(f"❌ StyleGAN-V: {{e}}")
    
    for check in checks:
        print(check)
    
    passed = sum(1 for c in checks if "✅" in c)
    total = len(checks)
    
    print(f"\\nResult: {{passed}}/{{total}} checks passed")
    
    if passed == total:
        print("🎉 Ready for training!")
        return True
    else:
        print("⚠️ Some issues found")
        return False

if __name__ == "__main__":
    verify()
"""
    
    with open(f"{MONOX_ROOT}/verify_setup.py", "w") as f:
        f.write(verify_script)
    
    print("✅ Created: setup_monox_colab.py")
    print("✅ Created: launch_training.py") 
    print("✅ Created: verify_setup.py")

def main():
    """Main function"""
    print("🚀 Creating MonoX Setup Files")
    print("=" * 40)
    
    # Create all files
    create_all_files()
    
    print("\n🎉 All files created successfully!")
    print("\nNext steps:")
    print("1. !python /content/MonoX/setup_monox_colab.py")
    print("2. !python /content/MonoX/verify_setup.py") 
    print("3. !python /content/MonoX/launch_training.py")
    
    # Show GPU status
    print("\n🖥️ Current GPU Status:")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ GPU detected")
            # Show GPU info
            for line in result.stdout.split('\n'):
                if any(gpu in line for gpu in ['Tesla', 'RTX', 'GTX', 'A100', 'V100', 'L4']):
                    print(f"   {line.strip()}")
        else:
            print("❌ No GPU detected")
    except:
        print("❌ GPU check failed")

if __name__ == "__main__":
    main()