#!/usr/bin/env python3
"""
Nuclear Colab Command - Single Copy-Paste Solution
Complete MonoX setup with maximum GPU forcing for fresh Colab session.
"""

# ===== COMPLETE NUCLEAR COLAB SETUP =====
# Copy and paste this ENTIRE block into a fresh Colab session

def nuclear_colab_setup():
    """Complete nuclear setup for Colab."""
    
    print("🚀 NUCLEAR COLAB SETUP STARTING")
    print("=" * 60)
    
    # Step 1: Mount drive
    print("📁 Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Step 2: Nuclear GPU environment
    print("🔥 Setting up nuclear GPU environment...")
    import os
    import torch
    
    # Force all GPU environment variables
    gpu_env = {
        'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_USE_CUDA_DSA': '1',
        'NVIDIA_TF32_OVERRIDE': '0',
        'FORCE_CUDA': '1',
        'USE_CUDA': '1',
        'CUDA_CACHE_DISABLE': '0'
    }
    
    for key, value in gpu_env.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    # Step 3: Verify GPU immediately
    print("\n🔍 GPU Verification:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        
        # Aggressive GPU test
        test = torch.randn(2000, 2000, device='cuda')
        result = torch.mm(test, test)
        memory_mb = torch.cuda.memory_allocated(0) / 1024**2
        print(f"   ✅ GPU test passed: {result.device}")
        print(f"   ✅ GPU memory: {memory_mb:.1f} MB")
        del test, result
        torch.cuda.empty_cache()
    else:
        print("   ❌ CUDA NOT AVAILABLE!")
        print("   ❌ Check Runtime → Change runtime type → GPU")
        return False
    
    return True

def clone_and_setup():
    """Clone MonoX and apply all patches."""
    
    print("\n📦 Cloning MonoX...")
    import subprocess
    import os
    
    # Clean slate
    subprocess.run(['rm', '-rf', '/content/MonoX'], check=False)
    
    # Clone repo
    subprocess.run(['git', 'clone', 'https://github.com/yakymchukluka-afk/MonoX'], 
                  cwd='/content', check=True)
    
    os.chdir('/content/MonoX')
    
    # Initialize submodules
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], check=True)
    
    # Pull latest
    subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
    subprocess.run(['git', 'submodule', 'update', '--remote'], check=True)
    
    print("✅ MonoX cloned and updated")

def install_and_patch():
    """Install dependencies and apply patches."""
    
    print("\n📦 Installing dependencies...")
    import subprocess
    
    # Install dependencies
    result = subprocess.run(['python', 'colab_install_v2.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Dependency installation failed: {result.stderr}")
        return False
    
    print("✅ Dependencies installed")
    
    # Apply GPU patches
    print("\n🔧 Applying nuclear GPU patches...")
    result = subprocess.run(['python', 'patch_stylegan_gpu.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ GPU patching failed: {result.stderr}")
        return False
    
    print("✅ Nuclear GPU patches applied")
    return True

def launch_nuclear_training():
    """Launch nuclear GPU-forced training."""
    
    print("\n🚀 LAUNCHING NUCLEAR GPU TRAINING...")
    print("=" * 60)
    
    import subprocess
    
    cmd = [
        'python', 'train_super_gpu_forced.py',
        'exp_suffix=nuclear_fresh_session',
        'dataset.path=/content/drive/MyDrive/MonoX/dataset',
        'dataset.resolution=256',
        'training.total_kimg=50',
        'training.snapshot_kimg=10',
        'visualizer.save_every_kimg=5',
        'num_gpus=1'
    ]
    
    print("Command:", ' '.join(cmd))
    print("\n🔥 NUCLEAR TRAINING OUTPUT:")
    print("=" * 60)
    
    # Run training with real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, universal_newlines=True)
    
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    return process.returncode == 0

def main():
    """Main nuclear setup function."""
    
    if not nuclear_colab_setup():
        return False
    
    clone_and_setup()
    
    if not install_and_patch():
        return False
    
    return launch_nuclear_training()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 NUCLEAR TRAINING LAUNCHED SUCCESSFULLY!")
    else:
        print("\n❌ NUCLEAR SETUP FAILED!")

# ===== SINGLE COMMAND VERSION =====
# Just run: exec(open('nuclear_colab_command.py').read())

# ===== OR COPY-PASTE THIS ENTIRE BLOCK =====
"""
from google.colab import drive
import os
import torch
import subprocess

# Mount drive
drive.mount('/content/drive')

# Nuclear GPU environment
os.environ.update({
    'CUDA_VISIBLE_DEVICES': '0',
    'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
    'CUDA_LAUNCH_BLOCKING': '1',
    'FORCE_CUDA': '1'
})

# Verify GPU
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    test = torch.randn(1000, 1000, device='cuda')
    print(f"GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    del test; torch.cuda.empty_cache()

# Setup MonoX
!rm -rf /content/MonoX
!git clone https://github.com/yakymchukluka-afk/MonoX
%cd /content/MonoX
!git submodule update --init --recursive
!git pull origin main && git submodule update --remote

# Install and patch
!python colab_install_v2.py
!python patch_stylegan_gpu.py

# NUCLEAR TRAINING
!python train_super_gpu_forced.py exp_suffix=nuclear dataset.path=/content/drive/MyDrive/MonoX/dataset dataset.resolution=256 training.total_kimg=25 training.snapshot_kimg=5 visualizer.save_every_kimg=2 num_gpus=1
"""