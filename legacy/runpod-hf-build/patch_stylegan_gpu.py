#!/usr/bin/env python3
"""
Patch StyleGAN-V to force GPU usage at the source level.
This directly modifies the StyleGAN-V training script to ensure GPU usage.
"""

import os
import sys
from pathlib import Path

def patch_stylegan_training_loop():
    """Patch the StyleGAN-V training loop to force GPU usage."""
    
    # Find the StyleGAN-V training loop file
    repo_root = Path.cwd()
    training_loop_file = repo_root / ".external" / "stylegan-v" / "src" / "training" / "training_loop.py"
    
    if not training_loop_file.exists():
        print(f"❌ Training loop file not found: {training_loop_file}")
        return False
    
    print(f"🔧 Patching StyleGAN-V training loop: {training_loop_file}")
    
    # Read the current file
    with open(training_loop_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "# MONOX GPU FORCING PATCH" in content:
        print("✅ StyleGAN-V already patched for GPU forcing")
        return True
    
    # Find the device line and add aggressive GPU forcing
    device_line = "device = torch.device('cuda', rank)"
    
    if device_line not in content:
        print("❌ Could not find device line to patch")
        return False
    
    # Create the GPU forcing patch
    gpu_patch = '''    # MONOX GPU FORCING PATCH - Start
    import torch
    print(f"🔥 MONOX: Forcing GPU device for rank {rank}")
    
    # Force CUDA device selection
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    
    # Verify GPU is working with a test tensor
    try:
        test_tensor = torch.randn(100, 100, device=device)
        test_result = torch.mm(test_tensor, test_tensor)
        gpu_memory = torch.cuda.memory_allocated(rank) / 1024**2
        print(f"🔥 MONOX: GPU verification successful on {device}")
        print(f"🔥 MONOX: GPU memory allocated: {gpu_memory:.1f} MB")
        del test_tensor, test_result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ MONOX: GPU verification failed: {e}")
        print("❌ MONOX: Training will likely use CPU!")
    
    # Force all future tensors to use this device
    torch.cuda.set_device(device)
    print(f"🔥 MONOX: Set default CUDA device to {device}")
    # MONOX GPU FORCING PATCH - End
    
    '''
    
    # Replace the device line with our patch
    patched_content = content.replace(
        f"    {device_line}",
        gpu_patch + f"    # Original: {device_line}"
    )
    
    # Write the patched file
    with open(training_loop_file, 'w') as f:
        f.write(patched_content)
    
    print("✅ StyleGAN-V training loop patched for GPU forcing")
    return True

def patch_stylegan_train_script():
    """Patch the main StyleGAN-V train.py to add GPU debugging."""
    
    repo_root = Path.cwd()
    train_file = repo_root / ".external" / "stylegan-v" / "src" / "train.py"
    
    if not train_file.exists():
        print(f"❌ Train file not found: {train_file}")
        return False
    
    print(f"🔧 Patching StyleGAN-V train.py: {train_file}")
    
    # Read the current file
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "# MONOX TRAIN GPU PATCH" in content:
        print("✅ StyleGAN-V train.py already patched")
        return True
    
    # Find the subprocess_fn function start
    subprocess_fn_line = "def subprocess_fn(rank, args, temp_dir):"
    
    if subprocess_fn_line not in content:
        print("❌ Could not find subprocess_fn to patch")
        return False
    
    # Create the train patch
    train_patch = '''def subprocess_fn(rank, args, temp_dir):
    # MONOX TRAIN GPU PATCH - Start
    import torch
    print(f"🔥 MONOX TRAIN: Starting subprocess for rank {rank}")
    print(f"🔥 MONOX TRAIN: args.num_gpus = {args.num_gpus}")
    
    # Force GPU setup at the very beginning
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        print(f"🔥 MONOX TRAIN: Set CUDA device to {rank}")
        
        # Test GPU immediately
        try:
            test = torch.randn(50, 50, device=f'cuda:{rank}')
            _ = torch.mm(test, test)
            print(f"🔥 MONOX TRAIN: GPU {rank} working, memory: {torch.cuda.memory_allocated(rank)/1024**2:.1f} MB")
            del test
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ MONOX TRAIN: GPU {rank} test failed: {e}")
    else:
        print("❌ MONOX TRAIN: CUDA not available!")
    # MONOX TRAIN GPU PATCH - End
    
'''
    
    # Replace the function definition
    patched_content = content.replace(subprocess_fn_line, train_patch)
    
    # Write the patched file
    with open(train_file, 'w') as f:
        f.write(patched_content)
    
    print("✅ StyleGAN-V train.py patched for GPU debugging")
    return True

def main():
    """Apply all GPU forcing patches to StyleGAN-V."""
    print("🔥 MonoX StyleGAN-V GPU Forcing Patcher")
    print("=" * 50)
    
    success = True
    
    # Patch training loop
    if not patch_stylegan_training_loop():
        success = False
    
    # Patch train script
    if not patch_stylegan_train_script():
        success = False
    
    if success:
        print("\n🎉 All StyleGAN-V GPU patches applied successfully!")
        print("🚀 Training should now force GPU usage at the source level")
    else:
        print("\n❌ Some patches failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()