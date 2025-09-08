#!/usr/bin/env python3
"""
GPU Diagnostic Script for MonoX Training
Diagnoses and fixes common GPU utilization issues in Colab.
"""

import torch
import subprocess
import sys
import os

def check_nvidia_smi():
    """Check if nvidia-smi is available and working."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("✅ nvidia-smi working:")
        print(result.stdout.split('\n')[0:3])  # First few lines
        return True
    except Exception as e:
        print(f"❌ nvidia-smi failed: {e}")
        return False

def check_cuda_availability():
    """Check CUDA availability in PyTorch."""
    print(f"🔍 PyTorch version: {torch.__version__}")
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        print(f"✅ Current CUDA device: {torch.cuda.current_device()}")
        print(f"✅ CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA capability: {torch.cuda.get_device_capability(0)}")
        return True
    else:
        print("❌ CUDA not available in PyTorch!")
        return False

def test_gpu_tensor():
    """Test GPU tensor operations."""
    try:
        # Create a tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.mm(x, y)  # Matrix multiplication on GPU
        print(f"✅ GPU tensor test passed: {z.shape} tensor created on {z.device}")
        return True
    except Exception as e:
        print(f"❌ GPU tensor test failed: {e}")
        return False

def check_environment():
    """Check relevant environment variables."""
    print("🔍 Environment variables:")
    relevant_vars = ['CUDA_VISIBLE_DEVICES', 'TORCH_EXTENSIONS_DIR', 'PYTHONPATH']
    for var in relevant_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")

def run_diagnostics():
    """Run comprehensive GPU diagnostics."""
    print("🔥 GPU Diagnostic Report")
    print("=" * 50)
    
    # Check nvidia-smi
    print("\n1. Checking nvidia-smi...")
    nvidia_ok = check_nvidia_smi()
    
    # Check CUDA in PyTorch
    print("\n2. Checking CUDA in PyTorch...")
    cuda_ok = check_cuda_availability()
    
    # Test GPU operations
    print("\n3. Testing GPU operations...")
    if cuda_ok:
        gpu_test_ok = test_gpu_tensor()
    else:
        gpu_test_ok = False
        print("⏭️ Skipping GPU test (CUDA not available)")
    
    # Check environment
    print("\n4. Checking environment...")
    check_environment()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY:")
    print(f"   nvidia-smi: {'✅' if nvidia_ok else '❌'}")
    print(f"   CUDA available: {'✅' if cuda_ok else '❌'}")
    print(f"   GPU operations: {'✅' if gpu_test_ok else '❌'}")
    
    if nvidia_ok and cuda_ok and gpu_test_ok:
        print("\n🎉 GPU is working correctly!")
        print("💡 Issue might be in training configuration...")
        return True
    else:
        print("\n❌ GPU issues detected!")
        print("💡 Need to fix GPU setup...")
        return False

def suggest_fixes():
    """Suggest fixes for common GPU issues."""
    print("\n🔧 SUGGESTED FIXES:")
    print("\n1. Restart Colab runtime:")
    print("   Runtime → Restart runtime")
    
    print("\n2. Ensure GPU runtime:")
    print("   Runtime → Change runtime type → GPU")
    
    print("\n3. Set environment variables:")
    print("   os.environ['CUDA_VISIBLE_DEVICES'] = '0'")
    print("   os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'")
    
    print("\n4. Reinstall PyTorch with CUDA:")
    print("   !pip install torch torchvision --force-reinstall")

if __name__ == "__main__":
    success = run_diagnostics()
    if not success:
        suggest_fixes()