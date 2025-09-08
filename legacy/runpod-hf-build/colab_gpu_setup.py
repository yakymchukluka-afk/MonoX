#!/usr/bin/env python3
"""
Colab GPU Setup and Verification Script
Ensures proper GPU configuration for MonoX training.
"""

import subprocess
import sys
import os
import torch

def check_colab_runtime():
    """Check if we're in Colab and what runtime type."""
    try:
        import google.colab
        print("✅ Running in Google Colab")
        return True
    except ImportError:
        print("❌ Not running in Google Colab")
        return False

def check_gpu_runtime():
    """Verify GPU runtime is selected."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("✅ GPU runtime detected")
        print("GPU Info:")
        lines = result.stdout.split('\n')
        for line in lines[0:6]:  # First few lines with GPU info
            if line.strip():
                print(f"   {line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ GPU runtime not detected!")
        print("💡 Go to Runtime → Change runtime type → Hardware accelerator → GPU")
        return False

def setup_gpu_environment():
    """Set up optimal GPU environment."""
    print("\n🔧 Setting up GPU environment...")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Create torch extensions directory
    os.makedirs('/tmp/torch_extensions', exist_ok=True)
    
    print("✅ Environment variables set:")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"   TORCH_EXTENSIONS_DIR: {os.environ.get('TORCH_EXTENSIONS_DIR')}")

def test_pytorch_gpu():
    """Test PyTorch GPU functionality."""
    print("\n🧪 Testing PyTorch GPU...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU tensor operations
        try:
            print("Testing GPU tensor operations...")
            device = torch.device('cuda:0')
            
            # Create tensors on GPU
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            
            # Perform computation
            c = torch.mm(a, b)
            
            # Check memory usage
            allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            cached = torch.cuda.memory_reserved(0) / 1024**2  # MB
            
            print(f"✅ GPU test passed!")
            print(f"   Tensor shape: {c.shape}")
            print(f"   Memory allocated: {allocated:.1f} MB")
            print(f"   Memory cached: {cached:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
    else:
        print("❌ CUDA not available in PyTorch")
        return False

def install_requirements():
    """Install/upgrade required packages."""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "torch>=2.0.0",
        "torchvision",
        "hydra-core>=1.2.0", 
        "omegaconf>=2.2.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                          check=True, capture_output=True)
            print(f"✅ {req} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")

def run_gpu_stress_test():
    """Run a brief GPU stress test to verify it's working."""
    print("\n🔥 Running GPU stress test...")
    
    if not torch.cuda.is_available():
        print("❌ Skipping stress test - CUDA not available")
        return False
    
    try:
        device = torch.device('cuda:0')
        
        # Create large tensors
        print("Creating large tensors on GPU...")
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        
        # Perform multiple operations
        print("Performing matrix operations...")
        for i in range(10):
            c = torch.mm(a, b)
            a = c[:2000, :2000]  # Keep size manageable
        
        # Check final memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"✅ Stress test passed! Memory: {allocated:.1f} MB")
        
        # Clean up
        del a, b, c
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False

def main():
    """Run complete GPU setup and verification."""
    print("🚀 Colab GPU Setup and Verification")
    print("=" * 50)
    
    # Check if in Colab
    in_colab = check_colab_runtime()
    
    # Check GPU runtime
    gpu_available = check_gpu_runtime()
    if not gpu_available:
        print("\n❌ GPU setup failed! Please select GPU runtime.")
        return False
    
    # Set up environment
    setup_gpu_environment()
    
    # Install requirements
    install_requirements()
    
    # Test PyTorch GPU
    pytorch_ok = test_pytorch_gpu()
    if not pytorch_ok:
        print("\n❌ PyTorch GPU test failed!")
        return False
    
    # Run stress test
    stress_ok = run_gpu_stress_test()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 GPU SETUP SUMMARY:")
    print(f"   Colab detected: {'✅' if in_colab else '❌'}")
    print(f"   GPU runtime: {'✅' if gpu_available else '❌'}")
    print(f"   PyTorch GPU: {'✅' if pytorch_ok else '❌'}")
    print(f"   Stress test: {'✅' if stress_ok else '❌'}")
    
    if gpu_available and pytorch_ok and stress_ok:
        print("\n🎉 GPU setup complete! Ready for training.")
        print("\n🚀 Next steps:")
        print("   1. Clone MonoX repo")
        print("   2. Run: python train_gpu_forced.py ...")
        return True
    else:
        print("\n❌ GPU setup incomplete. Check errors above.")
        return False

if __name__ == "__main__":
    main()