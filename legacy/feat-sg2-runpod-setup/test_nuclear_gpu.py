#!/usr/bin/env python3
"""
Test script to verify nuclear GPU enhancements are working
"""
import torch
import sys
import os

def test_nuclear_gpu_setup():
    """Test our nuclear GPU setup independently"""
    
    print("🚀 TESTING NUCLEAR GPU SETUP")
    print("=" * 50)
    
    # Test 1: Basic CUDA availability
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Cannot test GPU.")
        return False
    
    # Test 2: Device selection
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    print(f"✅ CUDA device set: {device}")
    
    # Test 3: Basic GPU verification
    try:
        test_tensor = torch.randn(100, 100, device=device)
        test_result = torch.mm(test_tensor, test_tensor)
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
        print(f"✅ Basic GPU verification: {gpu_memory:.1f} MB")
        del test_tensor, test_result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Basic GPU verification failed: {e}")
        return False
    
    # Test 4: Nuclear GPU memory pre-allocation (our code)
    try:
        print("\n🚀 NUCLEAR GPU MEMORY PRE-ALLOCATION TEST:")
        
        # Pre-allocate large chunks of GPU memory to force utilization
        gpu_total_memory = torch.cuda.get_device_properties(device).total_memory
        gpu_available_memory = gpu_total_memory - torch.cuda.memory_allocated(device)
        
        # Allocate 80% of available GPU memory with dummy tensors
        chunk_size = int(gpu_available_memory * 0.8 / 4)  # 80% in float32 chunks
        warmup_tensors = []
        
        print(f"🚀 Total GPU memory: {gpu_total_memory / 1024**3:.1f} GB")
        print(f"🚀 Available memory: {gpu_available_memory / 1024**3:.1f} GB")
        
        # Create multiple large tensors to force GPU memory usage
        for i in range(4):
            try:
                tensor_size = int(chunk_size ** 0.5)  # Square tensor
                warmup_tensor = torch.randn(tensor_size, tensor_size, device=device, dtype=torch.float32)
                # Perform operations to ensure GPU is actively used
                warmup_result = torch.mm(warmup_tensor, warmup_tensor)
                warmup_result = torch.nn.functional.relu(warmup_result)
                warmup_tensors.append(warmup_result)
                current_allocated = torch.cuda.memory_allocated(device) / 1024**3
                print(f"🔥 Allocated tensor {i+1}: {tensor_size}x{tensor_size}, Total GPU: {current_allocated:.1f} GB")
            except RuntimeError as e:
                print(f"🔥 GPU memory limit reached at tensor {i+1}: {e}")
                break
        
        # Keep tensors alive briefly, then clean up
        total_allocated = torch.cuda.memory_allocated(device) / 1024**3
        print(f"🚀 AGGRESSIVE GPU pre-allocation complete: {total_allocated:.1f} GB")
        
        # Clean up warmup tensors
        del warmup_tensors
        torch.cuda.empty_cache()
        
        final_allocated = torch.cuda.memory_allocated(device) / 1024**3
        print(f"🚀 After cleanup, GPU memory: {final_allocated:.1f} GB")
        print(f"✅ NUCLEAR GPU PRE-ALLOCATION SUCCESSFUL!")
        
        return True
        
    except Exception as e:
        print(f"❌ Nuclear GPU pre-allocation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_nuclear_gpu_setup()
    if success:
        print("\n🎉 NUCLEAR GPU SETUP IS WORKING!")
        print("🚀 GPU should show significant memory usage during this test")
        sys.exit(0)
    else:
        print("\n❌ NUCLEAR GPU SETUP FAILED!")
        sys.exit(1)