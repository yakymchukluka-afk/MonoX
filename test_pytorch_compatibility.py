#!/usr/bin/env python3
"""
Test script to verify PyTorch compatibility with StyleGAN2-ADA
"""

import sys
import os
import torch
import warnings

# Add StyleGAN2-ADA to path
sys.path.insert(0, '/workspace/train/runpod-hf/vendor/stylegan2ada')

def test_pytorch_version():
    """Test PyTorch version and CUDA availability"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
    print()

def test_grid_sample_gradfix():
    """Test the grid_sample_gradfix compatibility"""
    print("Testing grid_sample_gradfix compatibility...")
    
    try:
        from torch_utils.ops import grid_sample_gradfix
        
        # Test basic functionality
        print("✓ grid_sample_gradfix imported successfully")
        
        # Test with sample data
        batch_size = 2
        channels = 3
        height = 64
        width = 64
        
        # Create sample input and grid
        input_tensor = torch.randn(batch_size, channels, height, width)
        grid = torch.randn(batch_size, height, width, 2) * 2 - 1  # Normalize to [-1, 1]
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            grid = grid.cuda()
        
        # Test forward pass
        try:
            output = grid_sample_gradfix.grid_sample(input_tensor, grid)
            print(f"✓ Forward pass successful, output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
        
        # Test backward pass
        try:
            # Make sure input requires gradients
            input_tensor.requires_grad_(True)
            grid.requires_grad_(True)
            output = grid_sample_gradfix.grid_sample(input_tensor, grid)
            output.sum().backward()
            print("✓ Backward pass successful")
        except Exception as e:
            print(f"✗ Backward pass failed: {e}")
            return False
        
        print("✓ grid_sample_gradfix is working correctly")
        return True
        
    except Exception as e:
        print(f"✗ grid_sample_gradfix test failed: {e}")
        return False

def test_stylegan_imports():
    """Test StyleGAN2-ADA imports"""
    print("Testing StyleGAN2-ADA imports...")
    
    try:
        from training import networks
        print("✓ networks imported successfully")
        
        from training import training_loop
        print("✓ training_loop imported successfully")
        
        from training import augment
        print("✓ augment imported successfully")
        
        print("✓ All StyleGAN2-ADA imports successful")
        return True
        
    except Exception as e:
        print(f"✗ StyleGAN2-ADA import failed: {e}")
        return False

def test_network_creation():
    """Test creating StyleGAN2-ADA networks"""
    print("Testing network creation...")
    
    try:
        from training import networks
        
        # Test Generator creation
        generator = networks.Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=1024,
            img_channels=3
        )
        print("✓ Generator created successfully")
        
        # Test Discriminator creation
        discriminator = networks.Discriminator(
            c_dim=0,
            img_resolution=1024,
            img_channels=3
        )
        print("✓ Discriminator created successfully")
        
        # Test forward pass
        if torch.cuda.is_available():
            generator = generator.cuda()
            discriminator = discriminator.cuda()
        
        z = torch.randn(1, 512)
        if torch.cuda.is_available():
            z = z.cuda()
        
        with torch.no_grad():
            fake_img = generator(z, None)
            print(f"✓ Generator forward pass successful, output shape: {fake_img.shape}")
            
            fake_logits = discriminator(fake_img, None)
            print(f"✓ Discriminator forward pass successful, output shape: {fake_logits.shape}")
        
        print("✓ Network creation and forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Network creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all compatibility tests"""
    print("=" * 60)
    print("StyleGAN2-ADA PyTorch Compatibility Test")
    print("=" * 60)
    print()
    
    # Test PyTorch version
    test_pytorch_version()
    
    # Test grid_sample_gradfix
    grid_test_passed = test_grid_sample_gradfix()
    print()
    
    # Test StyleGAN2-ADA imports
    import_test_passed = test_stylegan_imports()
    print()
    
    # Test network creation
    network_test_passed = test_network_creation()
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    print(f"Grid Sample Gradfix: {'PASS' if grid_test_passed else 'FAIL'}")
    print(f"StyleGAN2-ADA Imports: {'PASS' if import_test_passed else 'FAIL'}")
    print(f"Network Creation: {'PASS' if network_test_passed else 'FAIL'}")
    
    all_passed = grid_test_passed and import_test_passed and network_test_passed
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! StyleGAN2-ADA is compatible with this PyTorch version.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())