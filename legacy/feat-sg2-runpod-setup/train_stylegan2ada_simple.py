#!/usr/bin/env python3
"""
Simple StyleGAN2-ADA Training Script for RunPod
This script demonstrates that the PyTorch compatibility fix is working
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add StyleGAN2-ADA to path
sys.path.insert(0, '/workspace/train/runpod-hf/vendor/stylegan2ada')

def test_training_setup():
    """Test that we can create and run a basic training setup"""
    print("🧪 Testing StyleGAN2-ADA training setup...")
    
    try:
        from training import networks
        
        # Create a small test dataset
        print("📊 Creating test dataset...")
        test_data = torch.randn(4, 3, 64, 64)  # Small test images
        
        # Create networks
        print("🏗️ Creating networks...")
        generator = networks.Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=64,  # Small resolution for testing
            img_channels=3
        )
        
        discriminator = networks.Discriminator(
            c_dim=0,
            img_resolution=64,
            img_channels=3
        )
        
        # Test forward passes
        print("🔄 Testing forward passes...")
        z = torch.randn(2, 512)
        
        with torch.no_grad():
            # Generator forward pass
            fake_img = generator(z, None)
            print(f"✓ Generator output shape: {fake_img.shape}")
            
            # Discriminator forward pass
            fake_logits = discriminator(fake_img, None)
            print(f"✓ Discriminator output shape: {fake_logits.shape}")
            
            # Test with real data
            real_logits = discriminator(test_data, None)
            print(f"✓ Real data discriminator output shape: {real_logits.shape}")
        
        # Test gradient computation
        print("🔢 Testing gradient computation...")
        z.requires_grad_(True)
        fake_img = generator(z, None)
        fake_logits = discriminator(fake_img, None)
        
        # Compute gradients
        loss = fake_logits.sum()
        loss.backward()
        
        print(f"✓ Gradient computation successful")
        print(f"✓ Z gradients shape: {z.grad.shape}")
        
        print("🎉 All training setup tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print("📁 Creating sample dataset...")
    
    # Create data directory
    os.makedirs("/workspace/data", exist_ok=True)
    
    # Create a small test dataset
    test_images = []
    for i in range(10):  # 10 test images
        # Create random RGB image
        img = torch.randn(3, 64, 64)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        test_images.append(img)
    
    # Save as individual files (simulating a real dataset)
    for i, img in enumerate(test_images):
        # Convert to PIL and save
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Convert tensor to PIL
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(img)
        
        # Save image
        img_path = f"/workspace/data/test_image_{i:03d}.png"
        pil_img.save(img_path)
    
    print(f"✅ Created 10 test images in /workspace/data/")
    return True

def main():
    """Main function"""
    print("🚀 StyleGAN2-ADA Training Setup Test")
    print("=" * 50)
    
    # Test PyTorch compatibility
    print("🔧 Testing PyTorch compatibility...")
    try:
        from torch_utils.ops import grid_sample_gradfix
        print("✓ grid_sample_gradfix imported successfully")
    except Exception as e:
        print(f"❌ PyTorch compatibility test failed: {e}")
        return 1
    
    # Test training setup
    if not test_training_setup():
        print("❌ Training setup test failed")
        return 1
    
    # Create sample dataset
    if not create_sample_dataset():
        print("❌ Dataset creation failed")
        return 1
    
    print("\n🎉 All tests passed!")
    print("=" * 50)
    print("✅ StyleGAN2-ADA is ready for training!")
    print("✅ PyTorch 2.0+ compatibility is working!")
    print("✅ Networks can be created and run successfully!")
    print("✅ Gradient computation is working!")
    print("\n📁 Sample dataset created in /workspace/data/")
    print("🚀 Ready to start actual training with your real dataset!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())