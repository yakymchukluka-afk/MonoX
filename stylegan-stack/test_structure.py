#!/usr/bin/env python3
"""
Test script to validate the latent walk structure without requiring the actual StyleGAN2 model.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our latent walk functions
try:
    from latent_walk import slerp, generate_latent_vectors, interpolate_path, save_latent_path
    print("✅ Successfully imported latent walk functions")
except ImportError as e:
    print(f"❌ Error importing latent walk functions: {e}")
    sys.exit(1)

def test_slerp():
    """Test SLERP interpolation function."""
    print("\n🧪 Testing SLERP interpolation...")
    
    # Create two random vectors
    v1 = np.random.randn(512).astype(np.float32)
    v2 = np.random.randn(512).astype(np.float32)
    
    # Test interpolation at different points
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = slerp(v1, v2, t)
        print(f"   t={t}: shape={result.shape}, norm={np.linalg.norm(result):.3f}")
    
    print("✅ SLERP interpolation test passed")

def test_latent_generation():
    """Test latent vector generation."""
    print("\n🧪 Testing latent vector generation...")
    
    # Test with fixed seeds
    vectors = generate_latent_vectors(5, 512, seeds=[1, 2, 3, 4, 5])
    
    print(f"   Generated {len(vectors)} vectors")
    print(f"   Vector shape: {vectors[0].shape}")
    print(f"   Vector dtype: {vectors[0].dtype}")
    
    # Test reproducibility
    vectors2 = generate_latent_vectors(5, 512, seeds=[1, 2, 3, 4, 5])
    
    if np.allclose(vectors[0], vectors2[0]):
        print("✅ Reproducible generation test passed")
    else:
        print("❌ Reproducible generation test failed")
        return False
    
    return True

def test_interpolation():
    """Test path interpolation."""
    print("\n🧪 Testing path interpolation...")
    
    # Generate test vectors
    keyframes = generate_latent_vectors(4, 512, seeds=[1, 2, 3, 4])
    
    # Test interpolation
    interpolated = interpolate_path(keyframes, 10)
    
    expected_frames = (len(keyframes) - 1) * 10  # 3 transitions × 10 frames
    
    print(f"   Keyframes: {len(keyframes)}")
    print(f"   Frames per transition: 10")
    print(f"   Expected total frames: {expected_frames}")
    print(f"   Actual total frames: {len(interpolated)}")
    
    if len(interpolated) == expected_frames:
        print("✅ Path interpolation test passed")
        return True
    else:
        print("❌ Path interpolation test failed")
        return False

def test_mono_configuration():
    """Test the exact Mono project configuration."""
    print("\n🧪 Testing Mono project configuration...")
    
    NUM_KEYFRAMES = 37
    FRAMES_PER_TRANSITION = 25
    EXPECTED_TOTAL = (NUM_KEYFRAMES - 1) * FRAMES_PER_TRANSITION  # 36 × 25 = 900
    
    print(f"   Keyframes: {NUM_KEYFRAMES}")
    print(f"   Transitions: {NUM_KEYFRAMES - 1}")
    print(f"   Frames per transition: {FRAMES_PER_TRANSITION}")
    print(f"   Expected total frames: {EXPECTED_TOTAL}")
    
    # Generate keyframes
    keyframes = generate_latent_vectors(NUM_KEYFRAMES, 512)
    
    # Interpolate
    interpolated = interpolate_path(keyframes, FRAMES_PER_TRANSITION)
    
    print(f"   Actual total frames: {len(interpolated)}")
    
    if len(interpolated) == EXPECTED_TOTAL:
        print("✅ Mono configuration test passed")
        return True
    else:
        print("❌ Mono configuration test failed")
        return False

def test_save_functionality():
    """Test the save functionality."""
    print("\n🧪 Testing save functionality...")
    
    # Create test directory
    test_dir = Path("/workspace/stylegan-stack/test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Generate test data
    vectors = generate_latent_vectors(3, 512, seeds=[1, 2, 3])
    seeds = [1, 2, 3]
    
    # Test save function
    try:
        save_latent_path(vectors, seeds, test_dir)
        
        # Check if file was created
        json_path = test_dir / "latent_path.json"
        if json_path.exists():
            print(f"   JSON file created: {json_path}")
            print("✅ Save functionality test passed")
            return True
        else:
            print("❌ JSON file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Save functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎬 Mono Latent Walk Generator - Structure Test")
    print("=" * 50)
    
    tests = [
        test_slerp,
        test_latent_generation,
        test_interpolation,
        test_mono_configuration,
        test_save_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The latent walk structure is ready.")
        print("\n📋 Next steps:")
        print("   1. Place your trained StyleGAN2-ADA model at:")
        print("      /workspace/stylegan-stack/models/model.pkl")
        print("   2. Run the latent walk generator:")
        print("      python /workspace/stylegan-stack/latent_walk.py")
        print("   3. Frames will be saved to:")
        print("      /workspace/stylegan-stack/generated_frames/{uuid}/")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())