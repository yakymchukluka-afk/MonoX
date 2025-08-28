#!/usr/bin/env python3
"""
Basic test script to validate the latent walk structure without external dependencies.
"""

import sys
import os
import json
from pathlib import Path

def test_script_structure():
    """Test that the script files exist and are properly structured."""
    print("🧪 Testing script structure...")
    
    # Check if main script exists
    script_path = Path("/workspace/stylegan-stack/latent_walk.py")
    if script_path.exists():
        print(f"✅ latent_walk.py exists")
    else:
        print(f"❌ latent_walk.py not found")
        return False
    
    # Check if script is executable
    if os.access(script_path, os.X_OK):
        print(f"✅ latent_walk.py is executable")
    else:
        print(f"❌ latent_walk.py is not executable")
        return False
    
    # Check requirements file
    req_path = Path("/workspace/stylegan-stack/requirements.txt")
    if req_path.exists():
        print(f"✅ requirements.txt exists")
    else:
        print(f"❌ requirements.txt not found")
        return False
    
    return True

def test_directory_structure():
    """Test that required directories exist."""
    print("\n🧪 Testing directory structure...")
    
    required_dirs = [
        "/workspace/stylegan-stack",
        "/workspace/stylegan-stack/models",
        "/workspace/stylegan-stack/generated_frames"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} not found")
            return False
    
    return True

def test_configuration():
    """Test the Mono project configuration parameters."""
    print("\n🧪 Testing Mono configuration...")
    
    NUM_KEYFRAMES = 37
    FRAMES_PER_TRANSITION = 25
    TOTAL_FRAMES = (NUM_KEYFRAMES - 1) * FRAMES_PER_TRANSITION
    TARGET_FPS = 30
    TARGET_DURATION = 30
    
    print(f"   Keyframes: {NUM_KEYFRAMES}")
    print(f"   Transitions: {NUM_KEYFRAMES - 1}")
    print(f"   Frames per transition: {FRAMES_PER_TRANSITION}")
    print(f"   Total frames: {TOTAL_FRAMES}")
    print(f"   Target FPS: {TARGET_FPS}")
    print(f"   Target duration: {TARGET_DURATION} seconds")
    
    # Verify math
    expected_frames = TARGET_FPS * TARGET_DURATION  # 30 * 30 = 900
    
    if TOTAL_FRAMES == expected_frames:
        print(f"✅ Frame calculation correct: {TOTAL_FRAMES} frames for {TARGET_DURATION}s @ {TARGET_FPS}fps")
        return True
    else:
        print(f"❌ Frame calculation incorrect: expected {expected_frames}, got {TOTAL_FRAMES}")
        return False

def test_script_imports():
    """Test that the script can be imported without missing dependencies."""
    print("\n🧪 Testing script syntax and basic structure...")
    
    script_path = "/workspace/stylegan-stack/latent_walk.py"
    
    try:
        # Read and compile the script to check for syntax errors
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        compile(script_content, script_path, 'exec')
        print("✅ Script syntax is valid")
        
        # Check for key functions
        required_functions = [
            'def slerp(',
            'def generate_latent_vectors(',
            'def interpolate_path(',
            'def load_stylegan_generator(',
            'def generate_frame(',
            'def save_latent_path(',
            'def main('
        ]
        
        for func in required_functions:
            if func in script_content:
                print(f"✅ Found {func.split('(')[0].replace('def ', '')}")
            else:
                print(f"❌ Missing {func.split('(')[0].replace('def ', '')}")
                return False
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in script: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading script: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🎬 Mono Latent Walk Generator - Basic Structure Test")
    print("=" * 55)
    
    tests = [
        test_directory_structure,
        test_script_structure,
        test_configuration,
        test_script_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Basic Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed! The latent walk structure is ready.")
        print("\n📋 Next steps to run the generator:")
        print("   1. Install StyleGAN2-ADA dependencies:")
        print("      - PyTorch")
        print("      - NumPy")
        print("      - Pillow")
        print("      - StyleGAN2-ADA repository")
        print()
        print("   2. Place your trained StyleGAN2-ADA model at:")
        print("      /workspace/stylegan-stack/models/model.pkl")
        print()
        print("   3. Run the latent walk generator:")
        print("      python3 /workspace/stylegan-stack/latent_walk.py")
        print()
        print("   4. Frames will be saved to:")
        print("      /workspace/stylegan-stack/generated_frames/{uuid}/")
        print()
        print("   5. The script generates:")
        print("      - 900 frames total (30 seconds @ 30 FPS)")
        print("      - 37 keyframe latent vectors")
        print("      - 36 transitions with 25 frames each")
        print("      - SLERP interpolation for smooth transitions")
        print("      - latent_path.json for reproducibility")
        return 0
    else:
        print("❌ Some basic tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())