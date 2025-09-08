#!/usr/bin/env python3
"""
Test Upload Functionality for MonoX
Simple test to verify uploads work correctly.
"""

import os
import time
from pathlib import Path
from monox_hybrid_auth import MonoXHybridAuth

def test_upload():
    """Test upload functionality."""
    print("🧪 Testing MonoX Upload Functionality")
    print("=" * 40)
    
    # Setup authentication
    auth = MonoXHybridAuth()
    if not auth.setup_authentication():
        print("❌ Authentication setup failed")
        return False
    
    print(f"✅ Authentication method: {auth.auth_method}")
    
    # Create test files
    test_files = []
    
    # Test sample image
    sample_file = Path('test_sample.png')
    with open(sample_file, 'w') as f:
        f.write("PNG test data")
    test_files.append(sample_file)
    
    # Test checkpoint
    checkpoint_file = Path('test_checkpoint.pth')
    with open(checkpoint_file, 'w') as f:
        f.write("PyTorch checkpoint test data")
    test_files.append(checkpoint_file)
    
    # Test log
    log_file = Path('test_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Test log entry\\nTimestamp: {time.time()}\\n")
    test_files.append(log_file)
    
    print(f"\\n📁 Created {len(test_files)} test files")
    
    # Test uploads
    success_count = 0
    for test_file in test_files:
        print(f"\\n📤 Uploading {test_file.name}...")
        if auth.upload_file(str(test_file)):
            success_count += 1
            print(f"✅ {test_file.name} uploaded successfully")
        else:
            print(f"❌ {test_file.name} upload failed")
    
    # Cleanup test files
    for test_file in test_files:
        test_file.unlink()
    
    print(f"\\n📊 Upload Results: {success_count}/{len(test_files)} successful")
    
    if success_count == len(test_files):
        print("\\n🎉 All uploads successful!")
        print("\\n📋 Ready for MonoX training with automatic uploads")
        print("🔗 Check repository: https://huggingface.co/lukua/monox-model")
        return True
    else:
        print("\\n❌ Some uploads failed")
        return False

if __name__ == "__main__":
    test_upload()