#!/usr/bin/env python3
"""
Test HF Space Authentication
Tests if the 'token' secret is working correctly.
"""

import os
from pathlib import Path

def test_space_secret():
    """Test if HF Space secret is accessible."""
    print("🧪 Testing HF Space Secret Authentication")
    print("=" * 50)
    
    # Check for the 'token' secret
    token = os.environ.get('token')
    
    if not token:
        print("❌ 'token' secret not found in environment")
        print("📝 Make sure you added the secret in HF Space settings:")
        print("   Secret name: token")
        print("   Secret value: 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")
        return False
    
    print(f"✅ 'token' secret found: {token[:8]}...{token[-4:]}")
    
    # Test authentication
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authentication successful!")
        print(f"   User: {user_info.get('name', 'Unknown')}")
        print(f"   Type: {user_info.get('type', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def test_upload():
    """Test upload functionality."""
    print("\n📤 Testing Upload Functionality...")
    
    try:
        from monox_hybrid_auth import MonoXHybridAuth
        
        auth = MonoXHybridAuth()
        if auth.setup_authentication():
            print(f"✅ Authentication method: {auth.auth_method}")
            
            # Create test file
            test_file = Path('test_space_auth.txt')
            with open(test_file, 'w') as f:
                f.write("HF Space Authentication Test\\n")
                f.write(f"Token: {os.environ.get('token', 'Not found')[:8]}...\\n")
            
            # Test upload
            if auth.upload_file(str(test_file)):
                print("✅ Upload test successful!")
                test_file.unlink()  # Cleanup
                return True
            else:
                print("❌ Upload test failed")
                return False
        else:
            print("❌ Authentication setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 HF Space Authentication Test")
    print("=" * 40)
    
    # Test secret access
    secret_ok = test_space_secret()
    
    if secret_ok:
        # Test upload
        upload_ok = test_upload()
        
        if upload_ok:
            print("\n🎉 All tests passed!")
            print("✅ HF Space authentication is working")
            print("✅ Uploads are working")
            print("✅ Ready for MonoX training")
        else:
            print("\n❌ Upload test failed")
            print("Check your repository permissions")
    else:
        print("\n❌ Secret test failed")
        print("Make sure 'token' secret is added to HF Space settings")

if __name__ == "__main__":
    main()