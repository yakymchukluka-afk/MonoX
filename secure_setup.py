#!/usr/bin/env python3
"""
Secure Setup Script - NO TOKENS IN CODE
This script helps you set up MonoX training securely.
"""

import os
from pathlib import Path

def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking Environment Configuration")
    print("=" * 40)
    
    # Check for HF token
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("✅ HF_TOKEN found in environment")
        print(f"   Token: {hf_token[:8]}...{hf_token[-4:]}")
        return True
    else:
        print("❌ HF_TOKEN not found in environment")
        print("📝 Please add HF_TOKEN to your HF Space secrets")
        return False

def test_authentication():
    """Test HF authentication."""
    print("\n🧪 Testing Authentication...")
    
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

def create_env_template():
    """Create .env template file."""
    print("\n📝 Creating .env template...")
    
    env_template = """# MonoX Training Environment Variables
# Copy this file to .env and add your actual values
# NEVER commit .env to version control!

# Hugging Face Token (get from https://huggingface.co/settings/tokens)
HF_TOKEN=your_hf_token_here

# SSH Private Key (if using SSH authentication)
SSH_PRIVATE_KEY=your_ssh_private_key_here
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.template")
    print("📝 Copy to .env and add your actual token")

def main():
    """Main setup function."""
    print("🚀 MonoX Secure Setup")
    print("=" * 20)
    
    # Check environment
    env_ok = check_environment()
    
    if env_ok:
        # Test authentication
        auth_ok = test_authentication()
        
        if auth_ok:
            print("\n🎉 Setup complete! Ready for training.")
            print("\n📋 Next steps:")
            print("1. Run: python3 monox_hybrid_auth.py")
            print("2. Run: python3 monox_training_with_hybrid_auth.py")
        else:
            print("\n❌ Authentication failed")
            print("Please check your HF_TOKEN")
    else:
        print("\n❌ Environment not configured")
        print("\n📋 To fix:")
        print("1. Add HF_TOKEN to HF Space secrets")
        print("2. Restart the Space")
        print("3. Run this script again")
    
    # Create template
    create_env_template()

if __name__ == "__main__":
    main()