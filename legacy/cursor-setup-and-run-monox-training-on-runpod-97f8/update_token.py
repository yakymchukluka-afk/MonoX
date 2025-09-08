#!/usr/bin/env python3
"""
Update Token Script - Use new token securely
This script helps you update to the new token without exposing it in code.
"""

import os
from pathlib import Path

def update_token_securely():
    """Update to new token using environment variables."""
    print("🔒 Updating Token Securely")
    print("=" * 30)
    
    # Get token from environment variable or user input
    new_token = os.environ.get('HF_TOKEN')
    
    if not new_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("📝 Please set HF_TOKEN environment variable or add to Space secrets")
        print("🔒 NEVER put tokens directly in code!")
        return False
    
    print("✅ Using token from environment variable")
    print("⚠️  Token is secure - not exposed in code")
    
    # Set environment variable
    os.environ['HF_TOKEN'] = new_token
    
    # Create .env file
    env_file = Path('.env')
    with open(env_file, 'w') as f:
        f.write(f"# MonoX Training Environment Variables\n")
        f.write(f"# Generated automatically - DO NOT COMMIT TO GIT\n")
        f.write(f"HF_TOKEN={new_token}\n")
    
    print("✅ Token stored in .env file")
    print("✅ Environment variable set")
    
    # Test the token
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Token works! Logged in as: {user_info.get('name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 MonoX Token Update")
    print("=" * 20)
    
    if update_token_securely():
        print("\n🎉 Token updated successfully!")
        print("\n📋 Next steps:")
        print("1. Add HF_TOKEN to your HF Space secrets")
        print("2. Test authentication: python3 monox_hybrid_auth.py")
        print("3. Start training: python3 monox_training_with_hybrid_auth.py")
    else:
        print("\n❌ Token update failed")
        print("Please check the token and try again")

if __name__ == "__main__":
    main()