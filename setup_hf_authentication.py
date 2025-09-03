#!/usr/bin/env python3
"""
HuggingFace Authentication Setup for MonoX
==========================================

Sets up HuggingFace authentication to access lukua/monox-dataset.
This script helps configure authentication in the HF Space environment.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import login, HfApi, HfFolder
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def check_current_auth():
    """Check current authentication status."""
    print("🔍 Checking current HuggingFace authentication...")
    
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"✅ Already authenticated as: {whoami['name']}")
        return True, whoami
    except Exception as e:
        print(f"❌ Not authenticated: {e}")
        return False, None


def setup_authentication():
    """Set up HuggingFace authentication."""
    print("🔑 Setting up HuggingFace authentication...")
    
    # Check for token in environment variables
    token_sources = [
        ('HF_TOKEN', os.environ.get('HF_TOKEN')),
        ('HUGGINGFACE_HUB_TOKEN', os.environ.get('HUGGINGFACE_HUB_TOKEN')),
        ('HF_HUB_TOKEN', os.environ.get('HF_HUB_TOKEN')),
        ('HUGGINGFACE_TOKEN', os.environ.get('HUGGINGFACE_TOKEN'))
    ]
    
    for source_name, token in token_sources:
        if token:
            print(f"🔑 Found token in {source_name}")
            try:
                login(token=token)
                print("✅ Authentication successful!")
                return True
            except Exception as e:
                print(f"❌ Authentication failed with {source_name}: {e}")
    
    print("❌ No valid tokens found in environment variables")
    print("\n💡 To set up authentication:")
    print("1. Get your HF token from: https://huggingface.co/settings/tokens")
    print("2. In your HF Space settings, add it as a secret:")
    print("   - Name: HF_TOKEN")
    print("   - Value: your_token_here")
    print("3. Restart the Space")
    
    return False


def test_dataset_access():
    """Test access to lukua/monox-dataset."""
    print("\n🧪 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to access the dataset
        print("🔗 Attempting to load lukua/monox-dataset...")
        dataset = load_dataset("lukua/monox-dataset", split="train[:1]", use_auth_token=True)
        
        print("✅ Dataset access successful!")
        print(f"📊 Sample count: {len(dataset)}")
        if len(dataset) > 0:
            print(f"📝 Sample keys: {list(dataset[0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("🔒 Authentication issue - check HF token")
        elif "not found" in str(e).lower():
            print("📝 Dataset not found - check repository name")
        else:
            print(f"🔧 Other error: {str(e)}")
        
        return False


def main():
    """Main authentication setup function."""
    print("🎨 MonoX HuggingFace Authentication Setup")
    print("=" * 50)
    
    # Check current authentication
    is_auth, user_info = check_current_auth()
    
    if not is_auth:
        # Try to set up authentication
        if setup_authentication():
            is_auth, user_info = check_current_auth()
    
    if is_auth:
        print(f"\n✅ Authentication confirmed: {user_info['name']}")
        
        # Test dataset access
        if test_dataset_access():
            print("\n🎉 COMPLETE: HuggingFace authentication and dataset access working!")
            print("🚀 MonoX is ready for training with lukua/monox-dataset")
        else:
            print("\n❌ Dataset access failed despite authentication")
            print("🔍 Check dataset permissions and repository name")
            sys.exit(1)
    else:
        print("\n❌ Authentication setup failed")
        print("🔧 Manual setup required - see instructions above")
        sys.exit(1)


if __name__ == "__main__":
    main()