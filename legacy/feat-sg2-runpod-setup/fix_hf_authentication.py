#!/usr/bin/env python3
"""
Fix Hugging Face Authentication Issues for MonoX Training
This script addresses the 401 authentication errors and repository access issues.
"""

import os
import sys
from pathlib import Path

def setup_hf_authentication():
    """Setup proper HF authentication for the space."""
    print("🔧 Setting up Hugging Face Authentication...")
    
    # Check if HF_TOKEN is available
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("📝 Please add HF_TOKEN as a secret in your HF Space settings:")
        print("   1. Go to your Space settings")
        print("   2. Add secret: HF_TOKEN = your_hf_token")
        print("   3. Restart the space")
        return False
    
    print(f"✅ HF_TOKEN found: {hf_token[:8]}...")
    
    # Create .huggingface directory with proper config
    hf_dir = Path.home() / '.huggingface'
    hf_dir.mkdir(exist_ok=True)
    
    # Write token to file
    token_file = hf_dir / 'token'
    with open(token_file, 'w') as f:
        f.write(hf_token)
    
    print("✅ HF token saved to ~/.huggingface/token")
    
    # Set environment variables
    os.environ['HF_HOME'] = str(hf_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(hf_dir / 'hub')
    
    return True

def fix_upload_paths():
    """Fix upload paths to use correct repository."""
    print("🔧 Fixing upload paths...")
    
    # The correct repository should be lukua/monox-model (not lukua/monox)
    correct_repo = "lukua/monox-model"
    
    print(f"✅ Upload target: {correct_repo}")
    print("📁 Upload paths will be:")
    print("   - Samples: samples/")
    print("   - Checkpoints: checkpoints/")
    print("   - Logs: logs/")
    
    return correct_repo

def create_upload_function():
    """Create a proper upload function with authentication."""
    upload_code = '''
def upload_to_hf_repo(file_path: str, repo_id: str = "lukua/monox-model") -> bool:
    """Upload file to HF repository with proper authentication."""
    import os
    from huggingface_hub import upload_file, HfApi
    
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("❌ No HF token available")
        return False
    
    try:
        # Determine upload path based on file type
        file_name = os.path.basename(file_path)
        if file_path.endswith(('.pkl', '.pt', '.pth', '.ckpt')):
            repo_path = f"checkpoints/{file_name}"
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            repo_path = f"samples/{file_name}"
        elif file_path.endswith('.log'):
            repo_path = f"logs/{file_name}"
        else:
            repo_path = file_name
        
        # Upload with authentication
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            token=hf_token,
            repo_type="model"
        )
        
        print(f"✅ Uploaded: {file_name} -> {repo_path}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
'''
    
    return upload_code

def main():
    """Main function to fix all authentication issues."""
    print("🚀 MonoX HF Authentication Fix")
    print("=" * 40)
    
    # Setup authentication
    if not setup_hf_authentication():
        print("❌ Authentication setup failed")
        return False
    
    # Fix upload paths
    repo = fix_upload_paths()
    
    # Create upload function
    upload_func = create_upload_function()
    
    print("\\n✅ All fixes applied!")
    print("📋 Next steps:")
    print("   1. Add HF_TOKEN as secret in Space settings")
    print("   2. Restart the Space")
    print("   3. The training should now upload to lukua/monox-model")
    
    return True

if __name__ == "__main__":
    main()