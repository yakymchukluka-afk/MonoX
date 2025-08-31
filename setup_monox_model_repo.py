#!/usr/bin/env python3
"""
🚀 Setup MonoX Model Repository Structure
========================================

This script sets up the complete folder structure and documentation
for the lukua/monox model repository.
"""

import os
import subprocess
import shutil
from pathlib import Path

def setup_local_structure():
    """Set up the local model repository structure."""
    print("📁 Setting up MonoX model repository structure...")
    
    base_dir = Path("/tmp/monox_model_upload")
    base_dir.mkdir(exist_ok=True)
    
    # Copy the prepared structure
    source_dir = Path("/tmp/monox_migration/monox-model-repo")
    if source_dir.exists():
        # Copy all files
        for item in source_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, base_dir / item.name)
            else:
                shutil.copytree(item, base_dir / item.name, dirs_exist_ok=True)
        
        print(f"✅ Repository structure copied to {base_dir}")
        return base_dir
    else:
        print("❌ Source structure not found")
        return None

def create_upload_script():
    """Create a script to upload to Hugging Face."""
    upload_script = '''#!/bin/bash
# MonoX Model Repository Upload Script

echo "🚀 Uploading MonoX Model Repository Structure"
echo "============================================="

# Check if authenticated
if ! hf auth whoami > /dev/null 2>&1; then
    echo "❌ Not authenticated with Hugging Face"
    echo "Please run: hf auth login"
    exit 1
fi

echo "✅ Hugging Face authentication verified"

# Upload the complete structure
echo "📁 Uploading repository structure..."
hf upload lukua/monox /tmp/monox_model_upload --repo-type=model

echo "✅ Upload complete!"
echo "🔗 Repository: https://huggingface.co/lukua/monox"
echo ""
echo "📂 Uploaded structure:"
echo "  ├── README.md (comprehensive documentation)"
echo "  ├── config.json (model configuration)"
echo "  ├── .gitattributes (LFS settings)"
echo "  ├── checkpoints/ (for model weights)"
echo "  ├── logs/ (for training logs)"
echo "  ├── previews/ (for training samples)"
echo "  ├── samples/ (for generated art)"
echo "  └── latent_walks/ (for latent explorations)"
echo ""
echo "🎯 Ready for training progress uploads!"
'''
    
    script_path = "/workspace/upload_monox_structure.sh"
    with open(script_path, "w") as f:
        f.write(upload_script)
    
    os.chmod(script_path, 0o755)
    print(f"✅ Upload script created: {script_path}")
    return script_path

def create_python_upload_script():
    """Create a Python script for uploading."""
    python_script = '''#!/usr/bin/env python3
"""Upload MonoX structure to Hugging Face using Python API."""

import os
from huggingface_hub import HfApi, upload_folder

def upload_structure():
    """Upload the MonoX repository structure."""
    print("🚀 Uploading MonoX Model Repository Structure")
    print("=" * 50)
    
    try:
        api = HfApi()
        
        # Upload the entire structure
        print("📁 Uploading complete structure...")
        api.upload_folder(
            folder_path="/tmp/monox_model_upload",
            repo_id="lukua/monox",
            repo_type="model"
        )
        
        print("✅ Upload successful!")
        print("🔗 Repository: https://huggingface.co/lukua/monox")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("💡 Try using the CLI method instead:")
        print("   ./upload_monox_structure.sh")

if __name__ == "__main__":
    upload_structure()
'''
    
    script_path = "/workspace/upload_monox_structure.py"
    with open(script_path, "w") as f:
        f.write(python_script)
    
    os.chmod(script_path, 0o755)
    print(f"✅ Python upload script created: {script_path}")
    return script_path

def main():
    """Main setup function."""
    print("🎨 MonoX Model Repository Setup")
    print("=" * 40)
    
    # Setup local structure
    upload_dir = setup_local_structure()
    if not upload_dir:
        return False
    
    # Create upload scripts
    bash_script = create_upload_script()
    python_script = create_python_upload_script()
    
    print("\n🎉 MonoX model repository structure ready!")
    print("\n📋 Next steps:")
    print("1. Upload structure to Hugging Face:")
    print(f"   bash {bash_script}")
    print(f"   OR python {python_script}")
    print("")
    print("2. Your lukua/monox repository will have:")
    print("   📁 Organized folder structure")
    print("   📖 Comprehensive documentation") 
    print("   🔧 Ready for training integration")
    print("")
    print("3. The MonoX Space will then automatically:")
    print("   💾 Save checkpoints to lukua/monox/checkpoints/")
    print("   📊 Save logs to lukua/monox/logs/")
    print("   🎨 Save samples to lukua/monox/previews/")
    
    return True

if __name__ == "__main__":
    main()
'''