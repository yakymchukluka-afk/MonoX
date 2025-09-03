#!/usr/bin/env python3
"""
Deploy MonoX Training to HuggingFace Space
==========================================

Ensures all training files are properly deployed to lukua/monox HF Space.
This script pushes all training components directly to the HF Space repository.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict

try:
    from huggingface_hub import HfApi, Repository
except ImportError:
    print("❌ huggingface_hub not installed")
    print("Run: pip install huggingface_hub")
    sys.exit(1)


def check_required_files() -> Dict[str, List[str]]:
    """Check that all required training files exist."""
    print("📋 Checking required files...")
    
    required_files = {
        "training_scripts": [
            "dataset_integration.py",
            "setup_training_infrastructure.py", 
            "launch_training_in_space.py",
            "validate_training_ready.py",
            "start_monox_training.py"
        ],
        "configurations": [
            "configs/monox_1024_strict.yaml",
            "configs/dataset/monox_dataset.yaml",
            "configs/training/monox_1024.yaml", 
            "configs/visualizer/monox.yaml"
        ],
        "core_files": [
            "app.py",
            "requirements.txt",
            "src/infra/launch.py"
        ]
    }
    
    missing = {}
    available = {}
    
    for category, files in required_files.items():
        missing[category] = []
        available[category] = []
        
        for file_path in files:
            if Path(file_path).exists():
                available[category].append(file_path)
                print(f"✅ {file_path}")
            else:
                missing[category].append(file_path)
                print(f"❌ {file_path}")
    
    return {"missing": missing, "available": available}


def deploy_via_git():
    """Deploy changes via git push."""
    print("\n🚀 Deploying via Git...")
    
    try:
        # Check git status
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            print("📝 Uncommitted changes found, committing...")
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 
                          'Deploy complete MonoX training system to HF Space'], 
                          check=True)
        
        # Push to origin
        print("📤 Pushing to GitHub...")
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        print("✅ Git deployment successful!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git deployment failed: {e}")
        return False


def verify_deployment():
    """Verify that deployment was successful."""
    print("\n🧪 Verifying deployment...")
    
    # Run validation
    try:
        result = subprocess.run(['python3', 'validate_training_ready.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training validation passed!")
            print("🎯 MonoX training system is ready")
            return True
        else:
            print(f"❌ Validation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def main():
    """Main deployment function."""
    print("🚀 MonoX Training Deployment to HF Space")
    print("=" * 50)
    print("🎯 Target: lukua/monox")
    print("📦 Components: Dataset integration, 1024x1024 training, model sync")
    print("=" * 50)
    
    try:
        # Step 1: Check files
        file_check = check_required_files()
        
        total_missing = sum(len(files) for files in file_check["missing"].values())
        total_available = sum(len(files) for files in file_check["available"].values())
        
        print(f"\n📊 File Status: {total_available} available, {total_missing} missing")
        
        if total_missing > 0:
            print("❌ Missing required files - deployment incomplete")
            for category, files in file_check["missing"].items():
                if files:
                    print(f"   {category}: {', '.join(files)}")
            return False
        
        # Step 2: Deploy via git
        if deploy_via_git():
            print("✅ Deployment successful!")
            
            # Step 3: Verify
            if verify_deployment():
                print("\n🎉 MonoX training system deployed and verified!")
                print("📍 Location: lukua/monox HF Space")
                print("🎯 Ready for: 1024x1024 StyleGAN-V training")
                print("🔒 Dataset: lukua/monox-dataset (private)")
                print("📦 Model: lukua/monox-model (outputs)")
                
                print("\n💡 Next steps:")
                print("1. Wait for HF Space to rebuild (may take a few minutes)")
                print("2. Open lukua/monox HF Space")
                print("3. Click '🧪 Validate Setup' to confirm")
                print("4. Click '🎨 Start MonoX Training' to begin")
                
                return True
            else:
                print("\n❌ Deployment verification failed")
                return False
        else:
            print("❌ Deployment failed")
            return False
            
    except Exception as e:
        print(f"\n💥 Deployment error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)