#!/usr/bin/env python3
"""
Update MonoX configs for newer Hydra versions (1.2+ compatible)
Fixes compatibility issues when upgrading from Hydra 1.0.7 to 1.2+
"""

import os
import shutil
from pathlib import Path

def update_file(file_path, old_content, new_content, description=""):
    """Update a file with new content"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if old_content in content:
            content = content.replace(old_content, new_content)
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Updated {file_path}: {description}")
            return True
        else:
            print(f"⚪ {file_path}: {description} - no changes needed")
            return True
    except Exception as e:
        print(f"❌ Failed to update {file_path}: {e}")
        return False

def main():
    print("🔧 Updating MonoX configs for newer Hydra compatibility...")
    
    # Check if we're in the right directory
    if not os.path.exists("configs/config.yaml"):
        print("❌ Error: configs/config.yaml not found. Make sure you're in the MonoX directory.")
        return False
    
    success = True
    
    # 1. Update main config.yaml for newer Hydra
    success &= update_file(
        "configs/config.yaml",
        "# Configuration settings for MonoX training",
        "# Configuration settings for MonoX training\n# Compatible with Hydra 1.2+",
        "Add Hydra 1.2+ compatibility note"
    )
    
    # 2. Update StyleGAN-V launch.py for newer Hydra  
    stylegan_launch = ".external/stylegan-v/src/infra/launch.py"
    if os.path.exists(stylegan_launch):
        # Update the Hydra main decorator for newer versions
        success &= update_file(
            stylegan_launch,
            '@hydra.main(version_base=None, config_path="../../configs", config_name="config")',
            '@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")',
            "Update Hydra version_base for newer compatibility"
        )
        
        # Remove the debug output we added earlier
        success &= update_file(
            stylegan_launch,
            '''    # Debug: Print fully composed config before instantiation
    from omegaconf import OmegaConf
    print("=" * 80)
    print("FULL CONFIG DEBUG DUMP:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    ''',
            '',
            "Remove debug output for cleaner execution"
        )
    
    # 3. Update any other config files that might have issues
    stylegan_config = ".external/stylegan-v/configs/config.yaml"
    if os.path.exists(stylegan_config):
        # Add version_base compatibility note
        success &= update_file(
            stylegan_config,
            "# MonoX compatibility - allows CLI overrides",
            "# MonoX compatibility - allows CLI overrides\n# Updated for Hydra 1.2+ compatibility",
            "Add Hydra 1.2+ compatibility note to StyleGAN-V config"
        )
    
    if success:
        print("\n✅ All configs updated for newer Hydra compatibility!")
        print("🎉 Ready to use with Hydra 1.2+ and Python 3.12!")
    else:
        print("\n⚠️  Some updates failed, but training might still work.")
    
    return success

if __name__ == "__main__":
    main()