#!/usr/bin/env python3
"""
Pre-build script for MonoX training
Handles setup that needs to happen before the main application starts.
"""

import os
import sys
from pathlib import Path

def setup_git_config():
    """Setup git configuration without global commands."""
    try:
        # Create .gitconfig in current directory
        gitconfig_content = """[user]
    email = lukua@users.noreply.huggingface.co
    name = lukua
[safe]
    directory = /app
    directory = /workspace
"""
        
        # Write to multiple possible locations
        locations = ["/app/.gitconfig", "./.gitconfig", os.path.expanduser("~/.gitconfig")]
        
        for location in locations:
            try:
                os.makedirs(os.path.dirname(location), exist_ok=True)
                with open(location, "w") as f:
                    f.write(gitconfig_content)
                print(f"✅ Git config created: {location}")
            except Exception as e:
                print(f"⚠️ Could not create {location}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Git config setup failed: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "/app/logs",
        "/app/checkpoints",
        "/app/previews", 
        "/app/training_output",
        "./logs",
        "./checkpoints",
        "./previews",
        "./training_output"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
        except Exception as e:
            print(f"⚠️ Could not create {directory}: {e}")

def main():
    """Main pre-build setup."""
    print("🔧 MonoX Pre-build Setup")
    print("=" * 30)
    
    # Setup git config
    setup_git_config()
    
    # Create directories
    create_directories()
    
    print("✅ Pre-build setup completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())