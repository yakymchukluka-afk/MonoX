#!/usr/bin/env python3
"""
Simple launcher script that ensures proper Python path for MonoX training
"""
import os
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located (should be MonoX root)
    repo_root = Path(__file__).resolve().parent
    
    # Add the repo root to Python path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Change to the repo directory
    os.chdir(repo_root)
    
    print(f"🚀 MonoX Training Launcher")
    print(f"Repository root: {repo_root}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path includes: {str(repo_root) in sys.path}")
    
    try:
        # Import and run the src.infra.launch module
        from src.infra.launch import main as launch_main
        print("✅ Successfully imported src.infra.launch")
        launch_main()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n🔧 Fallback: Try running with train.py instead:")
        print("python3 train.py exp_suffix=monox dataset.path=/path/to/dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()