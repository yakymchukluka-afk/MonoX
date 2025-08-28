#!/usr/bin/env python3
"""
Cleanup Old Scripts
==================

Removes the many redundant/broken setup scripts and keeps only the working ones.
"""

import os
import glob
from pathlib import Path

def cleanup_old_scripts():
    """Remove old, redundant setup scripts"""
    
    # Scripts to keep (the working ones)
    scripts_to_keep = {
        'colab_environment_setup.py',
        'colab_training_launcher.py', 
        'colab_debug_checklist.py',
        'colab_gpu_monitor.py',
        'example_colab_cells.py',
        'cleanup_old_scripts.py'
    }
    
    # Find all Python scripts in current directory
    all_scripts = glob.glob('*.py')
    
    scripts_to_remove = []
    for script in all_scripts:
        if script not in scripts_to_keep:
            # Check if it's one of the many old setup scripts
            if any(keyword in script.lower() for keyword in [
                'colab_', 'nuclear', 'ultimate', 'victory', 'absolute',
                'final', 'complete', 'perfect', 'emergency', 'force'
            ]):
                scripts_to_remove.append(script)
    
    print("🧹 Cleaning up old setup scripts...")
    print("=" * 40)
    
    if scripts_to_remove:
        print(f"Found {len(scripts_to_remove)} old scripts to remove:")
        for script in scripts_to_remove:
            print(f"  - {script}")
        
        # Ask for confirmation
        response = input(f"\nRemove these {len(scripts_to_remove)} scripts? (y/N): ")
        
        if response.lower() == 'y':
            removed_count = 0
            for script in scripts_to_remove:
                try:
                    os.remove(script)
                    print(f"✅ Removed {script}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {script}: {e}")
            
            print(f"\n🎉 Cleanup complete! Removed {removed_count} old scripts.")
        else:
            print("🚫 Cleanup cancelled.")
    else:
        print("✅ No old scripts found to clean up.")
    
    print(f"\n📋 Remaining working scripts:")
    for script in sorted(scripts_to_keep):
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} (missing)")

def cleanup_old_markdown():
    """Clean up old markdown files"""
    
    markdown_files_to_keep = {
        'COLAB_SETUP_GUIDE.md',
        'README.md'
    }
    
    all_markdown = glob.glob('*.md')
    markdown_to_remove = []
    
    for md_file in all_markdown:
        if md_file not in markdown_files_to_keep:
            if any(keyword in md_file.upper() for keyword in [
                'COLAB_', 'NUCLEAR', 'SUCCESS', 'TROUBLESHOOTING', 'DEBUG'
            ]):
                markdown_to_remove.append(md_file)
    
    if markdown_to_remove:
        print(f"\n📄 Found {len(markdown_to_remove)} old markdown files:")
        for md_file in markdown_to_remove:
            print(f"  - {md_file}")
        
        response = input(f"\nRemove these {len(markdown_to_remove)} files? (y/N): ")
        
        if response.lower() == 'y':
            for md_file in markdown_to_remove:
                try:
                    os.remove(md_file)
                    print(f"✅ Removed {md_file}")
                except Exception as e:
                    print(f"❌ Failed to remove {md_file}: {e}")

def main():
    """Main cleanup function"""
    print("🧹 MonoX Script Cleanup Utility")
    print("=" * 40)
    
    # Cleanup Python scripts
    cleanup_old_scripts()
    
    # Cleanup markdown files
    cleanup_old_markdown()
    
    print("\n📋 Final working file structure:")
    print("=" * 40)
    
    working_files = [
        'colab_environment_setup.py',
        'colab_training_launcher.py',
        'colab_debug_checklist.py', 
        'colab_gpu_monitor.py',
        'example_colab_cells.py',
        'COLAB_SETUP_GUIDE.md',
        'requirements_colab.txt'
    ]
    
    for file in working_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} (missing)")
    
    print("\n🎯 These are the only files you need for working Colab setup!")

if __name__ == "__main__":
    main()