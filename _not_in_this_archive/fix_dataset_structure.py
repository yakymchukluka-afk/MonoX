#!/usr/bin/env python3
"""
🔧 MONOX DATASET STRUCTURE FIXER
=====================================
Fix StyleGAN-V dataset structure to match expected format:

EXPECTED STRUCTURE:
dataset/
├── sequence1/
│   ├── frame1.jpg
│   ├── frame2.jpg
│   └── ...
├── sequence2/
│   ├── frame1.jpg
│   ├── frame2.jpg
│   └── ...

CURRENT STRUCTURE (likely):
dataset/
├── frame1.jpg
├── frame2.jpg
└── ...

This script reorganizes flat image files into sequence directories.
"""

import os
import shutil
import glob
from pathlib import Path

def fix_dataset_structure():
    """Fix the dataset structure for StyleGAN-V training."""
    
    dataset_path = "/content/drive/MyDrive/MonoX/dataset"
    
    print("🔧 MONOX: Analyzing dataset structure...")
    print(f"📂 Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Check current structure
    all_files = os.listdir(dataset_path)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    directories = [d for d in all_files if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"📊 Found: {len(image_files)} image files, {len(directories)} directories")
    
    if len(directories) >= 1 and len(image_files) == 0:
        print("✅ Dataset already has proper directory structure!")
        # Check if subdirectories have images
        for subdir in directories[:3]:  # Check first 3 subdirs
            subdir_path = os.path.join(dataset_path, subdir)
            subdir_images = glob.glob(os.path.join(subdir_path, "*.jpg")) + \
                          glob.glob(os.path.join(subdir_path, "*.jpeg")) + \
                          glob.glob(os.path.join(subdir_path, "*.png"))
            print(f"🗂️  {subdir}/: {len(subdir_images)} images")
        return True
    
    if len(image_files) == 0:
        print("❌ No image files found in dataset!")
        return False
    
    print("🔄 FIXING: Converting flat structure to sequence directories...")
    
    # Create backup
    backup_path = dataset_path + "_backup"
    if not os.path.exists(backup_path):
        print(f"💾 Creating backup: {backup_path}")
        shutil.copytree(dataset_path, backup_path)
    
    # Group images into sequences (every N images = 1 sequence)
    images_per_sequence = 16  # StyleGAN-V typically uses 16 frames per sequence
    
    # Sort image files
    image_files.sort()
    
    sequence_count = 0
    for i in range(0, len(image_files), images_per_sequence):
        sequence_dir = os.path.join(dataset_path, f"sequence_{sequence_count:04d}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Move images to sequence directory
        batch = image_files[i:i + images_per_sequence]
        for j, image_file in enumerate(batch):
            old_path = os.path.join(dataset_path, image_file)
            new_name = f"frame_{j:04d}" + Path(image_file).suffix
            new_path = os.path.join(sequence_dir, new_name)
            
            if os.path.exists(old_path):
                shutil.move(old_path, new_path)
        
        print(f"📁 Created sequence_{sequence_count:04d}/ with {len(batch)} frames")
        sequence_count += 1
    
    print(f"✅ FIXED: Created {sequence_count} sequence directories")
    print(f"💾 Backup saved at: {backup_path}")
    return True

def verify_structure():
    """Verify the dataset structure matches StyleGAN-V expectations."""
    dataset_path = "/content/drive/MyDrive/MonoX/dataset"
    
    print("\n🔍 VERIFYING DATASET STRUCTURE:")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path missing: {dataset_path}")
        return False
    
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(subdirs) == 0:
        print("❌ No subdirectories found - StyleGAN-V needs sequence directories")
        return False
    
    print(f"✅ Found {len(subdirs)} sequence directories")
    
    # Check first few subdirectories
    for i, subdir in enumerate(subdirs[:3]):
        subdir_path = os.path.join(dataset_path, subdir)
        images = glob.glob(os.path.join(subdir_path, "*.*"))
        images = [img for img in images if Path(img).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        print(f"📁 {subdir}/: {len(images)} images")
        
        if len(images) == 0:
            print(f"⚠️  WARNING: {subdir}/ is empty!")
    
    return True

if __name__ == "__main__":
    print("🚀 MONOX DATASET STRUCTURE FIXER")
    print("=" * 50)
    
    if fix_dataset_structure():
        verify_structure()
        print("\n🎉 Dataset structure fixed! Ready for StyleGAN-V training!")
    else:
        print("\n❌ Failed to fix dataset structure")