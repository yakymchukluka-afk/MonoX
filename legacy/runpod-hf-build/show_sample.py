#!/usr/bin/env python3
"""
Display training samples as base64 for viewing in terminal
"""

import base64
import os
import sys
from PIL import Image
import io

def image_to_base64_preview(image_path, max_size=(200, 200)):
    """Convert image to base64 string for preview."""
    try:
        with Image.open(image_path) as img:
            # Resize for preview
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
    except Exception as e:
        return f"Error: {e}"

def show_sample_info(sample_path):
    """Show detailed sample information."""
    if not os.path.exists(sample_path):
        print(f"❌ Sample not found: {sample_path}")
        return
    
    # File info
    stat = os.stat(sample_path)
    size_mb = stat.st_size / (1024 * 1024)
    
    print(f"🖼️  Sample: {os.path.basename(sample_path)}")
    print(f"   📏 Size: {size_mb:.2f} MB")
    print(f"   📅 Modified: {os.path.getctime(sample_path)}")
    print(f"   📍 Path: {sample_path}")
    
    # Try to get image dimensions
    try:
        with Image.open(sample_path) as img:
            print(f"   🔍 Dimensions: {img.size[0]}x{img.size[1]} pixels")
            print(f"   🎨 Mode: {img.mode}")
    except Exception as e:
        print(f"   ⚠️  Could not read image info: {e}")
    
    print(f"\n📱 To view this sample:")
    print(f"   1. Download: wget {sample_path}")
    print(f"   2. Copy path: {sample_path}")
    print(f"   3. Use image viewer in your environment")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_sample_info(sys.argv[1])
    else:
        # Show latest sample
        preview_dir = "previews"
        if os.path.exists(preview_dir):
            samples = sorted([f for f in os.listdir(preview_dir) if f.endswith('.png')])
            if samples:
                latest = os.path.join(preview_dir, samples[-1])
                show_sample_info(latest)
            else:
                print("No samples found")
        else:
            print("Preview directory not found")