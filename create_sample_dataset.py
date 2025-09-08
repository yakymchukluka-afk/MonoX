#!/usr/bin/env python3
"""
Create a sample dataset for testing StyleGAN2-ADA training
Generates synthetic images to test the training pipeline
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path

def create_sample_image(width, height, seed=None):
    """Create a sample image with random geometric shapes."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create image with random background color
    bg_color = tuple(np.random.randint(0, 256, 3))
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw random shapes
    num_shapes = random.randint(3, 8)
    
    for _ in range(num_shapes):
        # Random shape properties
        shape_type = random.choice(['circle', 'rectangle', 'ellipse', 'polygon'])
        color = tuple(np.random.randint(0, 256, 3))
        
        # Random position and size
        x1 = random.randint(0, width//2)
        y1 = random.randint(0, height//2)
        x2 = random.randint(x1 + 20, width)
        y2 = random.randint(y1 + 20, height)
        
        if shape_type == 'circle':
            # Draw circle
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
        elif shape_type == 'rectangle':
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
        elif shape_type == 'ellipse':
            # Draw ellipse
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
        elif shape_type == 'polygon':
            # Draw polygon
            points = []
            for _ in range(random.randint(3, 6)):
                px = random.randint(x1, x2)
                py = random.randint(y1, y2)
                points.append((px, py))
            draw.polygon(points, fill=color, outline=None)
    
    return img

def create_sample_dataset(output_dir, num_images=1000, resolution=1024):
    """Create a sample dataset for testing."""
    print(f"🎨 Creating sample dataset...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️ Number of images: {num_images}")
    print(f"📐 Resolution: {resolution}x{resolution}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate images
    for i in range(num_images):
        if i % 100 == 0:
            print(f"   Generated {i}/{num_images} images...")
        
        # Create image with unique seed
        img = create_sample_image(resolution, resolution, seed=i)
        
        # Save image
        img_path = Path(output_dir) / f"sample_{i:06d}.png"
        img.save(img_path, "PNG")
    
    print(f"✅ Sample dataset created successfully!")
    print(f"📦 Location: {output_dir}")
    print(f"🖼️ Images: {num_images}")
    
    # Show file size
    total_size = sum(f.stat().st_size for f in Path(output_dir).glob("*.png"))
    print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample dataset for testing")
    parser.add_argument("--output", "-o", default="/workspace/datasets/sample", 
                       help="Output directory for sample images")
    parser.add_argument("--num-images", "-n", type=int, default=1000,
                       help="Number of images to generate")
    parser.add_argument("--resolution", "-r", type=int, default=1024,
                       help="Image resolution")
    
    args = parser.parse_args()
    
    create_sample_dataset(args.output, args.num_images, args.resolution)

if __name__ == "__main__":
    main()