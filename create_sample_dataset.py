#!/usr/bin/env python3
"""
Create sample dataset for MonoX training
"""

import os
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

def create_sample_images():
    """Create sample monotype-style images for training."""
    dataset_path = Path("/workspace/dataset")
    dataset_path.mkdir(exist_ok=True)
    
    print("🎨 Creating sample MonoX dataset...")
    
    # Create 20 sample images with different monotype-style patterns
    for i in range(20):
        # Create a 512x512 image
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create different monotype-style patterns
        if i % 4 == 0:
            # Abstract brushstrokes
            for j in range(10):
                x1 = np.random.randint(0, 400)
                y1 = np.random.randint(0, 400)
                x2 = x1 + np.random.randint(50, 150)
                y2 = y1 + np.random.randint(10, 50)
                color = (np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(0, 100))
                draw.rectangle([x1, y1, x2, y2], fill=color)
        
        elif i % 4 == 1:
            # Circular patterns
            for j in range(8):
                x = np.random.randint(50, 450)
                y = np.random.randint(50, 450)
                r = np.random.randint(20, 80)
                color = (np.random.randint(0, 150), np.random.randint(0, 150), np.random.randint(0, 150))
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        
        elif i % 4 == 2:
            # Linear patterns
            for j in range(15):
                x1 = np.random.randint(0, 512)
                y1 = np.random.randint(0, 512)
                x2 = np.random.randint(0, 512)
                y2 = np.random.randint(0, 512)
                color = (np.random.randint(0, 120), np.random.randint(0, 120), np.random.randint(0, 120))
                draw.line([x1, y1, x2, y2], fill=color, width=np.random.randint(2, 8))
        
        else:
            # Textural patterns
            for j in range(100):
                x = np.random.randint(0, 512)
                y = np.random.randint(0, 512)
                size = np.random.randint(2, 10)
                color = (np.random.randint(0, 180), np.random.randint(0, 180), np.random.randint(0, 180))
                draw.rectangle([x, y, x+size, y+size], fill=color)
        
        # Save the image
        img_path = dataset_path / f"sample_{i:03d}.png"
        img.save(img_path)
        print(f"✅ Created: {img_path.name}")
    
    print(f"🎉 Created {len(list(dataset_path.glob('*.png')))} sample images!")
    return True

if __name__ == "__main__":
    create_sample_images()