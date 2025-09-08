#!/usr/bin/env python3
"""
StyleGAN2-ADA Sample Generation Script
Generates samples from a trained model
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def generate_samples(model_path, output_dir, num_samples=100, resolution=1024):
    """Generate samples from a trained StyleGAN2-ADA model."""
    
    print(f"🎨 Generating samples from: {model_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️ Number of samples: {num_samples}")
    print(f"📐 Resolution: {resolution}x{resolution}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Activate virtual environment
    venv_python = "/workspace/venv/bin/python"
    
    # Change to StyleGAN2-ADA directory
    stylegan_dir = "/workspace/stylegan2-ada"
    os.chdir(stylegan_dir)
    
    # Generate samples
    cmd = [
        venv_python, "generate.py",
        "--network", model_path,
        "--outdir", output_dir,
        "--seeds", "0-{}".format(num_samples-1),
        "--trunc", "1.0"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Sample generation completed!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating samples: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate samples from StyleGAN2-ADA model")
    parser.add_argument("model_path", help="Path to the trained model (.pkl file)")
    parser.add_argument("--output", "-o", default="/workspace/samples", help="Output directory for samples")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--resolution", "-r", type=int, default=1024, help="Resolution of generated images")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    # Generate samples
    success = generate_samples(
        args.model_path,
        args.output,
        args.num_samples,
        args.resolution
    )
    
    if success:
        print(f"🎉 Samples saved to: {args.output}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()