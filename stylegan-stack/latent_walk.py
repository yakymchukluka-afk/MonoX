#!/usr/bin/env python3
"""
Latent Interpolation Generator for Mono Project

This script generates a 30-second video worth of frames (900 frames at 30 FPS)
by interpolating through 37 latent points using StyleGAN2-ADA.

Features:
- Generates 37 random latent vectors (with optional fixed seeds)
- Creates 36 transitions with 25 frames each (36 × 25 = 900 frames)
- Uses SLERP (Spherical Linear Interpolation) for smooth transitions
- Saves frames sequentially for FFmpeg compilation
- Logs latent path to JSON for reproducibility
"""

import os
import sys
import json
import uuid
import argparse
import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import PIL.Image

# Add current directory to path for StyleGAN imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import dnnlib
    import legacy
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure you have PyTorch and StyleGAN2-ADA dependencies installed.")
    sys.exit(1)


def slerp(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical Linear Interpolation between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        t: Interpolation parameter (0.0 to 1.0)
        
    Returns:
        Interpolated vector
    """
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate the angle between vectors
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    theta = np.arccos(dot)
    
    # If vectors are nearly parallel, use linear interpolation
    if np.abs(theta) < 1e-6:
        return (1.0 - t) * v1 + t * v2
    
    # SLERP formula
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * v1 + w2 * v2


def generate_latent_vectors(num_vectors: int, z_dim: int, seeds: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    Generate random latent vectors for StyleGAN2.
    
    Args:
        num_vectors: Number of vectors to generate
        z_dim: Dimensionality of latent space
        seeds: Optional list of seeds for reproducible generation
        
    Returns:
        List of latent vectors
    """
    vectors = []
    
    for i in range(num_vectors):
        if seeds and i < len(seeds):
            np.random.seed(seeds[i])
            seed = seeds[i]
        else:
            seed = np.random.randint(0, 2**32 - 1)
            np.random.seed(seed)
        
        # Generate random latent vector from standard normal distribution
        z = np.random.randn(z_dim).astype(np.float32)
        vectors.append(z)
        
        print(f"Generated latent vector {i+1}/{num_vectors} (seed: {seed})")
    
    return vectors


def interpolate_path(latent_vectors: List[np.ndarray], frames_per_transition: int) -> List[np.ndarray]:
    """
    Create interpolated path through latent vectors.
    
    Args:
        latent_vectors: List of keyframe latent vectors
        frames_per_transition: Number of frames between each pair of vectors
        
    Returns:
        List of interpolated latent vectors for all frames
    """
    interpolated_frames = []
    
    for i in range(len(latent_vectors) - 1):
        v1 = latent_vectors[i]
        v2 = latent_vectors[i + 1]
        
        print(f"Interpolating transition {i+1}/{len(latent_vectors)-1}")
        
        for j in range(frames_per_transition):
            t = j / (frames_per_transition - 1) if frames_per_transition > 1 else 0.0
            interpolated = slerp(v1, v2, t)
            interpolated_frames.append(interpolated)
    
    return interpolated_frames


def load_stylegan_generator(model_path: str, device: torch.device):
    """
    Load StyleGAN2-ADA generator from pickle file.
    
    Args:
        model_path: Path to the .pkl model file
        device: PyTorch device to load model on
        
    Returns:
        Loaded generator network
    """
    print(f"Loading StyleGAN2 generator from {model_path}")
    
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    print(f"Generator loaded successfully. Latent dim: {G.z_dim}")
    return G


def generate_frame(G, latent_vector: np.ndarray, truncation_psi: float = 1.0) -> PIL.Image.Image:
    """
    Generate a single frame from a latent vector.
    
    Args:
        G: StyleGAN2 generator
        latent_vector: Input latent vector
        truncation_psi: Truncation parameter for style mixing
        
    Returns:
        Generated PIL Image
    """
    # Convert to tensor and add batch dimension
    z = torch.from_numpy(latent_vector).unsqueeze(0).to(G.device)
    
    # Generate image
    with torch.no_grad():
        img = G(z, None, truncation_psi=truncation_psi)
        
    # Convert to PIL Image
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    
    return img


def save_latent_path(latent_vectors: List[np.ndarray], seeds: List[int], output_dir: Path):
    """
    Save the latent path configuration to JSON for reproducibility.
    
    Args:
        latent_vectors: List of keyframe latent vectors
        seeds: List of seeds used to generate vectors
        output_dir: Output directory to save JSON file
    """
    latent_data = {
        "num_keyframes": len(latent_vectors),
        "total_frames": 900,
        "frames_per_transition": 25,
        "seeds": seeds,
        "latent_vectors": [vector.tolist() for vector in latent_vectors]
    }
    
    json_path = output_dir / "latent_path.json"
    with open(json_path, 'w') as f:
        json.dump(latent_data, f, indent=2)
    
    print(f"Latent path saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate latent walk frames for Mono project")
    parser.add_argument("--model", type=str, default="/workspace/stylegan-stack/models/model.pkl",
                        help="Path to StyleGAN2 model pickle file")
    parser.add_argument("--output_dir", type=str, default="/workspace/stylegan-stack/generated_frames",
                        help="Output directory for generated frames")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Optional fixed seeds for latent vector generation")
    parser.add_argument("--truncation_psi", type=float, default=1.0,
                        help="Truncation parameter for StyleGAN2 generation")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda, cpu, or auto)")
    
    args = parser.parse_args()
    
    # Configuration
    NUM_KEYFRAMES = 37
    FRAMES_PER_TRANSITION = 25
    TOTAL_FRAMES = (NUM_KEYFRAMES - 1) * FRAMES_PER_TRANSITION  # 36 × 25 = 900
    
    print(f"🎬 Mono Latent Walk Generator")
    print(f"   Keyframes: {NUM_KEYFRAMES}")
    print(f"   Transitions: {NUM_KEYFRAMES - 1}")
    print(f"   Frames per transition: {FRAMES_PER_TRANSITION}")
    print(f"   Total frames: {TOTAL_FRAMES}")
    print(f"   Duration: 30 seconds @ 30 FPS")
    print()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found at {args.model}")
        print("Please ensure the StyleGAN2 model is saved at the specified path.")
        return 1
    
    # Load generator
    try:
        G = load_stylegan_generator(args.model, device)
    except Exception as e:
        print(f"❌ Error loading generator: {e}")
        return 1
    
    # Create unique output directory
    session_id = str(uuid.uuid4())
    output_dir = Path(args.output_dir) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Generate keyframe latent vectors
    print(f"\n🎲 Generating {NUM_KEYFRAMES} keyframe latent vectors...")
    keyframe_seeds = []
    
    if args.seeds:
        # Use provided seeds and generate additional random ones if needed
        keyframe_seeds = args.seeds[:NUM_KEYFRAMES]
        while len(keyframe_seeds) < NUM_KEYFRAMES:
            keyframe_seeds.append(np.random.randint(0, 2**32 - 1))
    else:
        # Generate completely random seeds
        keyframe_seeds = [np.random.randint(0, 2**32 - 1) for _ in range(NUM_KEYFRAMES)]
    
    latent_vectors = generate_latent_vectors(NUM_KEYFRAMES, G.z_dim, keyframe_seeds)
    
    # Create interpolated path
    print(f"\n🌊 Creating interpolated path...")
    interpolated_frames = interpolate_path(latent_vectors, FRAMES_PER_TRANSITION)
    
    print(f"Generated {len(interpolated_frames)} interpolated frames")
    
    # Save latent path for reproducibility
    save_latent_path(latent_vectors, keyframe_seeds, output_dir)
    
    # Generate all frames
    print(f"\n🎨 Generating {len(interpolated_frames)} frames...")
    
    for i, latent_vector in enumerate(interpolated_frames):
        # Generate frame
        frame = generate_frame(G, latent_vector, args.truncation_psi)
        
        # Save frame
        frame_path = output_dir / f"frame_{i:04d}.png"
        frame.save(frame_path)
        
        # Progress update
        if (i + 1) % 50 == 0 or i == len(interpolated_frames) - 1:
            progress = (i + 1) / len(interpolated_frames) * 100
            print(f"   Progress: {i+1}/{len(interpolated_frames)} frames ({progress:.1f}%)")
    
    print(f"\n✅ Complete! Generated {len(interpolated_frames)} frames")
    print(f"📁 Frames saved to: {output_dir}")
    print(f"🎥 Ready for FFmpeg compilation to 30-second video")
    print(f"🔑 Session ID: {session_id}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())