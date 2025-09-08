#!/usr/bin/env python3
"""
GPU Configuration Script for RunPod
Automatically configures training parameters based on available GPU
"""

import torch
import yaml
from pathlib import Path

def detect_gpu_type():
    """Detect GPU type and return optimal configuration."""
    if not torch.cuda.is_available():
        return None
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🔍 Detected GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f}GB")
    
    # GPU-specific configurations
    gpu_configs = {
        't4': {
            'batch_size': 4,
            'gradient_accumulation': 2,
            'mixed_precision': True,
            'learning_rate': 2e-4
        },
        'v100': {
            'batch_size': 8,
            'gradient_accumulation': 1,
            'mixed_precision': True,
            'learning_rate': 2e-4
        },
        'a100': {
            'batch_size': 12,
            'gradient_accumulation': 1,
            'mixed_precision': True,
            'learning_rate': 2e-4
        },
        'rtx 4090': {
            'batch_size': 6,
            'gradient_accumulation': 1,
            'mixed_precision': True,
            'learning_rate': 2e-4
        }
    }
    
    # Find matching configuration
    for gpu_type, config in gpu_configs.items():
        if gpu_type in gpu_name:
            print(f"✅ Matched configuration for {gpu_type}")
            return config
    
    # Default configuration for unknown GPU
    print("⚠️ Unknown GPU, using default configuration")
    return {
        'batch_size': 4,
        'gradient_accumulation': 2,
        'mixed_precision': True,
        'learning_rate': 2e-4
    }

def update_training_config(gpu_config):
    """Update training configuration with GPU-specific settings."""
    config_path = Path("runpod_config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update training configuration
    if 'training' not in config:
        config['training'] = {}
    
    config['training'].update({
        'batch_size': gpu_config['batch_size'],
        'gradient_accumulation': gpu_config['gradient_accumulation'],
        'mixed_precision': gpu_config['mixed_precision'],
        'learning_rate': gpu_config['learning_rate']
    })
    
    # Save updated configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Updated configuration:")
    print(f"   Batch Size: {gpu_config['batch_size']}")
    print(f"   Gradient Accumulation: {gpu_config['gradient_accumulation']}")
    print(f"   Mixed Precision: {gpu_config['mixed_precision']}")
    print(f"   Learning Rate: {gpu_config['learning_rate']}")

def main():
    """Main configuration function."""
    print("🔧 GPU Configuration for RunPod")
    print("=" * 40)
    
    gpu_config = detect_gpu_type()
    if gpu_config:
        update_training_config(gpu_config)
        print("\n✅ GPU configuration complete!")
    else:
        print("\n❌ No GPU detected, using CPU configuration")

if __name__ == "__main__":
    main()