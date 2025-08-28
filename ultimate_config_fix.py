#!/usr/bin/env python3
"""
Ultimate Configuration Fix
==========================

Based on the error messages, this creates a complete config with ALL the fields
that StyleGAN-V expects in the env section.
"""

import os

def create_ultimate_config():
    """Create the ultimate complete configuration"""
    print("🔧 Creating ultimate complete configuration...")
    
    os.chdir("/content/MonoX")
    
    # Ultimate complete config with ALL possible env fields
    ultimate_config = """# Ultimate Complete MonoX + StyleGAN-V Configuration
defaults:
  - dataset: base
  - training: base
  - visualizer: base
  - _self_

# Complete environment configuration with all required fields
env:
  project_path: ${hydra:runtime.cwd}
  experiment_name: monox_stylegan_v
  run_name: ${now:%Y-%m-%d_%H-%M-%S}
  before_train_commands: []
  after_train_commands: []
  torch_extensions_dir: /tmp/torch_extensions
  results_dir: /content/MonoX/results
  checkpoints_dir: /content/MonoX/results/checkpoints
  logs_dir: /content/MonoX/results/logs
  samples_dir: /content/MonoX/results/previews

# Experiment configuration
exp_suffix: "monox"
num_gpus: 1

# Dataset configuration
dataset:
  path: /content/MonoX/dataset
  resolution: 1024
  c_dim: 0
  num_channels: 3

# Training configuration
training:
  total_kimg: 3000
  snapshot_kimg: 250
  batch_size: 4
  fp16: true
  num_gpus: 1
  log_dir: /content/MonoX/results/logs
  preview_dir: /content/MonoX/results/previews
  checkpoint_dir: /content/MonoX/results/checkpoints
  resume: ""

# Sampling configuration
sampling:
  truncation_psi: 1.0
  num_samples: 16

# Visualization configuration
visualizer:
  save_every_kimg: 50
  output_dir: /content/MonoX/results/previews
  grid_size: 4

# Hydra configuration
hydra:
  run:
    dir: /content/MonoX/results/logs
  job:
    chdir: false
  output_subdir: null
"""
    
    with open("configs/config.yaml", "w") as f:
        f.write(ultimate_config)
    
    print("✅ Ultimate config created with all env fields")

def create_torch_extensions_dir():
    """Create torch extensions directory"""
    torch_ext_dir = "/tmp/torch_extensions"
    os.makedirs(torch_ext_dir, exist_ok=True)
    print(f"✅ Created torch extensions dir: {torch_ext_dir}")

def main():
    """Ultimate configuration fix"""
    print("🔧 Ultimate Configuration Fix")
    print("=" * 50)
    
    create_ultimate_config()
    create_torch_extensions_dir()
    
    print("\n🎉 Ultimate configuration complete!")
    print("✅ Added ALL possible env fields")
    print("✅ torch_extensions_dir: /tmp/torch_extensions")
    print("✅ All directory paths configured")

if __name__ == "__main__":
    main()

# Run the ultimate fix
main()