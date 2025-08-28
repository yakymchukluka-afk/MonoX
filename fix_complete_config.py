#!/usr/bin/env python3
"""
Complete Configuration Fix for StyleGAN-V
=========================================

This adds all the missing fields that StyleGAN-V expects.
"""

import os

def create_complete_config():
    """Create a complete configuration that matches StyleGAN-V expectations"""
    print("🔧 Creating complete configuration...")
    
    os.chdir("/content/MonoX")
    
    # Complete config with all required fields
    complete_config = """# Complete MonoX + StyleGAN-V Configuration
defaults:
  - dataset: base
  - training: base
  - visualizer: base
  - _self_

# Environment configuration (all required fields)
env:
  project_path: ${hydra:runtime.cwd}
  experiment_name: monox_stylegan_v
  run_name: ${now:%Y-%m-%d_%H-%M-%S}
  before_train_commands: []  # Empty list of commands to run before training
  after_train_commands: []   # Empty list of commands to run after training

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
        f.write(complete_config)
    
    print("✅ Complete config created")

def main():
    """Fix configuration and restart"""
    print("🔧 Complete Configuration Fix")
    print("=" * 40)
    
    create_complete_config()
    
    print("\n🎉 Configuration completely fixed!")
    print("✅ Added all missing env fields")
    print("✅ before_train_commands: []")
    print("✅ after_train_commands: []")
    
if __name__ == "__main__":
    main()

# Run the fix
main()