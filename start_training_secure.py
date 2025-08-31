#!/usr/bin/env python3
"""
Secure MonoX Training Starter
Starts fresh training with secure authentication - NO TOKENS IN CODE!
"""

import os
import sys
import subprocess
import time
import yaml
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_secure_environment():
    """Validate that authentication is properly configured."""
    
    # Check if HF_TOKEN is available
    hf_token = os.environ.get('HF_TOKEN')
    
    if not hf_token:
        logger.error("❌ HF_TOKEN not found in environment!")
        logger.info("💡 To fix this:")
        logger.info("  1. For HF Spaces: Add HF_TOKEN to Repository secrets")
        logger.info("  2. For Dev Mode: export HF_TOKEN=your_token")
        logger.info("  3. Never hardcode tokens in files!")
        return False
    
    if not hf_token.startswith('hf_'):
        logger.error("❌ Invalid HF token format!")
        return False
    
    # Test authentication
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=hf_token)
        logger.info(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

def setup_training_environment():
    """Setup environment for training (no tokens)."""
    
    env_vars = {
        "PYTHONPATH": "/workspace/.external/stylegan-v:/workspace",
        "DATASET_DIR": "/workspace/dataset",
        "LOGS_DIR": "/workspace/logs",
        "CKPT_DIR": "/workspace/checkpoints",
        "PREVIEWS_DIR": "/workspace/previews",
        "PYTHONUNBUFFERED": "1",
        "PATH": "/home/ubuntu/.local/bin:" + os.environ.get("PATH", "")
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Create directories
    for directory in ["/workspace/logs", "/workspace/checkpoints", "/workspace/previews", "/workspace/training_output"]:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("✅ Environment configured (securely)")

def create_stylegan_config():
    """Create StyleGAN-V experiment configuration."""
    
    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    except ImportError:
        gpu_available = False
        gpu_count = 0
    
    config = {
        "data": "/workspace/dataset",
        "outdir": "/workspace/training_output", 
        "cfg": "auto",
        "gpus": max(gpu_count, 1),
        "kimg": 1000,  # Fresh training duration
        "snap": 50,    # Checkpoint every 5 epochs
        "batch_size": 4 if not gpu_available else 16,
        "resolution": 1024,
        "fp32": not gpu_available,
        "aug": "ada",
        "mirror": True,
        "metrics": [],
        "seed": 42,
        "num_workers": 2
    }
    
    # Save to StyleGAN-V directory
    config_path = "/workspace/.external/stylegan-v/experiment_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"✅ StyleGAN-V config created: {config_path}")
    return config

def start_fresh_training():
    """Start fresh training from scratch."""
    
    logger.info("🎨 Starting MonoX Fresh Training")
    logger.info("=" * 60)
    
    # Validate security first
    if not validate_secure_environment():
        return False
    
    # Setup environment
    setup_training_environment()
    
    # Create StyleGAN config
    config = create_stylegan_config()
    
    # Validate dataset
    dataset_path = Path("/workspace/dataset")
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    
    if len(image_files) == 0:
        logger.error("❌ No training images found!")
        return False
    
    logger.info(f"📊 Dataset: {len(image_files)} images at {config['resolution']}x{config['resolution']}")
    logger.info(f"🎯 Training: {config['kimg']} kimg with checkpoints every {config['snap']} kimg")
    logger.info(f"🖥️ Hardware: {config['gpus']} GPUs, batch size {config['batch_size']}")
    
    # Start training
    cmd = [sys.executable, "src/train.py"]
    
    try:
        # Change to StyleGAN-V directory
        original_cwd = os.getcwd()
        os.chdir("/workspace/.external/stylegan-v")
        
        logger.info(f"🚀 Starting training process...")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy()
        )
        
        # Stream output
        log_path = "/workspace/logs/fresh_training_secure.log"
        with open(log_path, "w") as log_file:
            log_file.write(f"MonoX Fresh Training Started (Secure)\\n")
            log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            log_file.write("=" * 80 + "\\n")
            
            for line in process.stdout:
                timestamp = time.strftime("%H:%M:%S")
                formatted_line = f"[{timestamp}] {line.rstrip()}"
                
                print(formatted_line)
                log_file.write(formatted_line + "\\n")
                log_file.flush()
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("🎉 Training completed successfully!")
            return True
        else:
            logger.error(f"❌ Training failed: {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Training execution failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    """Main function - completely secure."""
    print("🔒 MonoX Secure Fresh Training")
    print("🎯 Fresh training from scratch with 1024px dataset")
    print("=" * 60)
    
    # Validate authentication without exposing tokens
    if not validate_secure_environment():
        print("\\n❌ Authentication not configured!")
        print("💡 Set HF_TOKEN as environment variable or Space secret")
        return 1
    
    print("✅ Secure authentication validated")
    
    # Start training
    success = start_fresh_training()
    
    if success:
        print("\\n🎉 Fresh training completed successfully!")
        print("📁 Check lukua/monox model repo for outputs")
        return 0
    else:
        print("\\n❌ Training failed - check logs")
        return 1

if __name__ == "__main__":
    sys.exit(main())