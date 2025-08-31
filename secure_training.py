#!/usr/bin/env python3
"""
Secure MonoX Training Script
Uses environment variables for authentication - no hardcoded tokens.
"""

import os
import sys
import subprocess
import time
import json
import threading
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_authentication():
    """Check if HF authentication is properly configured."""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        logger.error("❌ HF_TOKEN not found in environment!")
        logger.info("💡 Set HF_TOKEN as environment variable or Space secret")
        return False
    
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=hf_token)
        logger.info(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

def upload_to_model_repo(file_path: str, repo_id: str = "lukua/monox") -> bool:
    """Securely upload file to HF model repository."""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        logger.error("❌ No HF token available for upload")
        return False
    
    try:
        from huggingface_hub import upload_file
        
        file_name = os.path.basename(file_path)
        if file_path.endswith(('.pkl', '.pt', '.pth', '.ckpt')):
            repo_path = f"checkpoints/{file_name}"
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            repo_path = f"previews/{file_name}"
        elif file_path.endswith('.log'):
            repo_path = f"logs/{file_name}"
        else:
            repo_path = file_name
        
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            token=hf_token,  # Use token from environment
            repo_type="model"
        )
        
        logger.info(f"✅ Uploaded {file_name} to {repo_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload failed for {file_path}: {e}")
        return False

def start_secure_training():
    """Start training with secure authentication."""
    
    # Validate authentication first
    if not check_authentication():
        return False
    
    # Training parameters
    training_config = {
        "kimg": 1000,
        "snap": 50,  # Every 5 epochs
        "resolution": 1024,
        "batch_size": 4,
        "gpus": 0  # Will auto-detect
    }
    
    logger.info("🎨 Starting Secure MonoX Training")
    logger.info(f"📊 Configuration: {training_config}")
    
    # Create experiment config for StyleGAN-V
    import yaml
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    except ImportError:
        gpu_available = False
        gpu_count = 0
    
    stylegan_config = {
        "data": "/workspace/dataset",
        "outdir": "/workspace/training_output",
        "cfg": "auto",
        "gpus": max(gpu_count, 1),
        "kimg": training_config["kimg"],
        "snap": training_config["snap"],
        "batch_size": training_config["batch_size"],
        "resolution": training_config["resolution"],
        "fp32": not gpu_available,
        "aug": "ada",
        "mirror": True,
        "metrics": [],
        "seed": 42
    }
    
    # Save config to StyleGAN-V directory
    config_path = "/workspace/.external/stylegan-v/experiment_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(stylegan_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"✅ Experiment config created: {config_path}")
    
    # Start training
    cmd = [sys.executable, "src/train.py"]
    
    try:
        os.chdir("/workspace/.external/stylegan-v")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy()
        )
        
        logger.info(f"🔥 Training started (PID: {process.pid})")
        
        # Monitor outputs in background
        def monitor_outputs():
            uploaded = set()
            while process.poll() is None:
                try:
                    output_dir = Path("/workspace/training_output")
                    if output_dir.exists():
                        for file_path in output_dir.rglob("*"):
                            if file_path.is_file() and str(file_path) not in uploaded:
                                if file_path.suffix in ['.pkl', '.png', '.log']:
                                    if upload_to_model_repo(str(file_path)):
                                        uploaded.add(str(file_path))
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_outputs, daemon=True)
        monitor_thread.start()
        
        # Stream output
        log_path = "/workspace/logs/secure_training.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "w") as log_file:
            for line in process.stdout:
                timestamp = time.strftime("%H:%M:%S")
                formatted_line = f"[{timestamp}] {line.rstrip()}"
                print(formatted_line)
                log_file.write(formatted_line + "\n")
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
        os.chdir("/workspace")

def main():
    """Main function."""
    print("🔒 MonoX Secure Training Setup")
    print("=" * 50)
    
    # Setup secure environment
    if not validate_secure_setup():
        print("❌ Secure setup validation failed!")
        return 1
    
    print("✅ Secure setup validated!")
    print("\n🚀 Starting secure training...")
    
    success = start_secure_training()
    
    if success:
        print("\n🎉 Secure training completed!")
        return 0
    else:
        print("\n❌ Training failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
