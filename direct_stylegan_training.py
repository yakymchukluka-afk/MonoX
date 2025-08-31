#!/usr/bin/env python3
"""
Direct StyleGAN-V Training for MonoX
Bypasses the complex infra layer and calls StyleGAN-V training directly.
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

def setup_environment():
    """Setup environment for direct StyleGAN-V training."""
    env_vars = {
        "PYTHONPATH": "/workspace/.external/stylegan-v:/workspace",
        "PYTHONUNBUFFERED": "1",
        # HF_TOKEN should be set via environment variable or Space secret
        "PATH": "/home/ubuntu/.local/bin:" + os.environ.get("PATH", ""),
        "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value

def upload_to_hf_repo(file_path: str, repo_id: str = "lukua/monox") -> bool:
    """Upload file to HF model repository."""
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
            token=os.environ.get("HF_TOKEN"),
            repo_type="model"
        )
        
        logger.info(f"✅ Uploaded {file_name} to {repo_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload failed for {file_path}: {e}")
        return False

def monitor_outputs():
    """Monitor training outputs and upload to HF."""
    uploaded_files = set()
    
    while True:
        try:
            # Check for new files in output directory
            output_dir = Path("/workspace/training_output")
            if output_dir.exists():
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file() and str(file_path) not in uploaded_files:
                        if file_path.suffix in ['.pkl', '.png', '.log', '.json']:
                            if upload_to_hf_repo(str(file_path)):
                                uploaded_files.add(str(file_path))
            
            time.sleep(30)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(60)

def start_direct_training():
    """Start training using direct StyleGAN-V call."""
    
    setup_environment()
    
    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    except ImportError:
        gpu_available = False
        gpu_count = 0
    
    logger.info(f"System: GPU Available={gpu_available}, Count={gpu_count}")
    
    # Prepare output directory
    output_dir = "/workspace/training_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training parameters for direct StyleGAN-V call
    training_args = [
        "--outdir", output_dir,
        "--data", "/workspace/dataset",
        "--gpus", str(max(gpu_count, 1)),  # Use at least 1 for CPU
        "--cfg", "auto",
        "--kimg", "100",  # Very short test training
        "--snap", "25",   # Checkpoint every 25 kimg
        "--batch", "4" if not gpu_available else "16",
        "--mirror", "true",
        "--aug", "ada",
        "--metrics", "none"  # Disable metrics for faster training
    ]
    
    if not gpu_available:
        training_args.extend(["--fp32", "true"])  # Use FP32 for CPU
    
    cmd = [sys.executable, "src/train.py"] + training_args
    
    logger.info("🚀 Starting Direct StyleGAN-V Training")
    logger.info("=" * 60)
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Dataset: /workspace/dataset")
    logger.info(f"GPU Count: {gpu_count}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)
    
    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_outputs, daemon=True)
    monitor_thread.start()
    
    try:
        # Change to StyleGAN-V directory
        os.chdir("/workspace/.external/stylegan-v")
        
        # Start training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy()
        )
        
        # Stream output
        log_file_path = "/workspace/logs/direct_training.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        with open(log_file_path, "w") as log_file:
            for line in process.stdout:
                timestamp = time.strftime("%H:%M:%S")
                formatted_line = f"[{timestamp}] {line.rstrip()}"
                
                print(formatted_line)
                log_file.write(formatted_line + "\\n")
                log_file.flush()
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("🎉 Direct training completed!")
            
            # Upload final log
            upload_to_hf_repo(log_file_path)
            
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
    """Main execution."""
    print("🎨 MonoX Direct StyleGAN-V Training")
    print("🎯 Goal: Fresh training from scratch with 1024px dataset")
    print("=" * 70)
    
    # Validate dataset
    dataset_path = Path("/workspace/dataset")
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    
    print(f"📊 Dataset: {len(image_files)} images at 1024x1024")
    print(f"📁 Output: /workspace/training_output")
    print(f"📤 Upload Target: lukua/monox model repo")
    
    if len(image_files) == 0:
        print("❌ No training images found!")
        return 1
    
    # Start training
    success = start_direct_training()
    
    if success:
        print("\\n🎉 Training completed successfully!")
        print("📋 Next steps:")
        print("  1. Check lukua/monox for uploaded outputs")
        print("  2. Review training logs")
        print("  3. Test generated samples")
        return 0
    else:
        print("\\n❌ Training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())