#!/usr/bin/env python3
"""
MonoX Fresh Training Script
Automated training with checkpoint saving every 5 epochs and comprehensive logging.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Setup environment
os.environ.update({
    "PYTHONPATH": "/workspace/.external/stylegan-v:/workspace",
    "DATASET_DIR": "/workspace/dataset",
    "LOGS_DIR": "/workspace/logs",
    "CKPT_DIR": "/workspace/checkpoints", 
    "PREVIEWS_DIR": "/workspace/previews",
    "PYTHONUNBUFFERED": "1",
    # HF_TOKEN should be set as environment variable or Space secret
    # Do not hardcode tokens in source files!
})

def upload_to_hf_model_repo(file_path: str, repo_path: str = "lukua/monox"):
    """Upload files to the HF model repository."""
    try:
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_path,
            token=os.environ.get("HF_TOKEN"),
            repo_type="model"
        )
        print(f"✅ Uploaded {file_path} to {repo_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {file_path}: {e}")
        return False

def monitor_training():
    """Monitor training progress and upload checkpoints/logs."""
    checkpoints_dir = Path("/workspace/checkpoints")
    logs_dir = Path("/workspace/logs")
    previews_dir = Path("/workspace/previews")
    
    uploaded_files = set()
    
    while True:
        try:
            # Check for new checkpoints
            for ckpt_file in checkpoints_dir.glob("*.pkl"):
                if str(ckpt_file) not in uploaded_files:
                    print(f"📤 New checkpoint found: {ckpt_file.name}")
                    if upload_to_hf_model_repo(str(ckpt_file)):
                        uploaded_files.add(str(ckpt_file))
            
            # Check for new preview images
            for preview_file in previews_dir.glob("*.png"):
                if str(preview_file) not in uploaded_files:
                    print(f"📤 New preview found: {preview_file.name}")
                    if upload_to_hf_model_repo(str(preview_file)):
                        uploaded_files.add(str(preview_file))
            
            # Upload latest log file
            latest_log = None
            for log_file in logs_dir.glob("**/*.log"):
                if latest_log is None or log_file.stat().st_mtime > latest_log.stat().st_mtime:
                    latest_log = log_file
            
            if latest_log and str(latest_log) not in uploaded_files:
                print(f"📤 Uploading log: {latest_log.name}")
                if upload_to_hf_model_repo(str(latest_log)):
                    uploaded_files.add(str(latest_log))
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            time.sleep(60)

def main():
    """Main training function."""
    print("🎨 Starting MonoX Fresh Training")
    print("=" * 60)
    
    # Training parameters optimized for fresh training
    training_params = {
        "total_kimg": 1000,  # Start with 1000 kimg for testing
        "resolution": 1024,
        "snapshot_kimg": 50,  # Save every 50 kimg (~5 epochs)
        "num_gpus": 1 if torch.cuda.is_available() else 0
    }
    
    print(f"Training Parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        f"dataset.path={training_params['dataset_path'] if 'dataset_path' in training_params else '/workspace/dataset'}",
        f"dataset.resolution={training_params['resolution']}",
        f"training.total_kimg={training_params['total_kimg']}",
        f"training.snapshot_kimg={training_params['snapshot_kimg']}",
        f"training.num_gpus={training_params['num_gpus']}",
        "hydra.run.dir=/workspace/logs",
        "visualizer.output_dir=/workspace/previews",
        f"visualizer.save_every_kimg={training_params['snapshot_kimg']}",
        "exp_suffix=monox_fresh_training"
    ]
    
    print(f"\nTraining Command:")
    print(" ".join(cmd))
    print("=" * 60)
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=monitor_training, daemon=True)
    monitor_thread.start()
    
    # Start training
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "/workspace/.external/stylegan-v:/workspace"
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd="/workspace"
        )
        
        # Stream output
        log_file = Path("/workspace/logs/fresh_training.log")
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, "w") as f:
            for line in process.stdout:
                print(line.strip())
                f.write(line)
                f.flush()
        
        return_code = process.wait()
        
        if return_code == 0:
            print("\n🎉 Training completed successfully!")
        else:
            print(f"\n❌ Training failed with return code: {return_code}")
        
        return return_code
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    import torch
    sys.exit(main())
