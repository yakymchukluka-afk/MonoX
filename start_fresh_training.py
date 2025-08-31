#!/usr/bin/env python3
"""
Start Fresh MonoX Training
Corrected script using proper StyleGAN-V parameter names and structure.
"""

import os
import sys
import subprocess
import time
import threading
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables."""
    env_vars = {
        "PYTHONPATH": "/workspace/.external/stylegan-v:/workspace",
        "DATASET_DIR": "/workspace/dataset",
        "LOGS_DIR": "/workspace/logs",
        "CKPT_DIR": "/workspace/checkpoints",
        "PREVIEWS_DIR": "/workspace/previews",
        "PYTHONUNBUFFERED": "1",
        "HF_TOKEN": "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",
        "PATH": "/home/ubuntu/.local/bin:" + os.environ.get("PATH", "")
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value

def upload_to_hf_repo(file_path: str, repo_id: str = "lukua/monox") -> bool:
    """Upload file to HF model repository."""
    try:
        from huggingface_hub import upload_file
        
        # Determine repo path based on file type
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
        
        logger.info(f"✅ Uploaded {file_name} to {repo_id}/{repo_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to upload {file_path}: {e}")
        return False

def monitor_training_outputs():
    """Monitor and upload training outputs."""
    uploaded_files = set()
    
    while True:
        try:
            # Monitor checkpoints
            for ckpt in Path("/workspace/checkpoints").glob("*.pkl"):
                if str(ckpt) not in uploaded_files:
                    if upload_to_hf_repo(str(ckpt)):
                        uploaded_files.add(str(ckpt))
            
            # Monitor previews
            for preview in Path("/workspace/previews").glob("*.png"):
                if str(preview) not in uploaded_files:
                    if upload_to_hf_repo(str(preview)):
                        uploaded_files.add(str(preview))
            
            # Monitor logs
            for log_file in Path("/workspace/logs").rglob("*.log"):
                if str(log_file) not in uploaded_files:
                    if upload_to_hf_repo(str(log_file)):
                        uploaded_files.add(str(log_file))
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(60)

def start_fresh_training():
    """Start fresh training with corrected parameters."""
    
    setup_environment()
    
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        logger.info(f"GPU Status: Available={gpu_available}, Count={gpu_count}")
    except ImportError:
        gpu_available = False
        gpu_count = 0
        logger.warning("PyTorch not available")
    
    # Training parameters using correct StyleGAN-V structure
    training_params = {
        "kimg": 1000,  # Total training iterations (1000 kimg for testing)
        "snap": 50,    # Snapshot/checkpoint interval (every 50 kimg)
        "batch_size": 4 if not gpu_available else 16,  # Smaller batch for CPU
        "gpus": gpu_count,  # Number of GPUs
        "fp32": not gpu_available,  # Use FP32 for CPU, FP16 for GPU
        "num_workers": 2,  # Reduced workers for stability
        "resolution": 1024,
        "aug": "ada",  # Adaptive augmentation
        "mirror": True  # Enable x-flips
    }
    
    # Build the corrected command using StyleGAN-V parameter names
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        f"dataset.path=/workspace/dataset",
        f"dataset.resolution={training_params['resolution']}",
        f"kimg={training_params['kimg']}",  # Correct parameter name
        f"snap={training_params['snap']}",  # Correct parameter name
        f"batch_size={training_params['batch_size']}",
        f"gpus={training_params['gpus']}",
        f"fp32={str(training_params['fp32']).lower()}",
        f"num_workers={training_params['num_workers']}",
        f"aug={training_params['aug']}",
        f"mirror={str(training_params['mirror']).lower()}",
        "hydra.run.dir=/workspace/logs",
        "exp_suffix=monox_fresh_training",
        "outdir=/workspace/checkpoints"  # Output directory for checkpoints
    ]
    
    logger.info("🚀 Starting MonoX Fresh Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: /workspace/dataset")
    logger.info(f"Resolution: {training_params['resolution']}x{training_params['resolution']}")
    logger.info(f"Total KImg: {training_params['kimg']}")
    logger.info(f"Checkpoint Interval: {training_params['snap']} kimg")
    logger.info(f"Batch Size: {training_params['batch_size']}")
    logger.info(f"GPUs: {training_params['gpus']}")
    logger.info(f"Mixed Precision: {not training_params['fp32']}")
    logger.info("=" * 60)
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_training_outputs, daemon=True)
    monitor_thread.start()
    logger.info("📤 Started file monitoring and upload service")
    
    # Create main log file
    main_log_path = "/workspace/logs/monox_fresh_training.log"
    os.makedirs(os.path.dirname(main_log_path), exist_ok=True)
    
    try:
        # Start training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/workspace",
            env=os.environ.copy()
        )
        
        logger.info(f"🔥 Training process started (PID: {process.pid})")
        
        # Stream output to both console and log file
        with open(main_log_path, "w") as log_file:
            log_file.write(f"MonoX Fresh Training Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            log_file.write(f"Command: {' '.join(cmd)}\\n")
            log_file.write("=" * 80 + "\\n")
            
            for line in process.stdout:
                timestamp = time.strftime("%H:%M:%S")
                formatted_line = f"[{timestamp}] {line.rstrip()}"
                
                print(formatted_line)
                log_file.write(formatted_line + "\\n")
                log_file.flush()
                
                # Log important events
                if "kimg" in line.lower() and any(word in line.lower() for word in ["saving", "snapshot", "checkpoint"]):
                    logger.info(f"📸 Checkpoint event: {line.strip()}")
                elif "error" in line.lower() or "exception" in line.lower():
                    logger.error(f"⚠️ Training error: {line.strip()}")
                elif "resuming" in line.lower() or "starting" in line.lower():
                    logger.info(f"🎯 Training milestone: {line.strip()}")
        
        # Wait for completion
        return_code = process.wait()
        
        # Final status
        if return_code == 0:
            logger.info("🎉 Training completed successfully!")
            
            # Create completion summary
            completion_summary = {
                "status": "completed",
                "completion_time": time.time(),
                "return_code": return_code,
                "final_outputs": {
                    "checkpoints": len(list(Path("/workspace/checkpoints").glob("*.pkl"))),
                    "previews": len(list(Path("/workspace/previews").glob("*.png"))),
                    "logs": len(list(Path("/workspace/logs").glob("*.log")))
                }
            }
            
            summary_path = "/workspace/logs/training_completion.json"
            with open(summary_path, "w") as f:
                json.dump(completion_summary, f, indent=2)
            
            upload_to_hf_repo(summary_path)
            
            return True
        else:
            logger.error(f"❌ Training failed with return code: {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Training execution failed: {e}")
        return False

def main():
    """Main execution function."""
    print("🎨 MonoX Fresh Training - Hugging Face Spaces")
    print("🔗 Target Model Repo: lukua/monox")
    print("📊 Dataset: lukua/monox-dataset (868 images)")
    print("=" * 70)
    
    # Validate dataset
    dataset_path = Path("/workspace/dataset")
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        return 1
    
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    if len(image_files) == 0:
        print("❌ No training images found!")
        return 1
    
    print(f"✅ Dataset validated: {len(image_files)} images")
    
    # Start training
    success = start_fresh_training()
    
    if success:
        print("\\n🎉 Fresh training completed successfully!")
        print("📁 Check lukua/monox model repo for:")
        print("  • Checkpoints saved every 5 epochs in /checkpoints")
        print("  • Preview images in /previews")
        print("  • Training logs in /logs")
        return 0
    else:
        print("\\n❌ Training failed - check logs for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())