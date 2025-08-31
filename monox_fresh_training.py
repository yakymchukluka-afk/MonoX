#!/usr/bin/env python3
"""
MonoX Fresh Training Script for Hugging Face Spaces
Comprehensive training setup with automatic checkpoint/log uploading to lukua/monox model repo.
"""

import os
import sys
import subprocess
import time
import json
import threading
import signal
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/logs/training_manager.log')
    ]
)
logger = logging.getLogger(__name__)

class MonoXTrainingManager:
    """Comprehensive training manager for MonoX StyleGAN-V."""
    
    def __init__(self):
        self.setup_environment()
        self.training_process = None
        self.monitoring_active = False
        self.uploaded_files = set()
        
        # Create directories
        for directory in ["/workspace/logs", "/workspace/checkpoints", "/workspace/previews"]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_environment(self):
        """Setup comprehensive environment for training."""
        env_vars = {
            "PYTHONPATH": "/workspace/.external/stylegan-v:/workspace",
            "DATASET_DIR": "/workspace/dataset",
            "LOGS_DIR": "/workspace/logs",
            "CKPT_DIR": "/workspace/checkpoints",
            "PREVIEWS_DIR": "/workspace/previews",
            "PYTHONUNBUFFERED": "1",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
            # HF_TOKEN should be set via environment variable or Space secret
            "PATH": "/home/ubuntu/.local/bin:" + os.environ.get("PATH", "")
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info("Environment configured for training")
    
    def validate_training_setup(self) -> Dict[str, Any]:
        """Comprehensive validation of training setup."""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "dataset_info": {},
            "system_info": {}
        }
        
        # Check dataset
        dataset_path = Path("/workspace/dataset")
        if not dataset_path.exists():
            validation_results["issues"].append("Dataset directory missing")
            validation_results["valid"] = False
        else:
            image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpeg"))
            validation_results["dataset_info"] = {
                "total_images": len(image_files),
                "sample_files": [f.name for f in image_files[:5]]
            }
            
            if len(image_files) == 0:
                validation_results["issues"].append("No training images found")
                validation_results["valid"] = False
            elif len(image_files) < 100:
                validation_results["warnings"].append(f"Only {len(image_files)} images - may need more for good results")
        
        # Check StyleGAN-V
        stylegan_train = Path(".external/stylegan-v/src/train.py")
        if not stylegan_train.exists():
            validation_results["issues"].append("StyleGAN-V training script missing")
            validation_results["valid"] = False
        
        # Check launch script
        launch_script = Path("src/infra/launch.py")
        if not launch_script.exists():
            validation_results["issues"].append("MonoX launch script missing")
            validation_results["valid"] = False
        
        # Check GPU/System
        try:
            import torch
            validation_results["system_info"] = {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            }
            
            if not torch.cuda.is_available():
                validation_results["warnings"].append("No GPU available - training will be very slow")
        
        except ImportError:
            validation_results["issues"].append("PyTorch not available")
            validation_results["valid"] = False
        
        # Check dependencies
        required_modules = ['hydra', 'omegaconf', 'huggingface_hub']
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                validation_results["issues"].append(f"Missing dependency: {module}")
                validation_results["valid"] = False
        
        return validation_results
    
    def upload_to_model_repo(self, file_path: str, repo_id: str = "lukua/monox") -> bool:
        """Upload file to Hugging Face model repository."""
        try:
            from huggingface_hub import upload_file
            
            # Determine the path in repo based on file type
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
    
    def monitor_and_upload(self):
        """Monitor training outputs and upload to HF model repo."""
        logger.info("🔍 Starting file monitoring and upload service")
        
        directories_to_monitor = {
            "/workspace/checkpoints": ["*.pkl", "*.pt", "*.pth", "*.ckpt"],
            "/workspace/previews": ["*.png", "*.jpg", "*.jpeg"],
            "/workspace/logs": ["*.log", "*.txt"]
        }
        
        while self.monitoring_active:
            try:
                for directory, patterns in directories_to_monitor.items():
                    dir_path = Path(directory)
                    if not dir_path.exists():
                        continue
                    
                    for pattern in patterns:
                        for file_path in dir_path.glob(pattern):
                            file_str = str(file_path)
                            if file_str not in self.uploaded_files:
                                logger.info(f"📤 New file detected: {file_path.name}")
                                if self.upload_to_model_repo(file_str):
                                    self.uploaded_files.add(file_str)
                
                # Upload training progress summary
                self.create_and_upload_progress_summary()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)
        
        logger.info("🛑 File monitoring stopped")
    
    def create_and_upload_progress_summary(self):
        """Create and upload training progress summary."""
        try:
            summary = {
                "timestamp": time.time(),
                "training_status": "running" if self.training_process and self.training_process.poll() is None else "stopped",
                "uploaded_files": len(self.uploaded_files),
                "checkpoints": len(list(Path("/workspace/checkpoints").glob("*.pkl"))),
                "previews": len(list(Path("/workspace/previews").glob("*.png"))),
                "log_files": len(list(Path("/workspace/logs").glob("*.log")))
            }
            
            summary_path = "/workspace/logs/training_progress.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            # Upload summary if it's new or updated
            if summary_path not in self.uploaded_files:
                if self.upload_to_model_repo(summary_path):
                    self.uploaded_files.add(summary_path)
        
        except Exception as e:
            logger.error(f"Failed to create progress summary: {e}")
    
    def start_training(self, 
                      total_kimg: int = 1000,
                      resolution: int = 1024,
                      snapshot_kimg: int = 50,
                      batch_size: int = 4) -> bool:
        """Start fresh training from scratch."""
        
        if self.training_process and self.training_process.poll() is None:
            logger.warning("Training is already running")
            return False
        
        # Validate setup first
        validation = self.validate_training_setup()
        if not validation["valid"]:
            logger.error("Training setup validation failed:")
            for issue in validation["issues"]:
                logger.error(f"  • {issue}")
            return False
        
        # Log warnings
        for warning in validation["warnings"]:
            logger.warning(f"  • {warning}")
        
        # Determine GPU usage
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Build training command
        cmd = [
            sys.executable, "-m", "src.infra.launch",
            f"dataset.path=/workspace/dataset",
            f"dataset.resolution={resolution}",
            f"training.total_kimg={total_kimg}",
            f"training.snapshot_kimg={snapshot_kimg}",
            f"training.batch={batch_size}",
            f"training.num_gpus={num_gpus}",
            f"training.fp16={str(num_gpus > 0).lower()}",  # Enable FP16 only with GPU
            "hydra.run.dir=/workspace/logs",
            "visualizer.output_dir=/workspace/previews",
            f"visualizer.save_every_kimg={snapshot_kimg}",
            "exp_suffix=monox_fresh_training"
        ]
        
        logger.info("🚀 Starting MonoX Fresh Training")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Dataset: {validation['dataset_info']['total_images']} images")
        logger.info(f"Resolution: {resolution}x{resolution}")
        logger.info(f"Total KImg: {total_kimg}")
        logger.info(f"Checkpoint every: {snapshot_kimg} kimg")
        logger.info(f"GPU Count: {num_gpus}")
        
        try:
            # Start training process
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/workspace",
                env=os.environ.copy()
            )
            
            # Start monitoring
            self.monitoring_active = True
            monitor_thread = threading.Thread(target=self.monitor_and_upload, daemon=True)
            monitor_thread.start()
            
            # Stream training output
            log_file_path = "/workspace/logs/fresh_training_output.log"
            with open(log_file_path, "w") as log_file:
                for line in self.training_process.stdout:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_line = f"[{timestamp}] {line.strip()}"
                    
                    print(log_line)
                    log_file.write(log_line + "\\n")
                    log_file.flush()
                    
                    # Check for important events
                    if "kimg" in line.lower() and "snapshot" in line.lower():
                        logger.info(f"📸 Checkpoint saved: {line.strip()}")
                    elif "error" in line.lower() or "exception" in line.lower():
                        logger.error(f"⚠️ Training error: {line.strip()}")
            
            # Wait for completion
            return_code = self.training_process.wait()
            self.monitoring_active = False
            
            if return_code == 0:
                logger.info("🎉 Training completed successfully!")
                self.create_final_summary()
                return True
            else:
                logger.error(f"❌ Training failed with return code: {return_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Training startup failed: {e}")
            self.monitoring_active = False
            return False
    
    def create_final_summary(self):
        """Create final training summary and upload to model repo."""
        try:
            # Count final outputs
            checkpoints = list(Path("/workspace/checkpoints").glob("*.pkl"))
            previews = list(Path("/workspace/previews").glob("*.png"))
            logs = list(Path("/workspace/logs").glob("*.log"))
            
            summary = {
                "training_completed": time.time(),
                "training_duration": "calculated_during_training",
                "final_checkpoints": len(checkpoints),
                "final_previews": len(previews),
                "log_files": len(logs),
                "latest_checkpoint": checkpoints[-1].name if checkpoints else None,
                "dataset_size": len(list(Path("/workspace/dataset").glob("*.jpg")) + list(Path("/workspace/dataset").glob("*.png"))),
                "configuration": {
                    "resolution": 1024,
                    "total_kimg": 1000,
                    "snapshot_kimg": 50,
                    "batch_size": 4
                }
            }
            
            summary_path = "/workspace/logs/final_training_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            # Upload final summary
            self.upload_to_model_repo(summary_path)
            
            logger.info("📋 Final training summary created and uploaded")
            
        except Exception as e:
            logger.error(f"Failed to create final summary: {e}")
    
    def stop_training(self):
        """Stop training gracefully."""
        if self.training_process and self.training_process.poll() is None:
            logger.info("🛑 Stopping training...")
            self.training_process.terminate()
            
            try:
                self.training_process.wait(timeout=30)
                logger.info("✅ Training stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("⚡ Force killing training process")
                self.training_process.kill()
                self.training_process.wait()
            
            self.monitoring_active = False
        else:
            logger.info("No training process to stop")

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info("\\n🛑 Interrupt received, stopping training...")
    if 'training_manager' in globals():
        training_manager.stop_training()
    sys.exit(0)

def main():
    """Main training execution."""
    global training_manager
    
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🎨 MonoX Fresh Training - Hugging Face Spaces Edition")
    print("=" * 70)
    
    # Initialize training manager
    training_manager = MonoXTrainingManager()
    
    # Validate setup
    validation = training_manager.validate_training_setup()
    
    print("📋 Setup Validation:")
    print(f"  Dataset Images: {validation['dataset_info'].get('total_images', 0)}")
    print(f"  PyTorch: {validation['system_info'].get('pytorch_version', 'Unknown')}")
    print(f"  CUDA Available: {validation['system_info'].get('cuda_available', False)}")
    print(f"  GPU Count: {validation['system_info'].get('gpu_count', 0)}")
    
    if not validation["valid"]:
        print("\\n❌ Setup validation failed:")
        for issue in validation["issues"]:
            print(f"  • {issue}")
        return 1
    
    if validation["warnings"]:
        print("\\n⚠️ Warnings:")
        for warning in validation["warnings"]:
            print(f"  • {warning}")
    
    print("\\n✅ Setup validation passed!")
    print("=" * 70)
    
    # Start training
    print("🚀 Starting fresh training from scratch...")
    print("📝 Training will:")
    print("  • Train for 1000 kimg (manageable for testing)")
    print("  • Save checkpoints every 50 kimg (~5 epochs)")
    print("  • Generate preview images at each checkpoint")
    print("  • Upload all outputs to lukua/monox model repo")
    print("  • Log everything for monitoring")
    
    print("\\n" + "=" * 70)
    
    success = training_manager.start_training(
        total_kimg=1000,
        resolution=1024,
        snapshot_kimg=50,
        batch_size=4
    )
    
    if success:
        print("\\n🎉 Training completed successfully!")
        print("📁 Check the lukua/monox model repo for:")
        print("  • Checkpoints in /checkpoints")
        print("  • Preview images in /previews") 
        print("  • Training logs in /logs")
        return 0
    else:
        print("\\n❌ Training failed!")
        print("📋 Check logs for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())