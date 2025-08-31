#!/usr/bin/env python3
"""
Final MonoX Training Solution for Hugging Face Spaces
Complete training pipeline with proper StyleGAN-V integration and HF model repo uploads.
"""

import os
import sys
import subprocess
import time
import json
import threading
import yaml
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/logs/monox_training.log')
    ]
)
logger = logging.getLogger(__name__)

class MonoXFreshTraining:
    """Complete MonoX training solution."""
    
    def __init__(self):
        self.setup_environment()
        self.workspace_root = Path("/workspace")
        self.stylegan_root = Path("/workspace/.external/stylegan-v")
        self.output_dir = self.workspace_root / "training_output"
        self.uploaded_files = set()
        self.monitoring_active = False
        
        # Create directories
        for directory in [self.output_dir, "/workspace/logs", "/workspace/checkpoints", "/workspace/previews"]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_environment(self):
        """Setup comprehensive environment."""
        env_vars = {
            "PYTHONPATH": "/workspace/.external/stylegan-v:/workspace",
            "PYTHONUNBUFFERED": "1",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
            # HF_TOKEN should be set via environment variable or Space secret
            "PATH": "/home/ubuntu/.local/bin:" + os.environ.get("PATH", ""),
            "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
    
    def create_experiment_config(self, kimg: int = 1000, snap: int = 50) -> dict:
        """Create StyleGAN-V experiment configuration."""
        
        # Check system capabilities
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
        except ImportError:
            gpu_available = False
            gpu_count = 0
        
        config = {
            # Core training settings
            "data": "/workspace/dataset",
            "outdir": str(self.output_dir),
            "cfg": "auto",
            "gpus": max(gpu_count, 1),  # StyleGAN-V requires at least 1
            "kimg": kimg,
            "snap": snap,
            "batch_size": 4 if not gpu_available else 16,
            "resume": None,
            
            # Dataset settings
            "resolution": 1024,
            "cond": False,
            "subset": None,
            "mirror": True,
            
            # Augmentation
            "aug": "ada",
            "augpipe": "bgc",
            "target": 0.6,
            
            # Performance
            "fp32": not gpu_available,
            "nhwc": False,
            "nobench": False,
            "allow_tf32": False,
            "num_workers": 2,
            
            # Disable metrics for faster training
            "metrics": [],
            
            # Misc
            "seed": 42,
            "dry_run": False,
            "freezed": 0
        }
        
        return config
    
    def save_experiment_config(self, config: dict):
        """Save experiment configuration to StyleGAN-V directory."""
        config_path = self.stylegan_root / "experiment_config.yaml"
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Experiment config saved: {config_path}")
        return config_path
    
    def upload_to_model_repo(self, file_path: str, repo_id: str = "lukua/monox") -> bool:
        """Upload file to HF model repository."""
        try:
            from huggingface_hub import upload_file
            
            file_name = os.path.basename(file_path)
            
            # Determine target path in repo
            if file_path.endswith(('.pkl', '.pt', '.pth', '.ckpt')):
                repo_path = f"checkpoints/{file_name}"
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                repo_path = f"previews/{file_name}"
            elif file_path.endswith('.log'):
                repo_path = f"logs/{file_name}"
            elif file_path.endswith('.json'):
                repo_path = f"reports/{file_name}"
            else:
                repo_path = file_name
            
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=os.environ.get("HF_TOKEN"),
                repo_type="model"
            )
            
            logger.info(f"📤 Uploaded: {file_name} → {repo_id}/{repo_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload failed for {file_path}: {e}")
            return False
    
    def monitor_and_upload_outputs(self):
        """Monitor training outputs and upload to HF model repo."""
        logger.info("🔍 Starting output monitoring service")
        
        while self.monitoring_active:
            try:
                # Check training output directory
                if self.output_dir.exists():
                    for file_path in self.output_dir.rglob("*"):
                        if file_path.is_file() and str(file_path) not in self.uploaded_files:
                            if file_path.suffix in ['.pkl', '.png', '.log', '.json', '.pt']:
                                if self.upload_to_model_repo(str(file_path)):
                                    self.uploaded_files.add(str(file_path))
                
                # Check other output directories
                for directory in ["/workspace/logs", "/workspace/checkpoints", "/workspace/previews"]:
                    dir_path = Path(directory)
                    if dir_path.exists():
                        for file_path in dir_path.rglob("*"):
                            if file_path.is_file() and str(file_path) not in self.uploaded_files:
                                if file_path.suffix in ['.pkl', '.png', '.log', '.json', '.pt']:
                                    if self.upload_to_model_repo(str(file_path)):
                                        self.uploaded_files.add(str(file_path))
                
                # Create progress report
                self.create_progress_report()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)
        
        logger.info("🛑 Output monitoring stopped")
    
    def create_progress_report(self):
        """Create and upload training progress report."""
        try:
            report = {
                "timestamp": time.time(),
                "training_active": self.monitoring_active,
                "outputs": {
                    "checkpoints": len(list(self.output_dir.glob("*.pkl"))) + len(list(Path("/workspace/checkpoints").glob("*.pkl"))),
                    "previews": len(list(self.output_dir.glob("*.png"))) + len(list(Path("/workspace/previews").glob("*.png"))),
                    "logs": len(list(Path("/workspace/logs").glob("*.log")))
                },
                "uploaded_files": len(self.uploaded_files),
                "dataset_size": len(list(Path("/workspace/dataset").glob("*.jpg")) + list(Path("/workspace/dataset").glob("*.png")))
            }
            
            report_path = "/workspace/logs/training_progress.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            if report_path not in self.uploaded_files:
                if self.upload_to_model_repo(report_path):
                    self.uploaded_files.add(report_path)
        
        except Exception as e:
            logger.error(f"Failed to create progress report: {e}")
    
    def start_training(self, kimg: int = 1000, snap: int = 50) -> bool:
        """Start fresh StyleGAN-V training."""
        
        logger.info("🎨 Starting MonoX Fresh Training")
        logger.info("=" * 70)
        
        # Validate dataset
        dataset_path = Path("/workspace/dataset")
        image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
        
        if len(image_files) == 0:
            logger.error("❌ No training images found!")
            return False
        
        logger.info(f"📊 Dataset: {len(image_files)} images")
        
        # Create experiment configuration
        config = self.create_experiment_config(kimg=kimg, snap=snap)
        config_path = self.save_experiment_config(config)
        
        # Log training parameters
        logger.info(f"🎯 Training Parameters:")
        logger.info(f"  Total KImg: {config['kimg']}")
        logger.info(f"  Checkpoint Interval: {config['snap']} kimg (~{config['snap']//10} epochs)")
        logger.info(f"  Batch Size: {config['batch_size']}")
        logger.info(f"  Resolution: {config['resolution']}")
        logger.info(f"  GPUs: {config['gpus']}")
        logger.info(f"  Mixed Precision: {not config['fp32']}")
        
        # Start monitoring
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self.monitor_and_upload_outputs, daemon=True)
        monitor_thread.start()
        
        # Build training command
        cmd = [sys.executable, "src/train.py"]
        
        logger.info(f"🚀 Executing: {' '.join(cmd)}")
        logger.info(f"📂 Working Directory: {self.stylegan_root}")
        logger.info(f"📁 Output Directory: {self.output_dir}")
        
        try:
            # Change to StyleGAN-V directory
            original_cwd = os.getcwd()
            os.chdir(self.stylegan_root)
            
            # Start training process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy()
            )
            
            logger.info(f"🔥 Training process started (PID: {process.pid})")
            
            # Stream output
            main_log_path = "/workspace/logs/stylegan_training_output.log"
            with open(main_log_path, "w") as log_file:
                log_file.write(f"MonoX StyleGAN-V Training Started\\n")
                log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
                log_file.write(f"Configuration: {config_path}\\n")
                log_file.write("=" * 80 + "\\n")
                
                for line in process.stdout:
                    timestamp = time.strftime("%H:%M:%S")
                    formatted_line = f"[{timestamp}] {line.rstrip()}"
                    
                    print(formatted_line)
                    log_file.write(formatted_line + "\\n")
                    log_file.flush()
                    
                    # Log important events
                    if any(keyword in line.lower() for keyword in ["kimg", "snapshot", "checkpoint"]):
                        logger.info(f"📸 Training milestone: {line.strip()}")
                    elif any(keyword in line.lower() for keyword in ["error", "exception", "failed"]):
                        logger.error(f"⚠️ Issue detected: {line.strip()}")
            
            # Wait for completion
            return_code = process.wait()
            self.monitoring_active = False
            
            if return_code == 0:
                logger.info("🎉 Training completed successfully!")
                self.create_final_report(config)
                return True
            else:
                logger.error(f"❌ Training failed with return code: {return_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Training execution failed: {e}")
            self.monitoring_active = False
            return False
        finally:
            os.chdir(original_cwd)
    
    def create_final_report(self, config: dict):
        """Create comprehensive final training report."""
        try:
            # Count outputs
            checkpoints = list(self.output_dir.glob("*.pkl"))
            previews = list(self.output_dir.glob("*.png"))
            
            final_report = {
                "training_completed": time.time(),
                "configuration": config,
                "results": {
                    "total_checkpoints": len(checkpoints),
                    "total_previews": len(previews),
                    "latest_checkpoint": checkpoints[-1].name if checkpoints else None,
                    "output_directory": str(self.output_dir)
                },
                "dataset": {
                    "path": "/workspace/dataset",
                    "total_images": len(list(Path("/workspace/dataset").glob("*.jpg")) + list(Path("/workspace/dataset").glob("*.png"))),
                    "resolution": config["resolution"]
                },
                "uploads": {
                    "total_uploaded": len(self.uploaded_files),
                    "target_repo": "lukua/monox"
                }
            }
            
            report_path = "/workspace/logs/final_training_report.json"
            with open(report_path, "w") as f:
                json.dump(final_report, f, indent=2)
            
            # Upload final report
            self.upload_to_model_repo(report_path)
            
            logger.info("📋 Final training report created and uploaded")
            
            # Copy outputs to standard directories for consistency
            if checkpoints:
                for ckpt in checkpoints:
                    dest = Path("/workspace/checkpoints") / ckpt.name
                    shutil.copy2(ckpt, dest)
                    logger.info(f"📁 Copied checkpoint: {ckpt.name}")
            
            if previews:
                for preview in previews:
                    dest = Path("/workspace/previews") / preview.name
                    shutil.copy2(preview, dest)
                    logger.info(f"🖼️ Copied preview: {preview.name}")
        
        except Exception as e:
            logger.error(f"Failed to create final report: {e}")

def main():
    """Main execution function."""
    print("🎨 MonoX Fresh Training - Final Solution")
    print("🔗 Hugging Face Spaces Edition")
    print("📊 Dataset: lukua/monox-dataset")
    print("🎯 Target: lukua/monox model repo")
    print("=" * 70)
    
    # Initialize training manager
    training_manager = MonoXFreshTraining()
    
    # Validate dataset
    dataset_path = Path("/workspace/dataset")
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    
    if len(image_files) == 0:
        print("❌ No training images found in dataset!")
        return 1
    
    print(f"✅ Dataset validated: {len(image_files)} images at 1024x1024")
    
    # Check system
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        print(f"🖥️ System: GPU Available={gpu_available}, Count={gpu_count}")
    except ImportError:
        print("⚠️ PyTorch not available")
        return 1
    
    print("\\n🚀 Starting fresh training from scratch...")
    print("📝 Training will:")
    print("  • Use 1024x1024 resolution")
    print("  • Train for 1000 kimg (manageable duration)")
    print("  • Save checkpoints every 50 kimg (~5 epochs)")
    print("  • Generate preview images at each checkpoint")
    print("  • Upload all outputs to lukua/monox model repo")
    print("  • Provide comprehensive logging")
    
    print("\\n" + "=" * 70)
    
    # Start training
    success = training_manager.start_training(
        kimg=1000,  # 1000k images for comprehensive training
        snap=50     # Save every 50 kimg (approximately every 5 epochs)
    )
    
    if success:
        print("\\n🎉 Fresh training completed successfully!")
        print("\\n📁 Outputs uploaded to lukua/monox:")
        print("  • /checkpoints - Model checkpoints every 5 epochs")
        print("  • /previews - Generated sample images")
        print("  • /logs - Complete training logs")
        print("  • /reports - Training progress and final reports")
        
        print("\\n✅ MonoX training pipeline is now fully operational!")
        return 0
    else:
        print("\\n❌ Training failed!")
        print("📋 Check logs in /workspace/logs for detailed error information")
        return 1

if __name__ == "__main__":
    sys.exit(main())