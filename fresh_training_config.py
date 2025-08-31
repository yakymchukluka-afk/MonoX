#!/usr/bin/env python3
"""
Fresh Training Configuration for MonoX StyleGAN-V
Optimized for Hugging Face Spaces with CPU fallback and proper checkpoint/logging setup.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

class MonoXTrainingConfig:
    """Configuration manager for MonoX training."""
    
    def __init__(self):
        self.setup_environment()
        self.workspace_root = Path("/workspace")
        self.dataset_path = self.workspace_root / "dataset"
        self.logs_dir = self.workspace_root / "logs"
        self.checkpoints_dir = self.workspace_root / "checkpoints"
        self.previews_dir = self.workspace_root / "previews"
        
        # Ensure directories exist
        for directory in [self.logs_dir, self.checkpoints_dir, self.previews_dir]:
            directory.mkdir(exist_ok=True)
    
    def setup_environment(self):
        """Setup environment variables for training."""
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
            # Do not hardcode tokens in source files!
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
    
    def get_training_command(self, 
                           total_kimg: int = 1000,
                           resolution: int = 1024,
                           snapshot_kimg: int = 50,  # Save every 5 epochs (assuming ~10 kimg per epoch)
                           num_gpus: int = 1) -> list:
        """Generate the training command with proper parameters."""
        
        # Use CPU if no GPU available
        import torch
        if not torch.cuda.is_available():
            num_gpus = 0
            print("⚠️ No GPU detected, using CPU training (will be slow)")
        
        cmd = [
            sys.executable, "-m", "src.infra.launch",
            f"dataset.path={self.dataset_path}",
            f"dataset.resolution={resolution}",
            f"training.total_kimg={total_kimg}",
            f"training.snapshot_kimg={snapshot_kimg}",
            f"training.num_gpus={num_gpus}",
            f"hydra.run.dir={self.logs_dir}",
            f"visualizer.output_dir={self.previews_dir}",
            f"visualizer.save_every_kimg={snapshot_kimg}",  # Save previews every checkpoint
            "exp_suffix=monox_fresh_training"
        ]
        
        return cmd
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate the training setup."""
        issues = []
        warnings = []
        
        # Check dataset
        if not self.dataset_path.exists():
            issues.append("Dataset directory does not exist")
        else:
            image_files = list(self.dataset_path.glob("*.jpg")) + list(self.dataset_path.glob("*.png"))
            if len(image_files) == 0:
                issues.append("No image files found in dataset")
            elif len(image_files) < 10:
                warnings.append(f"Only {len(image_files)} images found - may not be enough for training")
            else:
                print(f"✅ Dataset: {len(image_files)} images")
        
        # Check StyleGAN-V
        stylegan_path = Path(".external/stylegan-v/src/train.py")
        if not stylegan_path.exists():
            issues.append("StyleGAN-V training script not found")
        else:
            print("✅ StyleGAN-V training script found")
        
        # Check launch script
        launch_path = Path("src/infra/launch.py")
        if not launch_path.exists():
            issues.append("MonoX launch script not found")
        else:
            print("✅ MonoX launch script found")
        
        # Check GPU/CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
            else:
                warnings.append("No GPU available - training will be slow")
        except ImportError:
            issues.append("PyTorch not available")
        
        # Check dependencies
        required_modules = ['hydra', 'omegaconf', 'numpy', 'PIL']
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module} available")
            except ImportError:
                issues.append(f"Missing dependency: {module}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "dataset_images": len(list(self.dataset_path.glob("*.jpg")) + list(self.dataset_path.glob("*.png"))) if self.dataset_path.exists() else 0
        }
    
    def create_training_script(self) -> str:
        """Create a training script with proper error handling and monitoring."""
        script_content = f'''#!/usr/bin/env python3
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
os.environ.update({{
    "PYTHONPATH": "/workspace/.external/stylegan-v:/workspace",
    "DATASET_DIR": "/workspace/dataset",
    "LOGS_DIR": "/workspace/logs",
    "CKPT_DIR": "/workspace/checkpoints", 
    "PREVIEWS_DIR": "/workspace/previews",
    "PYTHONUNBUFFERED": "1",
    "HF_TOKEN": "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"
}})

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
        print(f"✅ Uploaded {{file_path}} to {{repo_path}}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {{file_path}}: {{e}}")
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
                    print(f"📤 New checkpoint found: {{ckpt_file.name}}")
                    if upload_to_hf_model_repo(str(ckpt_file)):
                        uploaded_files.add(str(ckpt_file))
            
            # Check for new preview images
            for preview_file in previews_dir.glob("*.png"):
                if str(preview_file) not in uploaded_files:
                    print(f"📤 New preview found: {{preview_file.name}}")
                    if upload_to_hf_model_repo(str(preview_file)):
                        uploaded_files.add(str(preview_file))
            
            # Upload latest log file
            latest_log = None
            for log_file in logs_dir.glob("**/*.log"):
                if latest_log is None or log_file.stat().st_mtime > latest_log.stat().st_mtime:
                    latest_log = log_file
            
            if latest_log and str(latest_log) not in uploaded_files:
                print(f"📤 Uploading log: {{latest_log.name}}")
                if upload_to_hf_model_repo(str(latest_log)):
                    uploaded_files.add(str(latest_log))
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\\n🛑 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Monitoring error: {{e}}")
            time.sleep(60)

def main():
    """Main training function."""
    print("🎨 Starting MonoX Fresh Training")
    print("=" * 60)
    
    # Training parameters optimized for fresh training
    training_params = {{
        "total_kimg": 1000,  # Start with 1000 kimg for testing
        "resolution": 1024,
        "snapshot_kimg": 50,  # Save every 50 kimg (~5 epochs)
        "num_gpus": 1 if torch.cuda.is_available() else 0
    }}
    
    print(f"Training Parameters:")
    for key, value in training_params.items():
        print(f"  {{key}}: {{value}}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        f"dataset.path={{training_params['dataset_path'] if 'dataset_path' in training_params else '/workspace/dataset'}}",
        f"dataset.resolution={{training_params['resolution']}}",
        f"training.total_kimg={{training_params['total_kimg']}}",
        f"training.snapshot_kimg={{training_params['snapshot_kimg']}}",
        f"training.num_gpus={{training_params['num_gpus']}}",
        "hydra.run.dir=/workspace/logs",
        "visualizer.output_dir=/workspace/previews",
        f"visualizer.save_every_kimg={{training_params['snapshot_kimg']}}",
        "exp_suffix=monox_fresh_training"
    ]
    
    print(f"\\nTraining Command:")
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
            print("\\n🎉 Training completed successfully!")
        else:
            print(f"\\n❌ Training failed with return code: {{return_code}}")
        
        return return_code
        
    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        return 1

if __name__ == "__main__":
    import torch
    sys.exit(main())
'''

        script_path = "run_fresh_training.py"
        with open(script_path, "w") as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def update_config_for_fresh_training(self):
        """Update configuration for fresh training with proper checkpoint intervals."""
        
        # Update main config for fresh training
        config_updates = {
            "training": {
                "total_kimg": 1000,  # Start with smaller training for testing
                "snapshot_kimg": 50,  # Save every 50 kimg (approximately every 5 epochs)
                "log_dir": "/workspace/logs",
                "preview_dir": "/workspace/previews", 
                "checkpoint_dir": "/workspace/checkpoints",
                "resume": "",  # Fresh training
                "num_gpus": 1
            },
            "dataset": {
                "path": "/workspace/dataset",
                "resolution": 1024
            },
            "visualizer": {
                "save_every_kimg": 50,  # Save previews every checkpoint
                "output_dir": "/workspace/previews"
            },
            "exp_suffix": "monox_fresh_training"
        }
        
        return config_updates

# Initialize configuration
config = MonoXTrainingConfig()

def main():
    """Main function to setup and validate training configuration."""
    print("🎨 MonoX Fresh Training Configuration")
    print("=" * 50)
    
    # Validate setup
    validation = config.validate_setup()
    
    if not validation["valid"]:
        print("❌ Setup validation failed:")
        for issue in validation["issues"]:
            print(f"  • {issue}")
        return False
    
    if validation["warnings"]:
        print("⚠️ Warnings:")
        for warning in validation["warnings"]:
            print(f"  • {warning}")
    
    print(f"✅ Setup validation passed")
    print(f"📊 Dataset: {validation['dataset_images']} images")
    
    # Generate training command
    cmd = config.get_training_command(
        total_kimg=1000,  # Fresh training start
        resolution=1024,
        snapshot_kimg=50,  # Every 5 epochs
        num_gpus=1
    )
    
    print("\\n🚀 Training Command:")
    print(" ".join(cmd))
    
    # Create training script
    script_path = config.create_training_script()
    print(f"\\n📝 Training script created: {script_path}")
    
    # Show configuration summary
    config_summary = config.update_config_for_fresh_training()
    print("\\n⚙️ Training Configuration:")
    print(json.dumps(config_summary, indent=2))
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)