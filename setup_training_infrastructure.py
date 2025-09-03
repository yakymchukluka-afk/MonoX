#!/usr/bin/env python3
"""
MonoX Training Infrastructure Setup
===================================

Sets up the training infrastructure in 'lukua/monox-model' HF Model repository
with proper directory structure for StyleGAN-V training outputs.

STRICT REQUIREMENT: Only works with authenticated access to lukua/monox-dataset
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import tempfile
import json

try:
    from huggingface_hub import HfApi, Repository, create_repo
    from dataset_integration import MonoDatasetLoader
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Run: pip install huggingface_hub")
    sys.exit(1)


class MonoXTrainingInfrastructure:
    """Setup training infrastructure for MonoX StyleGAN-V training."""
    
    def __init__(self, model_repo: str = "lukua/monox-model"):
        self.model_repo = model_repo
        self.api = HfApi()
        self.required_dirs = [
            "checkpoints",
            "samples", 
            "logs"
        ]
        
    def validate_dataset_access(self) -> bool:
        """STRICT: Validate that dataset access is properly configured."""
        print("🔒 STRICT VALIDATION: Checking dataset access...")
        
        loader = MonoDatasetLoader()
        if loader.connect():
            validation = loader.validate_dataset()
            if validation["valid"]:
                print("✅ Dataset access validated - training infrastructure setup can proceed")
                return True
            else:
                print(f"❌ Dataset validation failed: {validation['error']}")
                return False
        else:
            print("❌ Dataset connection failed - cannot setup training infrastructure")
            print("🚫 BLOCKED: Training infrastructure requires dataset access")
            return False
    
    def check_model_repo_exists(self) -> bool:
        """Check if the model repository exists."""
        try:
            repo_info = self.api.repo_info(repo_id=self.model_repo, repo_type="model")
            print(f"✅ Model repository exists: {self.model_repo}")
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"❌ Model repository not found: {self.model_repo}")
                return False
            else:
                print(f"❌ Error checking repository: {e}")
                return False
    
    def create_directory_structure(self) -> Dict[str, Any]:
        """Create the required directory structure in the model repository."""
        if not self.validate_dataset_access():
            return {"success": False, "error": "Dataset access validation failed"}
        
        if not self.check_model_repo_exists():
            return {"success": False, "error": f"Model repository {self.model_repo} does not exist"}
        
        try:
            print(f"📁 Setting up training infrastructure in {self.model_repo}")
            
            # Create temporary directory for repository operations
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = Path(temp_dir) / "monox-model"
                
                # Clone the repository
                print("📥 Cloning model repository...")
                repo = Repository(
                    local_dir=str(repo_path),
                    clone_from=self.model_repo,
                    repo_type="model"
                )
                
                # Create directory structure
                created_dirs = []
                for dir_name in self.required_dirs:
                    dir_path = repo_path / dir_name
                    dir_path.mkdir(exist_ok=True)
                    created_dirs.append(dir_name)
                    
                    # Create .gitkeep files to ensure directories are tracked
                    gitkeep_path = dir_path / ".gitkeep"
                    gitkeep_path.write_text("# Keep this directory in git\n")
                    
                    # Create README for each directory
                    readme_content = self._get_directory_readme(dir_name)
                    readme_path = dir_path / "README.md"
                    readme_path.write_text(readme_content)
                
                # Create main training infrastructure README
                main_readme = repo_path / "TRAINING_INFRASTRUCTURE.md"
                main_readme.write_text(self._get_main_readme())
                
                # Create training configuration
                config_path = repo_path / "training_config.json"
                config_path.write_text(json.dumps(self._get_training_config(), indent=2))
                
                # Commit and push changes
                print("📤 Committing training infrastructure...")
                repo.git_add()
                repo.git_commit("Setup MonoX training infrastructure with directory structure")
                repo.git_push()
                
                print(f"✅ Training infrastructure created successfully!")
                return {
                    "success": True,
                    "directories": created_dirs,
                    "repository": self.model_repo,
                    "message": "Training infrastructure ready for StyleGAN-V"
                }
                
        except Exception as e:
            print(f"❌ Failed to create training infrastructure: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_directory_readme(self, dir_name: str) -> str:
        """Get README content for specific directories."""
        readmes = {
            "checkpoints": """# MonoX Model Checkpoints

This directory stores StyleGAN-V model checkpoints during training.

## Structure:
- `epoch_*.pth` - Model weights at specific epochs
- `optimizer_*.pth` - Optimizer states for resuming training
- `latest.pth` - Most recent checkpoint for easy resuming

## Usage:
Checkpoints are automatically saved during training based on the snapshot_kimg configuration.
""",
            "samples": """# MonoX Generated Samples

This directory contains generated images during training for monitoring progress.

## Structure:
- `samples_epoch_*.png` - Sample grids at each epoch
- `progress_*.png` - Training progress visualizations
- `validation_*.png` - Validation samples

## Usage:
Samples are generated automatically during training to monitor quality and progress.
""",
            "logs": """# MonoX Training Logs

This directory contains training logs, metrics, and TensorBoard data.

## Structure:
- `train.log` - Main training log file
- `metrics.json` - Training metrics in JSON format
- `tensorboard/` - TensorBoard logs for visualization
- `config_*.yaml` - Training configurations used

## Usage:
Monitor training progress and debug issues using these logs.
"""
        }
        return readmes.get(dir_name, f"# {dir_name.title()}\n\nTraining data for MonoX StyleGAN-V.")
    
    def _get_main_readme(self) -> str:
        """Get main training infrastructure README."""
        return """# MonoX Training Infrastructure

This repository contains the training infrastructure for MonoX StyleGAN-V model.

## Directory Structure:

### 📁 checkpoints/
Model weights and training states saved during training.

### 📁 samples/  
Generated images per epoch for monitoring training progress.

### 📁 logs/
Training logs, metrics, and TensorBoard data.

## Training Configuration:
- **Resolution**: 1024x1024 pixels
- **Dataset**: `lukua/monox-dataset` (private)
- **Architecture**: StyleGAN-V
- **Authentication**: Required for dataset access

## Usage:
This infrastructure is automatically populated during MonoX training within the HF Space `lukua/monox`.

## Requirements:
- Authenticated access to `lukua/monox-dataset`
- HuggingFace authentication configured
- Training executed from `lukua/monox` HF Space

---
*Generated by MonoX Training Infrastructure Setup*
"""
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get training configuration JSON."""
        return {
            "model_name": "MonoX StyleGAN-V",
            "resolution": 1024,
            "dataset": "lukua/monox-dataset",
            "dataset_type": "private",
            "architecture": "StyleGAN-V",
            "training_space": "lukua/monox",
            "model_repository": "lukua/monox-model",
            "directories": {
                "checkpoints": "Model weights and training states",
                "samples": "Generated images for monitoring",
                "logs": "Training logs and metrics"
            },
            "requirements": [
                "Authenticated access to lukua/monox-dataset",
                "HuggingFace authentication configured",
                "Training from lukua/monox HF Space"
            ],
            "created_by": "MonoX Training Infrastructure Setup",
            "strict_mode": True,
            "fallback_datasets": None
        }


def main():
    """Main function to setup training infrastructure."""
    print("🏗️  MonoX Training Infrastructure Setup")
    print("=" * 50)
    print("🎯 Target: lukua/monox-model")
    print("🔒 STRICT MODE: Requires lukua/monox-dataset access")
    print("=" * 50)
    
    try:
        infrastructure = MonoXTrainingInfrastructure()
        result = infrastructure.create_directory_structure()
        
        if result["success"]:
            print(f"\n🎉 Training infrastructure setup complete!")
            print(f"📁 Repository: {result['repository']}")
            print(f"📂 Created directories: {', '.join(result['directories'])}")
            print(f"✅ {result['message']}")
            print(f"\n🚀 Ready for MonoX training at 1024x1024 resolution!")
        else:
            print(f"\n❌ Training infrastructure setup failed!")
            print(f"🚫 Error: {result['error']}")
            print(f"\n💡 Ensure:")
            print(f"   1. HuggingFace authentication is configured")
            print(f"   2. Access to 'lukua/monox-dataset' is available") 
            print(f"   3. Model repository 'lukua/monox-model' exists")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"🚫 TRAINING INFRASTRUCTURE BLOCKED")
        sys.exit(1)


if __name__ == "__main__":
    main()