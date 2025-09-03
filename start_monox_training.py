#!/usr/bin/env python3
"""
MonoX Training Launcher - PRODUCTION READY
==========================================

Complete training launcher for MonoX StyleGAN-V at 1024x1024 resolution.
STRICT REQUIREMENT: Requires authenticated access to lukua/monox-dataset.

This script:
1. Validates dataset connection (STRICT)
2. Sets up training infrastructure 
3. Configures StyleGAN-V for 1024x1024
4. Launches training with proper monitoring
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dataset_integration import MonoDatasetLoader
    from setup_training_infrastructure import MonoXTrainingInfrastructure
    from huggingface_hub import HfApi, login
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install: pip install datasets torch hydra-core omegaconf huggingface_hub")
    sys.exit(1)


class MonoXTrainingLauncher:
    """Production launcher for MonoX StyleGAN-V training."""
    
    def __init__(self):
        self.dataset_name = "lukua/monox-dataset"
        self.model_repo = "lukua/monox-model"
        self.config_name = "monox_1024_strict"
        self.resolution = 1024
        
    def validate_authentication(self) -> bool:
        """Validate HuggingFace authentication."""
        print("🔑 Validating HuggingFace Authentication...")
        
        try:
            api = HfApi()
            whoami = api.whoami()
            print(f"✅ Authenticated as: {whoami['name']}")
            print(f"📧 Email: {whoami.get('email', 'N/A')}")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("\n🔧 AUTHENTICATION REQUIRED:")
            print("1. Set HF_TOKEN in Space secrets:")
            print("   - Go to Space Settings → Repository secrets")
            print("   - Add: HF_TOKEN = your_huggingface_token")
            print("2. Get token from: https://huggingface.co/settings/tokens")
            print("3. Restart the Space")
            return False
    
    def validate_dataset_access(self) -> Dict[str, Any]:
        """STRICT: Validate dataset access."""
        print(f"\n🔒 Validating Dataset Access: {self.dataset_name}")
        
        loader = MonoDatasetLoader(resolution=self.resolution)
        
        if loader.connect():
            validation = loader.validate_dataset()
            if validation["valid"]:
                print(f"✅ Dataset validation successful!")
                print(f"📊 Samples: {validation.get('samples', 'unknown')}")
                print(f"🖼️  Image field: {validation.get('image_field', 'unknown')}")
                return {"success": True, "validation": validation}
            else:
                print(f"❌ Dataset validation failed: {validation['error']}")
                return {"success": False, "error": validation["error"]}
        else:
            print(f"❌ Dataset connection failed")
            return {"success": False, "error": "Connection failed"}
    
    def setup_training_infrastructure(self) -> bool:
        """Set up training infrastructure in model repository."""
        print(f"\n🏗️  Setting up training infrastructure: {self.model_repo}")
        
        try:
            infrastructure = MonoXTrainingInfrastructure(self.model_repo)
            result = infrastructure.create_directory_structure()
            
            if result["success"]:
                print(f"✅ Infrastructure setup complete!")
                print(f"📂 Created: {', '.join(result['directories'])}")
                return True
            else:
                print(f"❌ Infrastructure setup failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"❌ Infrastructure setup error: {e}")
            return False
    
    def launch_training(self) -> bool:
        """Launch StyleGAN-V training with MonoX configuration."""
        print(f"\n🚀 Launching StyleGAN-V Training...")
        print(f"🎯 Resolution: {self.resolution}x{self.resolution}")
        print(f"⚙️  Configuration: {self.config_name}")
        
        try:
            # Build training command
            cmd = [
                sys.executable,
                "src/infra/launch.py",
                "-cn", self.config_name,
                f"dataset.resolution={self.resolution}",
                f"training.batch_size=4",  # Optimized for 1024x1024
                "training.fp16=true",
                f"exp_suffix=monox_1024_{int(time.time())}"
            ]
            
            print(f"📋 Training command:")
            print(f"   {' '.join(cmd)}")
            
            # Set environment for training
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available
            
            # Launch training process
            print(f"\n🎬 Starting training process...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(Path(__file__).parent)
            )
            
            # Monitor training output
            print(f"📺 Training output:")
            print("-" * 50)
            
            for line in process.stdout:
                print(line.rstrip())
                
                # Check for success indicators
                if "Training started" in line or "Epoch" in line:
                    print("✅ Training successfully initiated!")
                
                # Check for errors
                if "ERROR" in line or "FAILED" in line:
                    print(f"❌ Training error detected: {line.strip()}")
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print("\n✅ Training completed successfully!")
                return True
            else:
                print(f"\n❌ Training failed with exit code: {return_code}")
                return False
                
        except Exception as e:
            print(f"❌ Training launch failed: {e}")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run complete MonoX training setup and launch."""
        print("🎨 MonoX StyleGAN-V Training Launcher")
        print("=" * 60)
        print(f"🎯 Target: {self.resolution}x{self.resolution} resolution")
        print(f"🔒 Dataset: {self.dataset_name} (private)")
        print(f"📦 Model: {self.model_repo}")
        print("=" * 60)
        
        # Step 1: Validate authentication
        if not self.validate_authentication():
            print("\n🚫 TRAINING BLOCKED: Authentication required")
            return False
        
        # Step 2: Validate dataset access
        dataset_result = self.validate_dataset_access()
        if not dataset_result["success"]:
            print(f"\n🚫 TRAINING BLOCKED: Dataset access failed")
            print(f"❌ Error: {dataset_result['error']}")
            return False
        
        # Step 3: Setup infrastructure
        if not self.setup_training_infrastructure():
            print(f"\n🚫 TRAINING BLOCKED: Infrastructure setup failed")
            return False
        
        # Step 4: Launch training
        if self.launch_training():
            print(f"\n🎉 MonoX training launched successfully!")
            print(f"📊 Monitor progress in:")
            print(f"   - Samples: samples/")
            print(f"   - Logs: logs/")
            print(f"   - Checkpoints: checkpoints/")
            print(f"   - Model repo: {self.model_repo}")
            return True
        else:
            print(f"\n❌ Training launch failed")
            return False


def main():
    """Main function."""
    try:
        launcher = MonoXTrainingLauncher()
        success = launcher.run_complete_setup()
        
        if success:
            print("\n🚀 MonoX training is running!")
            sys.exit(0)
        else:
            print("\n🚫 MonoX training failed to start")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Training launch interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()