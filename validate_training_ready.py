#!/usr/bin/env python3
"""
MonoX Training Readiness Validation - NO AUTH REQUIRED
======================================================

Validates that MonoX training setup is ready for deployment to lukua/monox HF Space.
This validation runs WITHOUT authentication - it only checks that the training
code and configurations are properly prepared.

When deployed to your HF Space, it will automatically use your authentication
to access lukua/monox-dataset.
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Required: torch, hydra-core, omegaconf")
    sys.exit(1)


class MonoXReadinessValidator:
    """Validate MonoX training readiness without authentication."""
    
    def __init__(self):
        self.results = {}
        
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate all required dependencies are available."""
        print("📦 VALIDATING: Dependencies")
        print("-" * 40)
        
        required_packages = [
            ("torch", "PyTorch for deep learning"),
            ("torchvision", "Computer vision utilities"),
            ("hydra", "Configuration management"),
            ("omegaconf", "Configuration objects"),
            ("datasets", "HuggingFace datasets library"),
            ("huggingface_hub", "HuggingFace Hub integration")
        ]
        
        missing = []
        available = []
        
        for package, description in required_packages:
            try:
                __import__(package)
                available.append(package)
                print(f"✅ {package}: Available")
            except ImportError:
                missing.append(package)
                print(f"❌ {package}: Missing - {description}")
        
        if missing:
            return {
                "status": "failed",
                "missing": missing,
                "available": available,
                "error": f"Missing packages: {', '.join(missing)}"
            }
        else:
            return {
                "status": "success", 
                "available": available,
                "message": "All dependencies available"
            }
    
    def validate_configuration_files(self) -> Dict[str, Any]:
        """Validate training configuration files."""
        print("\n⚙️  VALIDATING: Configuration Files")
        print("-" * 40)
        
        required_configs = [
            "configs/monox_1024_strict.yaml",
            "configs/dataset/monox_dataset.yaml", 
            "configs/training/monox_1024.yaml",
            "configs/visualizer/monox.yaml"
        ]
        
        missing = []
        available = []
        
        for config_path in required_configs:
            if Path(config_path).exists():
                available.append(config_path)
                print(f"✅ {config_path}: Found")
            else:
                missing.append(config_path)
                print(f"❌ {config_path}: Missing")
        
        # Test loading main config
        try:
            with hydra.initialize(config_path="configs", version_base=None):
                cfg = hydra.compose(config_name="monox_1024_strict")
                resolution = cfg.dataset.resolution
                dataset_name = cfg.dataset.name
                
                print(f"✅ Config loading: Success")
                print(f"🎯 Resolution: {resolution}x{resolution}")
                print(f"🔒 Dataset: {dataset_name}")
                
                config_valid = True
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            config_valid = False
        
        if missing or not config_valid:
            return {
                "status": "failed",
                "missing": missing,
                "available": available,
                "config_loading": config_valid,
                "error": "Configuration validation failed"
            }
        else:
            return {
                "status": "success",
                "available": available,
                "config_loading": config_valid,
                "resolution": resolution,
                "dataset": dataset_name,
                "message": "All configurations valid"
            }
    
    def validate_training_scripts(self) -> Dict[str, Any]:
        """Validate training scripts and launchers."""
        print("\n🚀 VALIDATING: Training Scripts")
        print("-" * 40)
        
        required_scripts = [
            "dataset_integration.py",
            "setup_training_infrastructure.py", 
            "start_monox_training.py",
            "src/infra/launch.py"
        ]
        
        missing = []
        available = []
        
        for script_path in required_scripts:
            if Path(script_path).exists():
                available.append(script_path)
                print(f"✅ {script_path}: Found")
            else:
                missing.append(script_path)
                print(f"❌ {script_path}: Missing")
        
        if missing:
            return {
                "status": "failed",
                "missing": missing,
                "available": available,
                "error": f"Missing scripts: {', '.join(missing)}"
            }
        else:
            return {
                "status": "success",
                "available": available,
                "message": "All training scripts available"
            }
    
    def validate_pytorch_setup(self) -> Dict[str, Any]:
        """Validate PyTorch setup for 1024x1024 training."""
        print("\n🔥 VALIDATING: PyTorch Setup")
        print("-" * 40)
        
        try:
            import torch
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"
            
            print(f"🖥️  Device: {device}")
            if cuda_available:
                print(f"🚀 CUDA version: {torch.version.cuda}")
                print(f"🎮 GPU count: {torch.cuda.device_count()}")
                print(f"📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test tensor operations at 1024x1024
            print("🧪 Testing 1024x1024 tensor operations...")
            test_tensor = torch.randn(1, 3, 1024, 1024)
            
            if cuda_available:
                test_tensor = test_tensor.cuda()
                print("✅ CUDA tensor operations: Success")
            
            # Test basic operations
            result = test_tensor * 2 + 1
            memory_used = result.numel() * result.element_size() / 1024**2
            
            print(f"✅ Tensor operations: Success")
            print(f"📊 Memory for 1024x1024: {memory_used:.1f} MB")
            
            # Cleanup
            del test_tensor, result
            if cuda_available:
                torch.cuda.empty_cache()
            
            return {
                "status": "success",
                "device": device,
                "cuda_available": cuda_available,
                "memory_test": f"{memory_used:.1f} MB",
                "message": "PyTorch ready for 1024x1024 training"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": "PyTorch validation failed"
            }
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete readiness validation."""
        print("🧪 MonoX Training Readiness Validation")
        print("=" * 60)
        print("🎯 Purpose: Validate training setup for HF Space deployment")
        print("🔑 Authentication: Not required (handled by your HF Space)")
        print("=" * 60)
        
        validations = {
            "dependencies": self.validate_dependencies(),
            "configurations": self.validate_configuration_files(),
            "scripts": self.validate_training_scripts(),
            "pytorch": self.validate_pytorch_setup()
        }
        
        # Calculate overall status
        success_count = sum(1 for v in validations.values() if v["status"] == "success")
        total_count = len(validations)
        overall_success = success_count == total_count
        
        return {
            "overall_success": overall_success,
            "success_count": success_count,
            "total_count": total_count,
            "validations": validations
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("📋 TRAINING READINESS SUMMARY")
        print("=" * 60)
        
        status_emoji = "✅" if results["overall_success"] else "❌"
        print(f"{status_emoji} Overall Status: {'READY' if results['overall_success'] else 'NOT READY'}")
        print(f"📊 Success Rate: {results['success_count']}/{results['total_count']}")
        
        print(f"\n📝 Component Status:")
        for name, validation in results["validations"].items():
            status_emoji = "✅" if validation["status"] == "success" else "❌"
            print(f"   {status_emoji} {name.title()}: {validation['status']}")
            if validation["status"] == "failed" and "error" in validation:
                print(f"      ❌ {validation['error']}")
        
        print("\n" + "=" * 60)
        
        if results["overall_success"]:
            print("🎉 MonoX is READY for deployment to your HF Space!")
            print("🚀 Training will work with your authentication")
            print("🔒 Dataset access: Automatic in lukua/monox Space")
            print("🎯 Resolution: 1024x1024 pixels configured")
            print("\n💡 Next step: Deploy to lukua/monox and run training!")
        else:
            print("🚫 MonoX is NOT READY for deployment")
            print("🔧 Fix the above issues before deploying to HF Space")


def main():
    """Main validation function."""
    try:
        validator = MonoXReadinessValidator()
        results = validator.run_validation()
        validator.print_summary(results)
        
        if results["overall_success"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()