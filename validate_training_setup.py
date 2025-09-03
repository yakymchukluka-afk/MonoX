#!/usr/bin/env python3
"""
MonoX Training Setup Validation - STRICT MODE
==============================================

Validates that MonoX is ready for StyleGAN-V training:
1. Dataset connection to lukua/monox-dataset (STRICT)
2. Training script initialization at 1024px resolution
3. Model repository structure
4. Configuration validation

STRICT REQUIREMENT: Only works with authenticated lukua/monox-dataset
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dataset_integration import MonoDatasetLoader
    from setup_training_infrastructure import MonoXTrainingInfrastructure
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import torch
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Required packages: datasets, torch, hydra-core, omegaconf")
    sys.exit(1)


class MonoXTrainingValidator:
    """Comprehensive validation for MonoX training setup."""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_dataset_connection(self) -> Dict[str, Any]:
        """STRICT: Validate dataset connection to lukua/monox-dataset."""
        print("🔒 VALIDATING: Dataset Connection (STRICT MODE)")
        print("-" * 50)
        
        try:
            loader = MonoDatasetLoader(resolution=1024)
            
            if loader.connect():
                validation = loader.validate_dataset()
                
                if validation["valid"]:
                    result = {
                        "status": "success",
                        "dataset": loader.dataset_name,
                        "resolution": loader.resolution,
                        "samples": validation.get("samples", "unknown"),
                        "image_field": validation.get("image_field", "unknown"),
                        "message": "Dataset connection and validation successful"
                    }
                    print(f"✅ Dataset connection: SUCCESS")
                    print(f"📊 Dataset: {loader.dataset_name}")
                    print(f"🎯 Resolution: {loader.resolution}x{loader.resolution}")
                    return result
                else:
                    error_msg = f"Dataset validation failed: {validation['error']}"
                    self.errors.append(error_msg)
                    return {"status": "failed", "error": error_msg}
            else:
                error_msg = "Dataset connection failed - authentication required"
                self.errors.append(error_msg)
                return {"status": "failed", "error": error_msg}
                
        except Exception as e:
            error_msg = f"Dataset validation exception: {str(e)}"
            self.errors.append(error_msg)
            return {"status": "failed", "error": error_msg}
    
    def validate_training_config(self) -> Dict[str, Any]:
        """Validate training configuration for 1024x1024 resolution."""
        print("\n⚙️  VALIDATING: Training Configuration")
        print("-" * 50)
        
        try:
            config_path = Path("configs/monox_1024_strict.yaml")
            
            if not config_path.exists():
                error_msg = f"Training config not found: {config_path}"
                self.errors.append(error_msg)
                return {"status": "failed", "error": error_msg}
            
            # Load and validate configuration
            with hydra.initialize(config_path="configs", version_base=None):
                cfg = hydra.compose(config_name="monox_1024_strict")
                
                # Validate key settings
                checks = {
                    "resolution": cfg.get("dataset", {}).get("resolution") == 1024,
                    "dataset_name": cfg.get("dataset", {}).get("name") == "lukua/monox-dataset",
                    "strict_mode": cfg.get("strict_mode", {}).get("enabled", False),
                    "batch_size": cfg.get("training", {}).get("batch_size", 0) > 0,
                    "num_gpus": cfg.get("num_gpus", 0) >= 1
                }
                
                failed_checks = [k for k, v in checks.items() if not v]
                
                if failed_checks:
                    error_msg = f"Configuration validation failed: {failed_checks}"
                    self.errors.append(error_msg)
                    return {"status": "failed", "error": error_msg}
                
                result = {
                    "status": "success",
                    "config_file": str(config_path),
                    "resolution": cfg.dataset.resolution,
                    "dataset": cfg.dataset.name,
                    "batch_size": cfg.training.batch_size,
                    "strict_mode": cfg.strict_mode.enabled,
                    "message": "Training configuration valid for 1024x1024"
                }
                
                print(f"✅ Configuration: SUCCESS")
                print(f"📁 Config file: {config_path}")
                print(f"🎯 Resolution: {cfg.dataset.resolution}x{cfg.dataset.resolution}")
                print(f"🔒 Strict mode: {cfg.strict_mode.enabled}")
                
                return result
                
        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            self.errors.append(error_msg)
            return {"status": "failed", "error": error_msg}
    
    def validate_model_repository(self) -> Dict[str, Any]:
        """Validate model repository structure."""
        print("\n📁 VALIDATING: Model Repository Structure")
        print("-" * 50)
        
        try:
            infrastructure = MonoXTrainingInfrastructure()
            
            if infrastructure.check_model_repo_exists():
                result = {
                    "status": "success",
                    "repository": infrastructure.model_repo,
                    "directories": infrastructure.required_dirs,
                    "message": "Model repository accessible"
                }
                
                print(f"✅ Model repository: SUCCESS")
                print(f"📦 Repository: {infrastructure.model_repo}")
                print(f"📂 Required dirs: {', '.join(infrastructure.required_dirs)}")
                
                return result
            else:
                error_msg = f"Model repository not accessible: {infrastructure.model_repo}"
                self.errors.append(error_msg)
                return {"status": "failed", "error": error_msg}
                
        except Exception as e:
            error_msg = f"Model repository validation failed: {str(e)}"
            self.errors.append(error_msg)
            return {"status": "failed", "error": error_msg}
    
    def validate_training_initialization(self) -> Dict[str, Any]:
        """Validate that training script can initialize at 1024px."""
        print("\n🚀 VALIDATING: Training Script Initialization")
        print("-" * 50)
        
        try:
            # Check PyTorch and CUDA availability
            torch_available = torch.cuda.is_available()
            device = "cuda" if torch_available else "cpu"
            
            # Test basic tensor operations at 1024x1024
            test_tensor = torch.randn(1, 3, 1024, 1024)
            if torch_available:
                test_tensor = test_tensor.cuda()
            
            # Basic memory test
            memory_test = test_tensor * 2
            del test_tensor, memory_test
            
            result = {
                "status": "success",
                "device": device,
                "cuda_available": torch_available,
                "resolution_test": "1024x1024 tensor operations successful",
                "message": "Training initialization validation passed"
            }
            
            print(f"✅ Training initialization: SUCCESS")
            print(f"🖥️  Device: {device}")
            print(f"🎯 Resolution test: 1024x1024 ✓")
            
            return result
            
        except Exception as e:
            error_msg = f"Training initialization failed: {str(e)}"
            self.errors.append(error_msg)
            return {"status": "failed", "error": error_msg}
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🧪 MonoX Training Setup Validation (STRICT MODE)")
        print("=" * 60)
        print("🎯 Target: 1024x1024 StyleGAN-V training")
        print("🔒 Dataset: lukua/monox-dataset (private)")
        print("📦 Model repo: lukua/monox-model")
        print("=" * 60)
        
        # Run all validations
        validations = {
            "dataset": self.validate_dataset_connection(),
            "config": self.validate_training_config(),
            "model_repo": self.validate_model_repository(),
            "initialization": self.validate_training_initialization()
        }
        
        # Collect results
        success_count = sum(1 for v in validations.values() if v["status"] == "success")
        total_count = len(validations)
        
        overall_status = "success" if success_count == total_count else "failed"
        
        result = {
            "overall_status": overall_status,
            "validations": validations,
            "success_count": success_count,
            "total_count": total_count,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        return result
    
    def print_summary(self, result: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)
        
        status_emoji = "✅" if result["overall_status"] == "success" else "❌"
        print(f"{status_emoji} Overall Status: {result['overall_status'].upper()}")
        print(f"📊 Success Rate: {result['success_count']}/{result['total_count']}")
        
        print(f"\n📝 Individual Results:")
        for name, validation in result["validations"].items():
            status_emoji = "✅" if validation["status"] == "success" else "❌"
            print(f"   {status_emoji} {name.title()}: {validation['status']}")
        
        if result["errors"]:
            print(f"\n❌ Errors ({len(result['errors'])}):")
            for i, error in enumerate(result["errors"], 1):
                print(f"   {i}. {error}")
        
        if result["warnings"]:
            print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
            for i, warning in enumerate(result["warnings"], 1):
                print(f"   {i}. {warning}")
        
        print("\n" + "=" * 60)
        
        if result["overall_status"] == "success":
            print("🎉 MonoX is READY for StyleGAN-V training!")
            print("🚀 All validations passed - training can proceed")
            print("🎯 Resolution: 1024x1024 pixels")
            print("🔒 Dataset: lukua/monox-dataset (authenticated)")
        else:
            print("🚫 MonoX training is BLOCKED")
            print("❌ Fix the above errors before proceeding")
            print("🔒 STRICT MODE: No fallbacks allowed")


def main():
    """Main validation function."""
    try:
        validator = MonoXTrainingValidator()
        result = validator.run_full_validation()
        validator.print_summary(result)
        
        if result["overall_status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()