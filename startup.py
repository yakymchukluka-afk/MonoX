#!/usr/bin/env python3
"""
Startup script for MonoX training in Hugging Face Spaces.
Handles environment setup, GPU detection, and dependency installation.
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and CUDA setup."""
    logger.info("Checking GPU availability...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        
        logger.info(f"CUDA Available: {cuda_available}")
        logger.info(f"GPU Count: {gpu_count}")
        
        if cuda_available:
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("No GPU detected. Training will use CPU (very slow).")
            
        return cuda_available, gpu_count
        
    except ImportError:
        logger.error("PyTorch not available. Installing...")
        return False, 0

def install_dependencies():
    """Install required dependencies."""
    logger.info("Checking dependencies...")
    
    # In Hugging Face Spaces, dependencies should be pre-installed
    # We'll just check if they're available
    required_packages = ['torch', 'gradio', 'fastapi', 'uvicorn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            logger.warning(f"❌ {package} is not available")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.info("In Hugging Face Spaces, ensure these are in requirements.txt")
        return False
    
    logger.info("All required dependencies are available")
    return True

def setup_workspace():
    """Setup workspace directories and files."""
    logger.info("Setting up workspace...")
    
    directories = [
        "logs",
        "checkpoints", 
        "previews",
        "dataset",
        "configs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create sample dataset directory structure if it doesn't exist
    dataset_dir = Path("dataset")
    if not any(dataset_dir.iterdir()) if dataset_dir.exists() else True:
        logger.info("Dataset directory is empty. You'll need to upload your training data.")
        
        # Create a placeholder
        with open(dataset_dir / "README.txt", "w") as f:
            f.write("Upload your training images to this directory.\n")
            f.write("Supported formats: PNG, JPG, JPEG\n")
            f.write("Recommended: Square images, consistent resolution\n")

def setup_environment_variables():
    """Setup required environment variables."""
    logger.info("Setting up environment variables...")
    
    env_vars = {
        "PYTHONUNBUFFERED": "1",
        "CUDA_LAUNCH_BLOCKING": "1",  # Better error reporting
        "TORCH_USE_CUDA_DSA": "1",    # Better CUDA debugging
        "DATASET_DIR": "/workspace/dataset",
        "LOGS_DIR": "/workspace/logs",
        "CKPT_DIR": "/workspace/checkpoints",
        "PREVIEWS_DIR": "/workspace/previews"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")

def check_stylegan_v_setup():
    """Check if StyleGAN-V submodule is properly setup."""
    logger.info("Checking StyleGAN-V setup...")
    
    sgv_path = Path(".external/stylegan-v")
    
    if sgv_path.exists() and (sgv_path / "src").exists():
        logger.info("StyleGAN-V submodule appears to be setup")
        return True
    else:
        logger.warning("StyleGAN-V submodule not found or incomplete")
        logger.info("The training launcher will attempt to clone it automatically")
        return False

def run_diagnostic_checks():
    """Run comprehensive diagnostic checks."""
    logger.info("Running diagnostic checks...")
    
    checks = {
        "Python version": sys.version_info >= (3, 7),
        "Workspace writable": os.access("/workspace", os.W_OK),
        "Launch script exists": os.path.exists("src/infra/launch.py"),
        "Requirements file exists": os.path.exists("requirements.txt"),
        "App file exists": os.path.exists("app.py")
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{check}: {status}")
        if not passed:
            all_passed = False
    
    return all_passed

def main():
    """Main startup sequence."""
    logger.info("=== MonoX Training Startup ===")
    
    # Setup environment
    setup_environment_variables()
    setup_workspace()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Check GPU
    cuda_available, gpu_count = check_gpu_availability()
    
    # Check StyleGAN-V
    check_stylegan_v_setup()
    
    # Run diagnostics
    if not run_diagnostic_checks():
        logger.warning("Some diagnostic checks failed. Training may not work properly.")
    
    # System summary
    logger.info("=== Startup Summary ===")
    logger.info(f"GPU Available: {cuda_available}")
    logger.info(f"GPU Count: {gpu_count}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info(f"Workspace Ready: {os.path.exists('app.py')}")
    
    logger.info("=== Startup Complete ===")
    logger.info("You can now:")
    logger.info("1. Use the Gradio web interface")
    logger.info("2. Access JSON API endpoints")
    logger.info("3. Upload training data to /dataset")
    logger.info("4. Start training via the interface")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)