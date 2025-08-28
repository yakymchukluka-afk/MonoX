#!/usr/bin/env python3
"""
Clean Training Launcher for MonoX + StyleGAN-V in Colab
======================================================

This script launches training with proper environment setup and validation.
Run this AFTER colab_environment_setup.py.

Usage:
    !python colab_training_launcher.py
    !python colab_training_launcher.py --config-name config_clean
    !python colab_training_launcher.py dataset.path=/custom/path training.total_kimg=1000
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional, List
import threading
import queue

# Configuration
MONOX_ROOT = "/content/MonoX"
STYLEGAN_V_DIR = "/content/MonoX/.external/stylegan-v"
RESULTS_DIR = "/content/MonoX/results"

def setup_environment():
    """Ensure environment is properly configured"""
    print("🔧 Setting up training environment...")
    
    # Set environment variables
    env_vars = {
        "MONOX_ROOT": MONOX_ROOT,
        "DATASET_DIR": f"{MONOX_ROOT}/dataset", 
        "LOGS_DIR": f"{RESULTS_DIR}/logs",
        "PREVIEWS_DIR": f"{RESULTS_DIR}/previews",
        "CKPT_DIR": f"{RESULTS_DIR}/checkpoints",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Ensure StyleGAN-V is in Python path
    if STYLEGAN_V_DIR not in sys.path:
        sys.path.insert(0, STYLEGAN_V_DIR)
    
    # Update PYTHONPATH for subprocesses
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = f"{STYLEGAN_V_DIR}:{MONOX_ROOT}"
    if existing_pythonpath:
        new_pythonpath += f":{existing_pythonpath}"
    os.environ["PYTHONPATH"] = new_pythonpath
    
    print(f"✅ PYTHONPATH: {new_pythonpath}")

def run_debug_checklist():
    """Run comprehensive debug checklist"""
    print("\n🔍 Running Debug Checklist")
    print("=" * 40)
    
    checklist_results = []
    
    # 1. Check PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    if STYLEGAN_V_DIR in pythonpath:
        checklist_results.append("✅ PYTHONPATH includes StyleGAN-V")
    else:
        checklist_results.append(f"❌ PYTHONPATH missing StyleGAN-V: {pythonpath}")
    
    # 2. Check src module discoverability
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, '/content/MonoX/.external/stylegan-v'); import src.infra.launch"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            checklist_results.append("✅ src.infra.launch module is discoverable")
        else:
            checklist_results.append(f"❌ src.infra.launch import failed: {result.stderr}")
    except Exception as e:
        checklist_results.append(f"❌ Could not test src module: {e}")
    
    # 3. Check config file exists
    config_path = f"{MONOX_ROOT}/configs/config_clean.yaml"
    if os.path.exists(config_path):
        checklist_results.append("✅ config_clean.yaml exists")
    else:
        checklist_results.append(f"❌ Config file missing: {config_path}")
    
    # 4. Check output directories
    for dir_name, dir_path in [
        ("logs", f"{RESULTS_DIR}/logs"),
        ("previews", f"{RESULTS_DIR}/previews"),
        ("checkpoints", f"{RESULTS_DIR}/checkpoints")
    ]:
        if os.path.exists(dir_path):
            checklist_results.append(f"✅ {dir_name} directory exists: {dir_path}")
        else:
            checklist_results.append(f"❌ {dir_name} directory missing: {dir_path}")
    
    # 5. Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            checklist_results.append(f"✅ GPU available: {gpu_name} ({gpu_count} GPU(s))")
        else:
            checklist_results.append("❌ No GPU available")
    except Exception as e:
        checklist_results.append(f"❌ GPU check failed: {e}")
    
    # 6. Check dataset
    dataset_dir = os.environ.get("DATASET_DIR", "")
    if os.path.exists(dataset_dir):
        files = list(Path(dataset_dir).rglob("*.png")) + list(Path(dataset_dir).rglob("*.jpg"))
        if files:
            checklist_results.append(f"✅ Dataset found: {len(files)} images in {dataset_dir}")
        else:
            checklist_results.append(f"⚠️  Dataset directory exists but no images found: {dataset_dir}")
    else:
        checklist_results.append(f"❌ Dataset directory missing: {dataset_dir}")
    
    # Print results
    for result in checklist_results:
        print(result)
    
    # Return True if all checks passed
    return all("✅" in result for result in checklist_results)

def monitor_gpu_usage():
    """Monitor GPU usage in a separate thread"""
    def gpu_monitor():
        while True:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")
                    print(f"🖥️  GPU: {gpu_util}% utilization, Memory: {mem_used}/{mem_total} MB")
                time.sleep(30)  # Check every 30 seconds
            except Exception:
                break
    
    monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

def launch_stylegan_training(config_overrides: List[str] = None):
    """Launch StyleGAN-V training with proper monitoring"""
    print("\n🚀 Launching StyleGAN-V Training")
    print("=" * 40)
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.infra.launch",
        "--config-path", f"{MONOX_ROOT}/configs",
        "--config-name", "config_clean"
    ]
    
    # Add any overrides
    if config_overrides:
        cmd.extend(config_overrides)
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {STYLEGAN_V_DIR}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Start GPU monitoring
    gpu_monitor_thread = monitor_gpu_usage()
    
    # Prepare environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        # Launch training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=STYLEGAN_V_DIR,
            bufsize=1
        )
        
        print("✅ Training process started!")
        print("📝 Streaming output (Ctrl+C to stop):")
        print("-" * 50)
        
        # Stream output in real time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.rstrip())
                
                # Check for key training milestones
                if "Starting training" in output or "Training started" in output:
                    print("🎯 Training has started!")
                elif "Saving snapshot" in output or "snapshot" in output.lower():
                    print("💾 Snapshot saved!")
                elif "GPU" in output and "%" in output:
                    print(f"🖥️  {output.rstrip()}")
        
        return_code = process.poll()
        print(f"\n🏁 Training process finished with code: {return_code}")
        
        return return_code == 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n💥 Training failed with error: {e}")
        return False

def validate_training_results():
    """Check if training produced expected outputs"""
    print("\n🔍 Validating Training Results")
    print("=" * 40)
    
    results = []
    
    # Check for log files
    logs_dir = f"{RESULTS_DIR}/logs"
    log_files = list(Path(logs_dir).glob("*.log")) if os.path.exists(logs_dir) else []
    if log_files:
        results.append(f"✅ Log files created: {len(log_files)} files")
        # Check if logs contain training progress
        for log_file in log_files[-3:]:  # Check last 3 log files
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "kimg" in content.lower() or "training" in content.lower():
                        results.append(f"✅ Training progress found in {log_file.name}")
                        break
            except:
                pass
    else:
        results.append(f"❌ No log files found in {logs_dir}")
    
    # Check for preview images
    previews_dir = f"{RESULTS_DIR}/previews"
    preview_files = list(Path(previews_dir).glob("*.png")) if os.path.exists(previews_dir) else []
    if preview_files:
        results.append(f"✅ Preview images created: {len(preview_files)} files")
    else:
        results.append(f"❌ No preview images found in {previews_dir}")
    
    # Check for checkpoints
    ckpt_dir = f"{RESULTS_DIR}/checkpoints"
    ckpt_files = list(Path(ckpt_dir).glob("*.pkl")) if os.path.exists(ckpt_dir) else []
    if ckpt_files:
        results.append(f"✅ Checkpoints created: {len(ckpt_files)} files")
    else:
        results.append(f"❌ No checkpoints found in {ckpt_dir}")
    
    # Check final GPU usage
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            mem_used = int(result.stdout.strip())
            if mem_used > 1000:  # More than 1GB
                results.append(f"✅ GPU memory in use: {mem_used} MB")
            else:
                results.append(f"⚠️  Low GPU memory usage: {mem_used} MB")
    except:
        results.append("❌ Could not check GPU memory usage")
    
    for result in results:
        print(result)
    
    return any("✅" in result for result in results)

def main():
    """Main training launcher"""
    print("🚀 MonoX StyleGAN-V Training Launcher")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. Setup environment
        setup_environment()
        
        # 2. Run debug checklist
        checklist_passed = run_debug_checklist()
        if not checklist_passed:
            print("\n⚠️  Some checks failed. Continuing anyway...")
        
        # 3. Parse command line arguments for config overrides
        config_overrides = sys.argv[1:] if len(sys.argv) > 1 else []
        
        # 4. Launch training
        training_success = launch_stylegan_training(config_overrides)
        
        # 5. Validate results
        if training_success:
            validation_success = validate_training_results()
            if validation_success:
                elapsed = time.time() - start_time
                print(f"\n🎉 Training completed successfully in {elapsed:.1f}s!")
                print(f"📁 Check results in: {RESULTS_DIR}")
            else:
                print(f"\n⚠️  Training process completed but no outputs found")
        else:
            print(f"\n❌ Training failed or was interrupted")
            
    except Exception as e:
        print(f"\n💥 Launcher failed: {e}")
        raise

if __name__ == "__main__":
    main()