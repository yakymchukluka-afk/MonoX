#!/usr/bin/env python3
"""
Enhanced MonoX Training Script with GPU Monitoring and Real-time Output
Provides better visibility into training progress and resource utilization.
"""

import sys
import os
import subprocess
import time
import threading
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig
import GPUtil

def monitor_gpu():
    """Monitor GPU usage in a separate thread."""
    while True:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"🔥 GPU: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% utilization")
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            time.sleep(30)

def _ensure_styleganv_repo(repo_root: Path) -> Path:
    """Ensure StyleGAN-V submodule is available and up to date."""
    external_dir = repo_root / ".external"
    external_dir.mkdir(parents=True, exist_ok=True)
    sgv_dir = external_dir / "stylegan-v"

    if (sgv_dir / ".git").is_file() or (sgv_dir / ".git").is_dir():
        # Ensure we're on a proper branch and try to update
        try:
            # Check if we're on a branch, if not checkout master
            result = subprocess.run(["git", "-C", str(sgv_dir), "branch", "--show-current"],
                                   capture_output=True, text=True, check=False)
            if not result.stdout.strip():
                # We're in detached HEAD, checkout master
                subprocess.run(["git", "-C", str(sgv_dir), "checkout", "master"], check=False)

            # Now try to pull from the correct remote
            subprocess.run(["git", "-C", str(sgv_dir), "pull", "origin", "master"], check=False)
        except Exception:
            pass
        return sgv_dir

    # Clone if missing - use our fork with the fixes
    repo_url = "https://github.com/yakymchukluka-afk/stylegan-v"
    print(f"Cloning StyleGAN-V from {repo_url} into {sgv_dir}...")
    subprocess.run(["git", "clone", repo_url, str(sgv_dir)], check=True)
    # Ensure we're on master branch after cloning
    subprocess.run(["git", "-C", str(sgv_dir), "checkout", "master"], check=False)
    return sgv_dir

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Launch MonoX training with enhanced monitoring."""
    
    print("🚀 Starting MonoX Training with Enhanced Monitoring...")
    print("=" * 80)
    
    # Get current working directory
    repo_root = Path.cwd()
    
    # Ensure StyleGAN-V repo is ready
    sgv_dir = _ensure_styleganv_repo(repo_root)
    print(f"✅ StyleGAN-V ready at: {sgv_dir}")
    
    # Set up logs directory
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Extract configuration
    dataset_cfg = cfg.get("dataset")
    if isinstance(dataset_cfg, DictConfig):
        dataset_path = str(dataset_cfg.get("path", os.environ.get("DATASET_DIR", "")))
        resolution = dataset_cfg.get("resolution", 1024)
    else:
        dataset_path = os.environ.get("DATASET_DIR", "")
        resolution = 1024

    launcher = cfg.get("launcher", "stylegan")
    exp_suffix = cfg.get("exp_suffix", "monox")
    num_gpus = cfg.get("num_gpus", 1)
    
    # Training parameters
    training = cfg.get("training", {})
    total_kimg = training.get("total_kimg", 3000)
    snapshot_kimg = training.get("snapshot_kimg", 250)
    
    # Visualizer parameters
    visualizer = cfg.get("visualizer", {})
    save_every_kimg = visualizer.get("save_every_kimg", 50)
    output_dir = visualizer.get("output_dir", "previews")
    
    # Sampling parameters
    sampling = cfg.get("sampling", {})
    truncation_psi = sampling.get("truncation_psi", 1.0)

    print(f"📊 Configuration:")
    print(f"   Dataset: {dataset_path}")
    print(f"   Resolution: {resolution}")
    print(f"   Training: {total_kimg} kimg, snapshots every {snapshot_kimg} kimg")
    print(f"   Visualizer: Save every {save_every_kimg} kimg to {output_dir}")
    print(f"   GPUs: {num_gpus}")
    print("=" * 80)

    if launcher == "stylegan":
        # Build StyleGAN-V command with our parameters
        cmd = [
            sys.executable, "-m", "src.infra.launch",
            f"hydra.run.dir=logs",
            f"exp_suffix={exp_suffix}",
            f"dataset.path={dataset_path}",
            f"dataset.resolution={resolution}",
            f"training.kimg={total_kimg}",
            f"training.snap={snapshot_kimg}",
            f"visualizer.save_every_kimg={save_every_kimg}",
            f"visualizer.output_dir={output_dir}",
            f"sampling.truncation_psi={truncation_psi}",
            f"num_gpus={num_gpus}"
        ]
        
        print("🎯 Launching StyleGAN-V with command:")
        print(" ".join(cmd))
        print("=" * 80)

        # Set up environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"
        
        # Ensure src imports work when running as module
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (str(sgv_dir) + (":" + existing_pp if existing_pp else ""))

        # Start GPU monitoring in background
        try:
            import GPUtil
            gpu_monitor = threading.Thread(target=monitor_gpu, daemon=True)
            gpu_monitor.start()
            print("🔥 GPU monitoring started...")
        except ImportError:
            print("⚠️  GPUtil not available, skipping GPU monitoring")

        # Stream logs with enhanced output
        log_file = logs_dir / "train_detailed.log"
        
        print("🚀 Starting training process...")
        print("📝 Logs will be saved to:", log_file)
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            with subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=env, 
                cwd=str(sgv_dir),
                bufsize=1,  # Line buffered
                universal_newlines=True
            ) as proc:
                
                with open(log_file, "w", buffering=1) as log_out:
                    log_out.write(f"=== MonoX Training Started at {time.ctime()} ===\n")
                    log_out.write(f"Command: {' '.join(cmd)}\n")
                    log_out.write("=" * 80 + "\n")
                    
                    line_count = 0
                    for line in proc.stdout:
                        line_count += 1
                        
                        # Print every line to console
                        print(line.rstrip())
                        
                        # Save to log file
                        log_out.write(line)
                        log_out.flush()
                        
                        # Show progress indicators
                        if line_count % 50 == 0:
                            elapsed = time.time() - start_time
                            print(f"⏱️  [{elapsed:.0f}s] {line_count} lines processed...")
                
                ret = proc.wait()
                
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user!")
            print("⏱️  Training ran for:", time.time() - start_time, "seconds")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except:
                proc.kill()
            return
        
        elapsed = time.time() - start_time
        print("=" * 80)
        print(f"⏱️  Training completed in {elapsed:.1f} seconds")
        print(f"✅ Process exited with code {ret}")
        
        if ret != 0:
            print("❌ Training failed! Check logs for details.")
            sys.exit(ret)
        else:
            print("🎉 Training completed successfully!")

    else:
        print(f"❌ Unknown launcher: {launcher}")
        sys.exit(1)

if __name__ == "__main__":
    main()