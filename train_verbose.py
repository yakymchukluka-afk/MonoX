#!/usr/bin/env python3
"""
Verbose MonoX Training Script
Enhanced version with better real-time output and progress monitoring.
"""

import sys
import os
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig

# Global flag for graceful shutdown
training_process = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global training_process
    print("\n🛑 Received interrupt signal! Stopping training...")
    if training_process:
        training_process.terminate()
        try:
            training_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            training_process.kill()
    sys.exit(0)

def _ensure_styleganv_repo(repo_root: Path) -> Path:
    """Ensure StyleGAN-V submodule is available and up to date."""
    external_dir = repo_root / ".external"
    external_dir.mkdir(parents=True, exist_ok=True)
    sgv_dir = external_dir / "stylegan-v"

    if (sgv_dir / ".git").is_file() or (sgv_dir / ".git").is_dir():
        try:
            result = subprocess.run(["git", "-C", str(sgv_dir), "branch", "--show-current"],
                                   capture_output=True, text=True, check=False)
            if not result.stdout.strip():
                subprocess.run(["git", "-C", str(sgv_dir), "checkout", "master"], check=False)
            subprocess.run(["git", "-C", str(sgv_dir), "pull", "origin", "master"], check=False)
        except Exception:
            pass
        return sgv_dir

    repo_url = "https://github.com/yakymchukluka-afk/stylegan-v"
    print(f"Cloning StyleGAN-V from {repo_url} into {sgv_dir}...")
    subprocess.run(["git", "clone", repo_url, str(sgv_dir)], check=True)
    subprocess.run(["git", "-C", str(sgv_dir), "checkout", "master"], check=False)
    return sgv_dir

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Launch MonoX training with verbose output."""
    
    global training_process
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting MonoX Training (Verbose Mode)...")
    print("💡 Press Ctrl+C to stop training gracefully")
    print("=" * 80)
    
    repo_root = Path.cwd()
    sgv_dir = _ensure_styleganv_repo(repo_root)
    print(f"✅ StyleGAN-V ready at: {sgv_dir}")
    
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Extract configuration with robust handling
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
    
    training = cfg.get("training", {})
    total_kimg = training.get("total_kimg", 3000)
    snapshot_kimg = training.get("snapshot_kimg", 250)
    
    visualizer = cfg.get("visualizer", {})
    save_every_kimg = visualizer.get("save_every_kimg", 50)
    output_dir = visualizer.get("output_dir", "previews")
    
    sampling = cfg.get("sampling", {})
    truncation_psi = sampling.get("truncation_psi", 1.0)

    print(f"📊 Training Configuration:")
    print(f"   🗂️  Dataset: {dataset_path}")
    print(f"   🖼️  Resolution: {resolution}x{resolution}")
    print(f"   🎯 Training: {total_kimg} kimg total")
    print(f"   💾 Snapshots: Every {snapshot_kimg} kimg")
    print(f"   🎨 Previews: Every {save_every_kimg} kimg → {output_dir}/")
    print(f"   🔥 GPUs: {num_gpus}")
    print(f"   🎲 Truncation: {truncation_psi}")
    print("=" * 80)

    if launcher == "stylegan":
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
        
        print("🎯 StyleGAN-V Command:")
        for i, arg in enumerate(cmd):
            if i == 0:
                print(f"   {arg} \\")
            elif i == len(cmd) - 1:
                print(f"     {arg}")
            else:
                print(f"     {arg} \\")
        print("=" * 80)

        # Enhanced environment setup - Force GPU usage
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU 0 is used
        env["CUDA_LAUNCH_BLOCKING"] = "1"  # Better CUDA error reporting
        env["TORCH_USE_CUDA_DSA"] = "1"    # Enable CUDA memory debugging
        
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (str(sgv_dir) + (":" + existing_pp if existing_pp else ""))

        log_file = logs_dir / f"train_verbose_{int(time.time())}.log"
        
        print(f"📝 Detailed logs: {log_file}")
        print("🚀 Starting training process...")
        print("⏱️  Training progress will be shown below:")
        print("=" * 80)
        
        start_time = time.time()
        line_count = 0
        last_progress_time = start_time
        
        try:
            # Use more aggressive output flushing
            training_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=env, 
                cwd=str(sgv_dir),
                bufsize=0,  # Unbuffered
                universal_newlines=True
            )
            
            with open(log_file, "w", buffering=1) as log_out:
                log_out.write(f"=== MonoX Training Started: {time.ctime()} ===\n")
                log_out.write(f"Command: {' '.join(cmd)}\n")
                log_out.write("=" * 80 + "\n")
                
                for line in training_process.stdout:
                    line_count += 1
                    current_time = time.time()
                    
                    # Always print the line
                    print(line.rstrip())
                    sys.stdout.flush()
                    
                    # Save to log
                    log_out.write(line)
                    log_out.flush()
                    
                    # Periodic progress indicators
                    if current_time - last_progress_time >= 60:  # Every minute
                        elapsed = current_time - start_time
                        print(f"\n⏱️  === PROGRESS UPDATE: {elapsed:.0f}s elapsed, {line_count} lines processed ===\n")
                        last_progress_time = current_time
                
                ret = training_process.wait()
                
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user!")
            elapsed = time.time() - start_time
            print(f"⏱️  Training ran for {elapsed:.1f} seconds")
            return
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            return
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"⏱️  Training duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"📊 Total output lines: {line_count}")
        
        if ret == 0:
            print("🎉 Training completed successfully!")
        else:
            print(f"❌ Training failed with exit code {ret}")
            print(f"📝 Check detailed logs: {log_file}")
            sys.exit(ret)

    else:
        print(f"❌ Unknown launcher: {launcher}")
        sys.exit(1)

if __name__ == "__main__":
    main()