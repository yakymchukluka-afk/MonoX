#!/usr/bin/env python3
"""
GPU-Forced MonoX Training Script
Explicitly ensures GPU usage and monitors it in real-time.
"""

import sys
import os
import subprocess
import time
import signal
import torch
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

def check_gpu_setup():
    """Verify GPU is available and working."""
    print("🔍 GPU Setup Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU tensor
        try:
            x = torch.randn(100, 100, device='cuda')
            y = torch.mm(x, x)
            print(f"   ✅ GPU test passed: {y.device}")
            return True
        except Exception as e:
            print(f"   ❌ GPU test failed: {e}")
            return False
    else:
        print("   ❌ CUDA not available!")
        return False

def force_gpu_environment():
    """Set up environment to force GPU usage."""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'
    
    # Force PyTorch to use GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("✅ Forced GPU environment setup complete")
    else:
        print("❌ Cannot force GPU - CUDA not available")

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
    """Launch MonoX training with forced GPU usage."""
    
    global training_process
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting MonoX Training (GPU-Forced Mode)...")
    print("=" * 80)
    
    # Force GPU setup
    force_gpu_environment()
    
    # Check GPU availability
    if not check_gpu_setup():
        print("❌ GPU setup failed! Exiting...")
        sys.exit(1)
    
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
    print(f"   🔥 GPUs: {num_gpus} (forced to use GPU 0)")
    print(f"   🎲 Truncation: {truncation_psi}")
    print("=" * 80)

    if launcher == "stylegan":
        # Build command with explicit GPU forcing
        cmd = [
            sys.executable, "-m", "src.infra.launch",
            f"hydra.run.dir=logs",
            f"exp_suffix={exp_suffix}",
            f"dataset.path={dataset_path}",
            f"dataset.resolution={resolution}",
            f"training.kimg={total_kimg}",
            f"training.snap={snapshot_kimg}",
            f"training.gpus={num_gpus}",  # Explicit GPU count
            f"visualizer.save_every_kimg={save_every_kimg}",
            f"visualizer.output_dir={output_dir}",
            f"sampling.truncation_psi={truncation_psi}",
            f"num_gpus={num_gpus}"
        ]
        
        print("🎯 StyleGAN-V Command:")
        print("   " + " \\\n   ".join(cmd))
        print("=" * 80)

        # GPU-focused environment setup
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["TORCH_USE_CUDA_DSA"] = "1"
        env["NVIDIA_TF32_OVERRIDE"] = "0"  # Disable TF32 for better compatibility
        
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (str(sgv_dir) + (":" + existing_pp if existing_pp else ""))

        log_file = logs_dir / f"train_gpu_forced_{int(time.time())}.log"
        
        print(f"📝 Detailed logs: {log_file}")
        print("🚀 Starting training process with GPU enforcement...")
        print("⏱️  Training progress:")
        print("=" * 80)
        
        start_time = time.time()
        line_count = 0
        last_progress_time = start_time
        
        try:
            # Start with maximum output flushing
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
                log_out.write(f"=== GPU-Forced MonoX Training Started: {time.ctime()} ===\n")
                log_out.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')}\n")
                log_out.write(f"Command: {' '.join(cmd)}\n")
                log_out.write("=" * 80 + "\n")
                
                for line in training_process.stdout:
                    line_count += 1
                    current_time = time.time()
                    
                    # Always print the line immediately
                    print(line.rstrip())
                    sys.stdout.flush()
                    
                    # Save to log
                    log_out.write(line)
                    log_out.flush()
                    
                    # Show GPU status periodically
                    if current_time - last_progress_time >= 30:  # Every 30 seconds
                        elapsed = current_time - start_time
                        print(f"\n🔥 === GPU STATUS UPDATE: {elapsed:.0f}s elapsed ===")
                        
                        # Quick GPU check
                        try:
                            result = subprocess.run([
                                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader,nounits'
                            ], capture_output=True, text=True, timeout=5)
                            
                            if result.returncode == 0:
                                parts = result.stdout.strip().split(', ')
                                if len(parts) >= 3:
                                    gpu_util, mem_used, mem_total = parts
                                    mem_percent = (int(mem_used) / int(mem_total)) * 100
                                    print(f"🔥 GPU: {gpu_util}% util | {mem_percent:.1f}% memory ({mem_used}/{mem_total}MB)")
                                else:
                                    print("🔥 GPU: Status unavailable")
                            else:
                                print("🔥 GPU: nvidia-smi failed")
                        except:
                            print("🔥 GPU: Status check error")
                        
                        print(f"📊 Lines processed: {line_count}")
                        print("=" * 30)
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