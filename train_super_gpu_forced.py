#!/usr/bin/env python3
"""
Super GPU-Forced MonoX Training Script
Aggressively forces GPU usage at every level.
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

def super_force_gpu_environment():
    """Aggressively force GPU usage at system level."""
    print("🔥 SUPER GPU FORCING MODE ACTIVATED!")
    
    # Set all possible GPU environment variables
    gpu_env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_USE_CUDA_DSA': '1',
        'NVIDIA_TF32_OVERRIDE': '0',
        'CUDA_CACHE_DISABLE': '0',
        'TORCH_CUDA_ARCH_LIST': '7.5;8.0;8.6',
        'FORCE_CUDA': '1',
        'USE_CUDA': '1'
    }
    
    for key, value in gpu_env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    # Force PyTorch to use GPU immediately
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Allocate a small tensor to "warm up" GPU
        warmup = torch.randn(100, 100, device='cuda')
        _ = torch.mm(warmup, warmup)
        print(f"   ✅ GPU warmup successful: {warmup.device}")
        del warmup
        torch.cuda.empty_cache()
    else:
        print("   ❌ CUDA not available - cannot force GPU!")
        return False
    
    return True

def verify_gpu_working():
    """Verify GPU is actually working before training."""
    print("\n🔍 AGGRESSIVE GPU VERIFICATION:")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available!")
        return False
    
    try:
        # Test GPU memory allocation
        device = torch.device('cuda:0')
        print(f"   Testing device: {device}")
        
        # Allocate significant GPU memory to verify it's working
        test_size = 1024
        a = torch.randn(test_size, test_size, device=device, dtype=torch.float32)
        b = torch.randn(test_size, test_size, device=device, dtype=torch.float32)
        
        # Perform computation
        c = torch.mm(a, b)
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"   ✅ GPU memory allocated: {allocated:.1f} MB")
        
        if allocated < 10:
            print("   ⚠️  Low GPU memory usage - GPU might not be working properly")
            return False
        
        # Clean up
        del a, b, c
        torch.cuda.empty_cache()
        
        print("   ✅ GPU verification passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ GPU verification failed: {e}")
        return False

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
    """Launch MonoX training with super aggressive GPU forcing."""
    
    global training_process
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 SUPER GPU-FORCED TRAINING MODE")
    print("=" * 80)
    
    # Super aggressive GPU forcing
    if not super_force_gpu_environment():
        print("❌ Failed to force GPU environment! Exiting...")
        sys.exit(1)
    
    # Verify GPU is actually working
    if not verify_gpu_working():
        print("❌ GPU verification failed! Exiting...")
        sys.exit(1)
    
    print("=" * 80)
    
    repo_root = Path.cwd()
    sgv_dir = _ensure_styleganv_repo(repo_root)
    print(f"✅ StyleGAN-V ready at: {sgv_dir}")
    
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
    
    training = cfg.get("training", {})
    total_kimg = training.get("total_kimg", 3000)
    snapshot_kimg = training.get("snapshot_kimg", 250)
    
    visualizer = cfg.get("visualizer", {})
    save_every_kimg = visualizer.get("save_every_kimg", 50)
    output_dir = visualizer.get("output_dir", "previews")
    
    sampling = cfg.get("sampling", {})
    truncation_psi = sampling.get("truncation_psi", 1.0)

    print(f"📊 SUPER GPU-FORCED Training Configuration:")
    print(f"   🗂️  Dataset: {dataset_path}")
    print(f"   🖼️  Resolution: {resolution}x{resolution}")
    print(f"   🎯 Training: {total_kimg} kimg total")
    print(f"   💾 Snapshots: Every {snapshot_kimg} kimg")
    print(f"   🎨 Previews: Every {save_every_kimg} kimg → {output_dir}/")
    print(f"   🔥 GPUs: {num_gpus} (SUPER FORCED)")
    print(f"   🎲 Truncation: {truncation_psi}")
    print("=" * 80)

    if launcher == "stylegan":
        # Build command with SUPER aggressive GPU parameters
        cmd = [
            sys.executable, "-m", "src.infra.launch",
            f"hydra.run.dir=logs",
            f"exp_suffix={exp_suffix}",
            f"dataset.path={dataset_path}",
            f"dataset.resolution={resolution}",
            f"training.kimg={total_kimg}",
            f"training.snap={snapshot_kimg}",
            f"num_gpus={num_gpus}",
            f"+training.gpus={num_gpus}",  # ADD training.gpus (Hydra suggested + prefix)
            f"gpus={num_gpus}",  # Also set top-level for safety
            f"+training.batch_size=8",  # ADD training.batch_size (+ prefix to append)
            f"+training.fp32=false",  # ADD training.fp32 (+ prefix to append)
            f"+training.nobench=false",  # ADD training.nobench (+ prefix to append)
            f"+training.allow_tf32=false",  # ADD training.allow_tf32 (+ prefix to append)
            f"+training.metrics=[fid50k_full]",  # ADD training.metrics (required parameter)
            f"+training.seed=0",  # ADD training.seed (required parameter)
            f"+training.data={dataset_path}",  # ADD training.data (required parameter)
            f"+model.loss_kwargs.style_mixing_prob=0.0",  # ADD model.loss_kwargs (required parameter)
            f"+training.num_workers=8",  # ADD training.num_workers (required parameter)
            f"+training.subset=null",  # ADD training.subset (required parameter)
            f"+training.mirror=true",  # ADD training.mirror (required parameter)
            f"visualizer.save_every_kimg={save_every_kimg}",
            f"visualizer.output_dir={output_dir}",
            f"sampling.truncation_psi={truncation_psi}"
        ]
        
        print("🔥 SUPER GPU-FORCED StyleGAN-V Command:")
        print("   " + " \\\n   ".join(cmd))
        print("=" * 80)

        # SUPER aggressive GPU environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Add all our GPU forcing variables
        gpu_env_vars = {
            'CUDA_VISIBLE_DEVICES': '0',
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
            'CUDA_LAUNCH_BLOCKING': '1',
            'TORCH_USE_CUDA_DSA': '1',
            'NVIDIA_TF32_OVERRIDE': '0',
            'FORCE_CUDA': '1',
            'USE_CUDA': '1'
        }
        
        for key, value in gpu_env_vars.items():
            env[key] = value
        
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (str(sgv_dir) + (":" + existing_pp if existing_pp else ""))

        log_file = logs_dir / f"train_super_gpu_{int(time.time())}.log"
        
        print(f"📝 SUPER GPU logs: {log_file}")
        print("🔥 Starting SUPER GPU-FORCED training...")
        print("⏱️  GPU monitoring every 15 seconds:")
        print("=" * 80)
        
        start_time = time.time()
        line_count = 0
        last_progress_time = start_time
        last_gpu_check = start_time
        
        try:
            # Start training with aggressive GPU monitoring
            training_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=env, 
                cwd=str(sgv_dir),
                bufsize=0,
                universal_newlines=True
            )
            
            with open(log_file, "w", buffering=1) as log_out:
                log_out.write(f"=== SUPER GPU-FORCED Training: {time.ctime()} ===\n")
                log_out.write(f"GPU Environment: {gpu_env_vars}\n")
                log_out.write(f"Command: {' '.join(cmd)}\n")
                log_out.write("=" * 80 + "\n")
                
                for line in training_process.stdout:
                    line_count += 1
                    current_time = time.time()
                    
                    # Print all output immediately
                    print(line.rstrip())
                    sys.stdout.flush()
                    
                    # Save to log
                    log_out.write(line)
                    log_out.flush()
                    
                    # AGGRESSIVE GPU monitoring every 15 seconds
                    if current_time - last_gpu_check >= 15:
                        elapsed = current_time - start_time
                        print(f"\n🔥 === SUPER GPU STATUS: {elapsed:.0f}s ===")
                        
                        # Check GPU usage with detailed info
                        try:
                            result = subprocess.run([
                                'nvidia-smi', 
                                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                '--format=csv,noheader,nounits'
                            ], capture_output=True, text=True, timeout=5)
                            
                            if result.returncode == 0:
                                parts = result.stdout.strip().split(', ')
                                if len(parts) >= 5:
                                    gpu_util, mem_used, mem_total, temp, power = parts
                                    mem_percent = (int(mem_used) / int(mem_total)) * 100
                                    print(f"🔥 GPU: {gpu_util}% util | {mem_percent:.1f}% mem ({mem_used}/{mem_total}MB) | {temp}°C | {power}W")
                                    
                                    # Alert if GPU usage is low
                                    if int(gpu_util) < 50:
                                        print("⚠️  WARNING: LOW GPU UTILIZATION!")
                                    if int(mem_used) < 1000:
                                        print("⚠️  WARNING: LOW GPU MEMORY USAGE!")
                                else:
                                    print("🔥 GPU: Status format error")
                            else:
                                print("🔥 GPU: nvidia-smi failed")
                        except Exception as e:
                            print(f"🔥 GPU: Error checking status: {e}")
                        
                        print(f"📊 Lines: {line_count} | Time: {elapsed:.0f}s")
                        print("=" * 40)
                        last_gpu_check = current_time
                
                ret = training_process.wait()
                
        except KeyboardInterrupt:
            print("\n🛑 SUPER GPU training interrupted!")
            elapsed = time.time() - start_time
            print(f"⏱️  Training ran for {elapsed:.1f} seconds")
            return
        except Exception as e:
            print(f"\n❌ SUPER GPU training error: {e}")
            return
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"⏱️  SUPER GPU training duration: {elapsed:.1f} seconds")
        print(f"📊 Total output lines: {line_count}")
        
        if ret == 0:
            print("🎉 SUPER GPU training completed successfully!")
        else:
            print(f"❌ SUPER GPU training failed with exit code {ret}")
            print(f"📝 Check logs: {log_file}")
            sys.exit(ret)

    else:
        print(f"❌ Unknown launcher: {launcher}")
        sys.exit(1)

if __name__ == "__main__":
    main()