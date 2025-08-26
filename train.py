#!/usr/bin/env python3
"""
MonoX Training Script - Single Entrypoint

This is the main entrypoint for MonoX training using StyleGAN-V.
It loads Hydra configs from configs/ and delegates to the StyleGAN-V submodule.

Usage:
    python train.py dataset=ffs training.total_kimg=3000
    python train.py -cp configs -cn config dataset=ffs training.steps=10
"""

import os
import sys
import subprocess
import time
import glob
from pathlib import Path
from typing import Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint file."""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    patterns: List[str] = ["*.pkl", "*.pt", "*.ckpt", "*.pth"]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _ensure_styleganv_repo(repo_root: Path) -> Path:
    """Ensure StyleGAN-V submodule is available and up to date."""
    external_dir = repo_root / ".external"
    external_dir.mkdir(parents=True, exist_ok=True)
    sgv_dir = external_dir / "stylegan-v"

    if (sgv_dir / ".git").is_file() or (sgv_dir / ".git").is_dir():
        # Try to update
        try:
            subprocess.run(["git", "-C", str(sgv_dir), "pull", "--ff-only"], check=False)
        except Exception:
            pass
        return sgv_dir

    # Clone if missing
    repo_url = "https://github.com/universome/stylegan-v"
    print(f"Cloning StyleGAN-V from {repo_url} into {sgv_dir}...")
    subprocess.run(["git", "clone", repo_url, str(sgv_dir)], check=True)
    return sgv_dir


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint using Hydra configuration."""
    # Resolve repo root from this file location, not from Hydra's runtime cwd
    repo_root = Path(__file__).resolve().parent

    # Normalize dataset configuration to handle both string and dict formats
    # This fixes the AttributeError when using dataset=ffs vs dataset.name=ffs
    dataset_cfg = cfg.get("dataset", {})
    if isinstance(dataset_cfg, str):
        # Case: dataset=ffs - Hydra replaces entire dataset config with string
        # Normalize to dict format: {"name": "ffs"}
        dataset_cfg = {"name": dataset_cfg}
        dataset_name = dataset_cfg.get("name")
        dataset_path = os.environ.get("DATASET_DIR", "")
        resolution = 1024
    else:
        # Case: dataset.name=ffs - Hydra preserves dict structure
        # Handle DictConfig or regular dict - access directly without conversion
        dataset_name = dataset_cfg.get("name") if dataset_cfg else None
        dataset_path = str(dataset_cfg.get("path", "") or os.environ.get("DATASET_DIR", "")) if dataset_cfg else os.environ.get("DATASET_DIR", "")
        resolution = int(dataset_cfg.get("resolution", 1024)) if dataset_cfg else 1024
    
    # Backfill defaults for known datasets
    if dataset_name in ["ffs", "mead", "ucf101"]:
        # Update resolution if not set
        if not resolution or resolution == 1024:
            resolution = 1024 if dataset_name == "ffs" else 256

    total_kimg = int(cfg.get("training", {}).get("total_kimg", 3000))
    snapshot_kimg = int(cfg.get("training", {}).get("snapshot_kimg", 250))
    num_gpus = int(cfg.get("training", {}).get("num_gpus", 1))
    
    # Additional training parameters for smoke testing
    steps = cfg.get("training", {}).get("steps")
    batch = cfg.get("training", {}).get("batch")
    num_workers = cfg.get("training", {}).get("num_workers")
    fp16 = cfg.get("training", {}).get("fp16")
    launcher_mode = cfg.get("launcher", "stylegan")

    logs_dir = str(cfg.get("training", {}).get("log_dir") or os.environ.get("LOGS_DIR", "logs"))
    previews_dir = str(cfg.get("training", {}).get("preview_dir") or os.environ.get("PREVIEWS_DIR", "previews"))
    ckpt_dir = str(cfg.get("training", {}).get("checkpoint_dir") or os.environ.get("CKPT_DIR", "checkpoints"))

    truncation_psi = float(cfg.get("sampling", {}).get("truncation_psi", 1.0))

    preview_every_kimg = int(cfg.get("visualizer", {}).get("save_every_kimg", 50))

    resume_cfg = str(cfg.get("training", {}).get("resume") or "").strip()

    # Ensure output directories
    for d in [logs_dir, previews_dir, ckpt_dir]:
        _ensure_dir(d)

    # Determine resume file: config or auto-detect latest
    resume_path = resume_cfg if resume_cfg else _find_latest_checkpoint(ckpt_dir)
    if resume_path:
        print(f"Will resume from: {resume_path}")

    # Skip dataset validation for local launcher mode
    if launcher_mode != "local" and (not dataset_path or not os.path.exists(dataset_path)):
        print("ERROR: dataset.path is not set or does not exist.")
        print("Set it in configs/config.yaml or via env var DATASET_DIR or Hydra override: dataset.path=/path/to/data")
        sys.exit(2)

    # Ensure StyleGAN-V is available locally and get its launcher path
    sgv_dir = _ensure_styleganv_repo(repo_root)
    sgv_launcher_module = "src.infra.launch"

    # Handle special launcher modes
    if launcher_mode == "local":
        print("🔧 Local launcher mode - running configuration validation")
        print("=" * 50)
        print(f"📁 Dataset: {dataset_name or 'default'}")
        print(f"🏋️  Training steps: {steps if steps is not None else f'{total_kimg}k images'}")
        print(f"📊 Batch size: {batch if batch is not None else 'default'}")
        print(f"🔄 Workers: {num_workers if num_workers is not None else 'default'}")
        print(f"⚡ FP16: {fp16 if fp16 is not None else 'default'}")
        print(f"📝 Logs: {logs_dir}")
        print(f"🖼️  Previews: {previews_dir}")
        print(f"💾 Checkpoints: {ckpt_dir}")
        print("=" * 50)
        print("✅ Configuration validation passed!")
        print("💡 To run actual training, change launcher=stylegan")
        return
    
    # Build Hydra overrides for StyleGAN-V
    cmd = [
        sys.executable, "-m", sgv_launcher_module,
        f"hydra.run.dir={logs_dir}",
        "exp_suffix=monox",
        f"dataset.path={dataset_path}",
        f"dataset.resolution={resolution}",
    ]
    
    # Use steps if specified, otherwise use total_kimg
    if steps is not None:
        cmd.append(f"training.kimg={steps}")  # StyleGAN-V uses 'kimg' parameter
    else:
        cmd.append(f"training.kimg={total_kimg}")
    
    cmd.extend([
        f"training.snap={snapshot_kimg}",
        f"visualizer.save_every_kimg={preview_every_kimg}",
        f"visualizer.output_dir={previews_dir}",
        f"sampling.truncation_psi={truncation_psi}",
        f"num_gpus={num_gpus}",
    ])
    
    # Add optional parameters if specified
    if batch is not None:
        cmd.append(f"training.batch_size={batch}")
    if num_workers is not None:
        cmd.append(f"training.num_workers={num_workers}")
    if fp16 is not None:
        if fp16:
            cmd.append("training.fp32=false")
        else:
            cmd.append("training.fp32=true")
    if resume_path:
        cmd.append(f"training.resume={resume_path}")

    print("Launching StyleGAN-V with command:\n" + " ".join(cmd))

    # Stream logs to file in Drive
    log_file = os.path.join(logs_dir, "train.log")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure src imports work when running as module
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(sgv_dir) + (":" + existing_pp if existing_pp else ""))

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(sgv_dir)) as proc:
        with open(log_file, "a", buffering=1) as out:
            for line in proc.stdout:  # type: ignore[attr-defined]
                sys.stdout.write(line)
                out.write(line)
        ret = proc.wait()
    print(f"Training process exited with code {ret}")
    if ret != 0:
        sys.exit(ret)


if __name__ == "__main__":
    main()