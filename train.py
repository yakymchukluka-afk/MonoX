#!/usr/bin/env python3
import os
import sys
import argparse
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def build_overrides(args):
  overrides = []
  if args.dataset:
    overrides.append(f"dataset.path={args.dataset}")
  if args.resolution:
    overrides.append(f"dataset.resolution={args.resolution}")
  if args.total_kimg:
    overrides.append(f"training.total_kimg={args.total_kimg}")
  if args.snapshot_kimg:
    overrides.append(f"training.snapshot_kimg={args.snapshot_kimg}")
  if args.truncation_psi is not None:
    overrides.append(f"sampling.truncation_psi={args.truncation_psi}")
  if args.num_gpus:
    overrides.append(f"num_gpus={args.num_gpus}")
  if args.logs:
    overrides.append(f"training.log_dir={args.logs}")
  if args.previews:
    overrides.append(f"training.preview_dir={args.previews}")
  if args.checkpoints:
    overrides.append(f"training.checkpoint_dir={args.checkpoints}")
  if args.resume:
    overrides.append(f"training.resume={args.resume}")
  return overrides


def main():
  parser = argparse.ArgumentParser(description="MonoX StyleGAN-V training wrapper (Hydra)")
  parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides (k=v)")
  parser.add_argument("--dataset", type=str, default=os.environ.get("DATASET_DIR", ""), help="Dataset path")
  parser.add_argument("--resolution", type=int, default=1024)
  parser.add_argument("--total-kimg", type=int, default=3000)
  parser.add_argument("--snapshot-kimg", type=int, default=250)
  parser.add_argument("--truncation-psi", type=float, default=1.0)
  parser.add_argument("--num-gpus", type=int, default=1)
  parser.add_argument("--logs", type=str, default=os.environ.get("LOGS_DIR", "logs"))
  parser.add_argument("--previews", type=str, default=os.environ.get("PREVIEWS_DIR", "previews"))
  parser.add_argument("--checkpoints", type=str, default=os.environ.get("CKPT_DIR", "checkpoints"))
  parser.add_argument("--resume", type=str, default="")

  args = parser.parse_args()

  if not args.dataset:
    print("Dataset path is required. Supply --dataset or set DATASET_DIR env var.\n")
    parser.print_help()
    sys.exit(2)

  launcher = REPO_ROOT / "src" / "infra" / "launch.py"
  if not launcher.exists():
    print(f"ERROR: Launcher not found at {launcher}")
    sys.exit(3)

  cmd = [sys.executable, str(launcher)] + build_overrides(args) + list(args.overrides)
  print("Running:", " ".join(shlex.quote(x) for x in cmd))
  env = os.environ.copy()
  env["PYTHONUNBUFFERED"] = "1"
  sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
  main()