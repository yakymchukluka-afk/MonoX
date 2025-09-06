#!/usr/bin/env python3
"""
MonoX Training Script for RunPod with StyleGAN-V Integration
"""
import os
import sys
import signal
import yaml
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Add vendored stylegan-v to path
sys.path.insert(0, '/workspace/.external/stylegan-v/src')

from training import training_loop
from training.dataset import ImageFolderDataset
from training.networks import Generator, Discriminator
from training.loss import StyleGAN2Loss
from src.torch_utils import training_stats
from src.torch_utils import custom_ops

# Global state for signal handling
_last_state = {"model": None, "opt": None, "scaler": None, "step": 0, "outdir": None}

def _save_final_ckpt(signum, frame):
    """Save final checkpoint on graceful shutdown"""
    try:
        if _last_state["model"] is not None:
            outdir = Path(_last_state["outdir"])
            outdir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": _last_state["model"].state_dict(),
                "opt": _last_state["opt"].state_dict() if _last_state["opt"] else None,
                "scaler": _last_state["scaler"].state_dict() if _last_state["scaler"] else None,
                "step": _last_state["step"]
            }, outdir / f"monox_{_last_state['step']:08d}_final.pt")
            print(f"[signal {signum}] Saved final checkpoint at step {_last_state['step']}")
    finally:
        sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGTERM, _save_final_ckpt)
signal.signal(signal.SIGINT, _save_final_ckpt)

def load_config(config_path):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_torch_optimizations():
    """Configure PyTorch for optimal performance"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable channels_last memory format for better performance
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

def create_dataloader(config):
    """Create optimized dataloader for training"""
    dataset_config = config['dataset']
    
    # Create dataset
    dataset = ImageFolderDataset(
        path=dataset_config['path'],
        resolution=dataset_config['resolution'],
        use_labels=False
    )
    
    # Create dataloader with optimized settings
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=dataset_config['workers'],
        pin_memory=dataset_config['pin_memory'],
        persistent_workers=dataset_config['persistent_workers'],
        drop_last=True
    )
    
    return dataloader

def create_models(config):
    """Create generator and discriminator models"""
    model_config = config['model']
    training_config = config['training']
    
    # Create generator
    generator = Generator(
        z_dim=model_config['z_dim'],
        w_dim=model_config['w_dim'],
        img_resolution=model_config['img_resolution'],
        img_channels=model_config['img_channels'],
        synthesis_kwargs=model_config['synthesis_kwargs']
    )
    
    # Create discriminator
    discriminator = Discriminator(
        img_resolution=model_config['img_resolution'],
        img_channels=model_config['img_channels']
    )
    
    return generator, discriminator

def create_optimizers(generator, discriminator, config):
    """Create optimizers for generator and discriminator"""
    training_config = config['training']
    
    # Generator optimizer
    g_opt = torch.optim.Adam(
        generator.parameters(),
        lr=training_config['learning_rate'],
        betas=(training_config['beta1'], training_config['beta2'])
    )
    
    # Discriminator optimizer
    d_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=training_config['learning_rate'],
        betas=(training_config['beta1'], training_config['beta2'])
    )
    
    return g_opt, d_opt

def create_loss_function(config):
    """Create loss function"""
    training_config = config['training']
    
    loss = StyleGAN2Loss(
        device=torch.device('cuda'),
        lambda_gp=training_config['lambda_gp'],
        lambda_pl=training_config['lambda_pl']
    )
    
    return loss

def save_checkpoint(generator, discriminator, g_opt, d_opt, scaler, step, outdir):
    """Save training checkpoint"""
    checkpoint_dir = Path(outdir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"monox_{step:08d}.pt"
    
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_opt": g_opt.state_dict(),
        "d_opt": d_opt.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "step": step
    }, checkpoint_path)
    
    print(f"[checkpoint] Saved checkpoint at step {step}")

def save_samples(generator, step, outdir, num_samples=16):
    """Save sample images"""
    samples_dir = Path(outdir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, generator.z_dim, device='cuda')
        samples = generator(z, None)
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        
        # Save as grid
        from torchvision.utils import save_image
        save_image(samples, samples_dir / f"samples_{step:08d}.png", nrow=4, padding=2)
    
    generator.train()
    print(f"[samples] Saved samples at step {step}")

def find_latest_checkpoint(outdir):
    """Find the latest checkpoint for resuming training"""
    checkpoint_dir = Path(outdir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("monox_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
    return checkpoints[-1]

def load_checkpoint(checkpoint_path, generator, discriminator, g_opt, d_opt, scaler):
    """Load checkpoint and return step number"""
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    g_opt.load_state_dict(checkpoint['g_opt'])
    d_opt.load_state_dict(checkpoint['d_opt'])
    
    if scaler and checkpoint.get('scaler'):
        scaler.load_state_dict(checkpoint['scaler'])
    
    step = checkpoint['step']
    print(f"[resume] Loaded checkpoint from step {step}")
    return step

def main():
    parser = argparse.ArgumentParser(description='Train MonoX model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup PyTorch optimizations
    setup_torch_optimizations()
    
    # Create output directories
    outdir = config['training']['outdir']
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(outdir, 'checkpoints').mkdir(parents=True, exist_ok=True)
    Path(outdir, 'samples').mkdir(parents=True, exist_ok=True)
    Path(outdir, 'logs').mkdir(parents=True, exist_ok=True)
    
    # Create dataloader
    print("[setup] Creating dataloader...")
    dataloader = create_dataloader(config)
    
    # Create models
    print("[setup] Creating models...")
    generator, discriminator = create_models(config)
    
    # Move to GPU
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    
    # Enable channels_last memory format
    if config['performance']['channels_last']:
        generator = generator.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)
    
    # Create optimizers
    print("[setup] Creating optimizers...")
    g_opt, d_opt = create_optimizers(generator, discriminator, config)
    
    # Create loss function
    loss_fn = create_loss_function(config)
    
    # Create scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if config['performance']['amp'] else None
    
    # Compile models if requested
    if config['performance']['torch_compile']:
        print("[setup] Compiling models with torch.compile...")
        generator = torch.compile(generator, mode=config['performance']['compile_mode'])
        discriminator = torch.compile(discriminator, mode=config['performance']['compile_mode'])
    
    # Resume from checkpoint if available
    start_step = 0
    latest_checkpoint = find_latest_checkpoint(outdir)
    if latest_checkpoint:
        start_step = load_checkpoint(latest_checkpoint, generator, discriminator, g_opt, d_opt, scaler)
    
    # Update global state for signal handling
    _last_state.update({
        "model": generator,
        "opt": g_opt,
        "scaler": scaler,
        "step": start_step,
        "outdir": Path(outdir) / "checkpoints"
    })
    
    print(f"[training] Starting training from step {start_step}")
    print(f"[training] Batch size: {config['training']['batch_size']}")
    print(f"[training] Learning rate: {config['training']['learning_rate']}")
    print(f"[training] AMP: {config['performance']['amp']}")
    print(f"[training] Channels last: {config['performance']['channels_last']}")
    print(f"[training] Torch compile: {config['performance']['torch_compile']}")
    
    # Training loop
    step = start_step
    checkpoint_interval = config['training']['checkpoint_interval']
    sample_interval = config['training']['sample_interval']
    
    for batch_idx, (real_images, _) in enumerate(dataloader):
        step += 1
        
        # Move to GPU and channels_last if enabled
        real_images = real_images.cuda()
        if config['performance']['channels_last']:
            real_images = real_images.to(memory_format=torch.channels_last)
        
        # Training step
        if config['performance']['amp']:
            with torch.cuda.amp.autocast():
                loss_g, loss_d = loss_fn(generator, discriminator, real_images)
        else:
            loss_g, loss_d = loss_fn(generator, discriminator, real_images)
        
        # Update generator
        g_opt.zero_grad()
        if scaler:
            scaler.scale(loss_g).backward()
            scaler.step(g_opt)
            scaler.update()
        else:
            loss_g.backward()
            g_opt.step()
        
        # Update discriminator
        d_opt.zero_grad()
        if scaler:
            scaler.scale(loss_d).backward()
            scaler.step(d_opt)
            scaler.update()
        else:
            loss_d.backward()
            d_opt.step()
        
        # Update global state
        _last_state["step"] = step
        
        # Logging
        if step % config['logging']['log_interval'] == 0:
            print(f"[step {step}] G_loss: {loss_g.item():.4f}, D_loss: {loss_d.item():.4f}")
        
        # Save checkpoint
        if step % checkpoint_interval == 0:
            save_checkpoint(generator, discriminator, g_opt, d_opt, scaler, step, outdir)
        
        # Save samples
        if step % sample_interval == 0:
            save_samples(generator, step, outdir)

if __name__ == "__main__":
    main()