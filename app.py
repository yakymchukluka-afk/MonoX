#!/usr/bin/env python3
"""
MonoX Training Interface - Minimal Version
Designed to work even with HF Spaces infrastructure issues
"""

import gradio as gr
import os
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Minimal environment setup."""
    # Create directories
    Path("previews").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    return "✅ Environment ready"

def check_training_status():
    """Check if training is running."""
    try:
        # Look for either CPU or GPU training scripts
        result = subprocess.run(
            ['bash', '-lc', "pgrep -f 'simple_gan_training.py|gpu_gan_training.py'"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return "🟢 Training is RUNNING"
        else:
            return "🔴 Training is STOPPED"
    except Exception:
        return "❓ Status unknown"

def get_progress_info():
    """Get current progress information."""
    preview_dir = Path("previews")
    checkpoint_dir = Path("checkpoints")
    
    previews = len(list(preview_dir.glob("*.png"))) if preview_dir.exists() else 0
    checkpoints = len(list(checkpoint_dir.glob("*.pth"))) if checkpoint_dir.exists() else 0
    
    return f"""
📊 **MonoX Training Progress**

**Status**: {check_training_status()}
**Previews**: {previews} samples generated
**Checkpoints**: {checkpoints} saved
**Progress**: {previews}/50 epochs ({previews*2}%)

**Hardware Options**:
- CPU: ~15 min/epoch (current)
- GPU T4: ~30 sec/epoch (30x faster!)
- GPU A10G: ~15 sec/epoch (60x faster!)

**Cost for GPU T4**: Only $0.25 for complete training!
**Time savings**: 12+ hours → 25 minutes
    """

def start_cpu_training():
    """Start CPU training."""
    try:
        subprocess.Popen(['python3', 'simple_gan_training.py'])
        return "🚀 CPU training started! Check back in 15 minutes for next sample."
    except Exception as e:
        return f"❌ Failed to start training: {e}"

def start_gpu_training():
    """Start GPU training if available."""
    try:
        import torch
        if torch.cuda.is_available():
            subprocess.Popen(['python3', 'gpu_gan_training.py'])
            return "🚀 GPU training started! Much faster - check back in 30 seconds!"
        else:
            return "⚠️ No GPU detected. Upgrade hardware in Space settings first."
    except Exception as e:
        return f"❌ Failed to start GPU training: {e}"

def get_latest_sample():
    """Get the latest generated sample."""
    preview_dir = Path("previews")
    if not preview_dir.exists():
        return None, "No samples generated yet"
    
    # Support both CPU and GPU training sample naming
    patterns = ["samples_epoch_*.png", "gpu_samples_epoch_*.png"]
    samples = []
    for pattern in patterns:
        samples.extend(preview_dir.glob(pattern))
    samples = sorted(samples, key=lambda x: x.stat().st_mtime)
    if not samples:
        return None, "No samples found"
    
    latest = samples[-1]
    try:
        epoch_num = int(latest.stem.split('_')[-1])
    except Exception:
        epoch_num = -1
    size_mb = latest.stat().st_size / (1024*1024)
    
    return str(latest), f"Epoch {epoch_num} - {size_mb:.1f}MB"

# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="MonoX Training", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 MonoX StyleGAN-V Training")
        gr.Markdown("*Generate monotype-inspired artwork using AI*")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 Status & Control")
                
                status_output = gr.Textbox(
                    label="Training Status",
                    value=check_training_status(),
                    interactive=False
                )
                
                progress_output = gr.Markdown(
                    value=get_progress_info()
                )
                
                with gr.Row():
                    cpu_btn = gr.Button("🖥️ Start CPU Training", variant="secondary")
                    gpu_btn = gr.Button("🚀 Start GPU Training", variant="primary")
                
                result_output = gr.Textbox(
                    label="Action Result",
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("## 🖼️ Latest Sample")
                
                sample_image = gr.Image(
                    label="Generated Artwork",
                    type="filepath"
                )
                
                sample_info = gr.Textbox(
                    label="Sample Info",
                    interactive=False
                )
                
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        
        # Event handlers
        cpu_btn.click(
            fn=start_cpu_training,
            outputs=result_output
        )
        
        gpu_btn.click(
            fn=start_gpu_training,
            outputs=result_output
        )
        
        def refresh_all():
            status = check_training_status()
            progress = get_progress_info()
            sample_path, sample_desc = get_latest_sample()
            return status, progress, sample_path, sample_desc
        
        refresh_btn.click(
            fn=refresh_all,
            outputs=[status_output, progress_output, sample_image, sample_info]
        )
        
        # Auto-refresh every 10 seconds for CPU visibility
        interface.load(
            fn=refresh_all,
            outputs=[status_output, progress_output, sample_image, sample_info],
            every=10
        )
    
    return interface

def main():
    """Main application."""
    print("🎨 MonoX Training Interface Starting...")
    
    # Setup
    setup_result = setup_environment()
    print(setup_result)
    
    # Optionally auto-start training on Space boot
    auto_start = os.environ.get("AUTO_START_TRAINING", "").strip().lower() in {"1", "true", "yes", "on"}
    if auto_start:
        try:
            import torch
            if torch.cuda.is_available():
                print("🚀 AUTO_START_TRAINING=1 detected: starting GPU training...")
                start_gpu_training()
            else:
                print("🚀 AUTO_START_TRAINING=1 detected: starting CPU training...")
                start_cpu_training()
        except Exception as e:
            print(f"⚠️ Auto-start failed: {e}")
    
    # Create and launch interface
    interface = create_interface()
    port = int(os.environ.get("PORT", "7860"))
    interface.launch(server_name="0.0.0.0", server_port=port, share=False)

"""Expose Gradio Blocks for Spaces runner."""
demo = create_interface()

if __name__ == "__main__":
    main()
