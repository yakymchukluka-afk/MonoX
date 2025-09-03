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
        result = subprocess.run(['pgrep', '-f', 'gan_training'], capture_output=True, text=True)
        if result.stdout.strip():
            return "🟢 Training is RUNNING"
        else:
            return "🔴 Training is STOPPED"
    except:
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

def start_monox_training():
    """Start MonoX StyleGAN-V training with lukua/monox-dataset."""
    try:
        # Launch the MonoX training
        subprocess.Popen(['python3', 'launch_training_in_space.py'])
        return "🚀 MonoX training started! StyleGAN-V at 1024x1024 resolution with lukua/monox-dataset"
    except Exception as e:
        return f"❌ Failed to start MonoX training: {e}"

def validate_setup():
    """Validate training setup without authentication."""
    try:
        result = subprocess.run(['python3', 'validate_training_ready.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return "✅ Training setup validated - ready for MonoX training!"
        else:
            return f"❌ Setup validation failed: {result.stderr}"
    except Exception as e:
        return f"❌ Validation error: {e}"

def get_latest_sample():
    """Get the latest generated sample."""
    preview_dir = Path("previews")
    if not preview_dir.exists():
        return None, "No samples generated yet"
    
    samples = sorted(preview_dir.glob("samples_epoch_*.png"), key=lambda x: x.stat().st_mtime)
    if not samples:
        return None, "No samples found"
    
    latest = samples[-1]
    epoch_num = int(latest.stem.split('_')[-1])
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
                    validate_btn = gr.Button("🧪 Validate Setup", variant="secondary")
                    monox_btn = gr.Button("🎨 Start MonoX Training", variant="primary")
                
                result_output = gr.Textbox(
                    label="Action Result",
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("## 🖼️ Latest Sample")
                
                sample_path_init, sample_desc_init = get_latest_sample()
                sample_image = gr.Image(
                    label="Generated Artwork",
                    type="filepath",
                    value=sample_path_init
                )
                
                sample_info = gr.Textbox(
                    label="Sample Info",
                    interactive=False,
                    value=sample_desc_init
                )
                
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        
        # Event handlers
        validate_btn.click(
            fn=validate_setup,
            outputs=result_output
        )
        
        monox_btn.click(
            fn=start_monox_training,
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
        
        # Auto-refresh every 30 seconds (Gradio 4+ API)
        auto_timer = gr.Timer(30.0)
        auto_timer.tick(
            fn=refresh_all,
            outputs=[status_output, progress_output, sample_image, sample_info]
        )
    
    return interface

def main():
    """Main application."""
    print("🎨 MonoX Training Interface Starting...")
    
    # Setup
    setup_result = setup_environment()
    print(setup_result)
    
    # Launch interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

# Expose Gradio app for Hugging Face Spaces
demo = create_interface()

if __name__ == "__main__":
    main()