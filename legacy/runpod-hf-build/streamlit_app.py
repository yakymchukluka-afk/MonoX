#!/usr/bin/env python3
"""
MonoX Training Interface - Streamlit Version
Alternative to Gradio for HF Spaces compatibility
"""

import streamlit as st
import subprocess
import os
import time
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="MonoX Training Interface",
    page_icon="🎨",
    layout="wide"
)

def setup_environment():
    """Setup environment variables and paths."""
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['HF_HOME'] = '/app/.cache/huggingface'
    
    # Create directories
    Path("/app/previews").mkdir(exist_ok=True)
    Path("/app/checkpoints").mkdir(exist_ok=True)
    Path("/app/logs").mkdir(exist_ok=True)

def check_training_status():
    """Check if training is currently running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'simple_gan_training'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def start_training():
    """Start the training process."""
    try:
        # Run pre-build setup first
        subprocess.run(['python', 'pre_build.py'], check=True)
        
        # Start training in background
        subprocess.Popen(['python', 'simple_gan_training.py'])
        return True
    except Exception as e:
        st.error(f"Failed to start training: {e}")
        return False

def main():
    """Main Streamlit application."""
    st.title("🎨 MonoX Training Interface")
    st.markdown("---")
    
    # Run setup
    setup_environment()
    
    # Check training status
    is_training = check_training_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Training Status")
        if is_training:
            st.success("✅ Training is running")
        else:
            st.warning("⚠️ Training is not running")
        
        if st.button("🚀 Start Training", disabled=is_training):
            if start_training():
                st.success("Training started!")
                st.rerun()
    
    with col2:
        st.subheader("🖼️ Latest Outputs")
        
        # Show latest preview images
        preview_dir = Path("/app/previews")
        if preview_dir.exists():
            preview_files = sorted(preview_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            for i, preview_file in enumerate(preview_files[:3]):
                st.image(str(preview_file), caption=f"Latest Sample: {preview_file.name}", width=300)
                if i >= 2:  # Show max 3 images
                    break
        else:
            st.info("No preview images yet")
    
    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()