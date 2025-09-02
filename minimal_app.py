#!/usr/bin/env python3
import gradio as gr
import spaces
import torch
import os
import time
from pathlib import Path

print("🚀 MonoX Minimal App Starting...")

# Setup
Path("previews").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)

@spaces.GPU
def test_gpu():
    """Test GPU function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return f"GPU Test: {device}"

@spaces.GPU  
def run_training():
    """Run minimal training."""
    try:
        print("Starting training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test file
        test_file = f"previews/test_{int(time.time())}.txt"
        with open(test_file, 'w') as f:
            f.write(f"Training test at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try upload
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=test_file,
                path_in_repo=test_file,
                repo_id="lukua/monox",
                repo_type="model",
                token=hf_token
            )
            return f"✅ Training test completed! File uploaded: {test_file}"
        else:
            return f"✅ Training test completed! File saved: {test_file} (no token for upload)"
            
    except Exception as e:
        return f"❌ Training failed: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 MonoX Minimal Test")
    
    test_btn = gr.Button("🧪 Test GPU")
    train_btn = gr.Button("🚀 Run Training Test") 
    result = gr.Textbox(label="Result")
    
    test_btn.click(test_gpu, outputs=result)
    train_btn.click(run_training, outputs=result)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))