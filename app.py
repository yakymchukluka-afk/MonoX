#!/usr/bin/env python3
"""
MonoX Training Interface - HF Spaces Compatible
Main application file that handles git configuration issues and starts training.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def fix_git_config():
    """Fix git configuration issues at startup."""
    try:
        # Try to create a local git config
        home_dir = os.path.expanduser("~")
        os.makedirs(home_dir, exist_ok=True)
        
        gitconfig_path = os.path.join(home_dir, ".gitconfig")
        with open(gitconfig_path, "w") as f:
            f.write("[user]\n")
            f.write("    email = lukua@users.noreply.huggingface.co\n")
            f.write("    name = lukua\n")
            f.write("[safe]\n")
            f.write("    directory = /app\n")
            f.write("    directory = /workspace\n")
        
        print(f"✅ Git config created at {gitconfig_path}")
        return True
    except Exception as e:
        print(f"⚠️ Could not create git config: {e}")
        return False

def setup_environment():
    """Setup environment for training."""
    # Fix git config first
    fix_git_config()
    
    # Setup environment variables
    env_vars = {
        "PYTHONPATH": "/app/.external/stylegan-v:/app",
        "PYTHONUNBUFFERED": "1",
        "HOME": os.path.expanduser("~")
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Create directories
    directories = [
        "/app/logs",
        "/app/checkpoints", 
        "/app/previews",
        "/app/training_output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Environment setup completed")

def check_training_capability():
    """Check if training can be started."""
    try:
        import torch
        import gradio as gr
        from huggingface_hub import whoami
        
        # Check HF authentication
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            try:
                user_info = whoami(token=hf_token)
                print(f"✅ HF authenticated as: {user_info['name']}")
            except Exception as e:
                print(f"⚠️ HF authentication issue: {e}")
        else:
            print("⚠️ HF_TOKEN not found - set as Space secret")
        
        # Check PyTorch
        print(f"✅ PyTorch {torch.__version__} available")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check Gradio
        print(f"✅ Gradio {gr.__version__} available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_simple_interface():
    """Create a simple Gradio interface for MonoX training."""
    
    try:
        import gradio as gr
        import json
        
        def get_system_status():
            """Get current system status."""
            try:
                import torch
                from huggingface_hub import whoami
                
                hf_token = os.environ.get('HF_TOKEN')
                hf_status = "Not configured"
                
                if hf_token:
                    try:
                        user_info = whoami(token=hf_token)
                        hf_status = f"Authenticated as {user_info['name']}"
                    except:
                        hf_status = "Token invalid"
                
                status = {
                    "pytorch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "hf_authentication": hf_status,
                    "workspace_ready": os.path.exists("/app"),
                    "training_files": {
                        "simple_gan_training.py": os.path.exists("simple_gan_training.py"),
                        "monitor_training.py": os.path.exists("monitor_training.py")
                    }
                }
                
                return json.dumps(status, indent=2)
                
            except Exception as e:
                return f"Error getting status: {e}"
        
        def start_training():
            """Start the training process."""
            try:
                # Check if HF_TOKEN is available
                if not os.environ.get('HF_TOKEN'):
                    return "❌ HF_TOKEN not found. Please set it as a Space secret."
                
                # Check if training script exists
                if not os.path.exists("simple_gan_training.py"):
                    return "❌ Training script not found."
                
                # Start training in background
                process = subprocess.Popen(
                    [sys.executable, "simple_gan_training.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                return f"✅ Training started successfully! Process ID: {process.pid}"
                
            except Exception as e:
                return f"❌ Failed to start training: {e}"
        
        def check_training_progress():
            """Check training progress."""
            try:
                # Check if training process is running
                result = subprocess.run(
                    ["ps", "aux"],
                    capture_output=True,
                    text=True
                )
                
                training_running = any("simple_gan_training.py" in line for line in result.stdout.split('\n'))
                
                # Count outputs
                checkpoints = len(list(Path("/app/checkpoints").glob("*.pt")))
                previews = len(list(Path("/app/previews").glob("*.png")))
                
                progress = {
                    "training_active": training_running,
                    "checkpoints_saved": checkpoints,
                    "previews_generated": previews,
                    "last_check": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return json.dumps(progress, indent=2)
                
            except Exception as e:
                return f"Error checking progress: {e}"
        
        # Create interface
        with gr.Blocks(title="MonoX Training Interface") as demo:
            gr.Markdown("# 🎨 MonoX Training Interface")
            gr.Markdown("Fresh training from scratch using the 1024px monotype dataset")
            
            with gr.Tab("System Status"):
                status_btn = gr.Button("Check System Status")
                status_output = gr.Textbox(label="System Status", lines=10)
                status_btn.click(get_system_status, outputs=status_output)
            
            with gr.Tab("Training Control"):
                gr.Markdown("### Start Fresh Training")
                gr.Markdown("This will train a GAN on your monotype dataset with automatic checkpoint saving every 5 epochs.")
                
                start_btn = gr.Button("🚀 Start Training", variant="primary")
                start_output = gr.Textbox(label="Training Status")
                start_btn.click(start_training, outputs=start_output)
            
            with gr.Tab("Progress Monitoring"):
                progress_btn = gr.Button("Check Training Progress")
                progress_output = gr.Textbox(label="Training Progress", lines=8)
                progress_btn.click(check_training_progress, outputs=progress_output)
            
            with gr.Tab("Instructions"):
                gr.Markdown("""
                ### 🎯 MonoX Fresh Training Setup
                
                **Current Status**: Ready for fresh training from scratch
                
                **What this does:**
                - Trains a GAN on 868 monotype images at 1024x1024 resolution
                - Saves checkpoints every 5 epochs to lukua/monox model repo
                - Generates preview images every epoch
                - Provides comprehensive logging and monitoring
                
                **To start training:**
                1. Make sure HF_TOKEN is set as a Space secret
                2. Click "Start Training" in the Training Control tab
                3. Monitor progress in the Progress Monitoring tab
                4. Check lukua/monox model repo for outputs
                
                **Security Note**: All tokens are handled securely via environment variables.
                """)
        
        return demo
        
    except ImportError:
        # Fallback if Gradio is not available
        print("❌ Gradio not available - creating simple HTTP interface")
        return None

def main():
    """Main application entry point."""
    print("🎨 MonoX Training Interface Starting...")
    print("=" * 50)
    
    # Run pre-build setup
    try:
        from pre_build import main as pre_build_main
        pre_build_main()
    except Exception as e:
        print(f"⚠️ Pre-build setup warning: {e}")
    
    # Setup environment
    setup_environment()
    
    # Check capabilities
    if not check_training_capability():
        print("❌ Training capabilities check failed")
        # Continue anyway - might work in production
    
    # Create and launch interface
    demo = create_simple_interface()
    
    if demo:
        print("🚀 Launching Gradio interface...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    else:
        # Fallback - simple HTTP server
        print("🔄 Fallback: Starting simple HTTP server...")
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class SimpleHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                html = """
                <!DOCTYPE html>
                <html>
                <head><title>MonoX Training</title></head>
                <body>
                    <h1>🎨 MonoX Training Interface</h1>
                    <p>Training interface is starting...</p>
                    <p>Fresh training from scratch using 1024px dataset</p>
                    <p>Outputs will be uploaded to lukua/monox model repo</p>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
        
        server = HTTPServer(('0.0.0.0', 7860), SimpleHandler)
        print("✅ Simple server running on port 7860")
        server.serve_forever()

if __name__ == "__main__":
    main()