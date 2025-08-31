import gradio as gr
import os
import sys
import json
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonoXTrainingAPI:
    def __init__(self):
        self.training_process = None
        self.training_status = "idle"
        self.training_logs = []
        self.setup_workspace()
    
    def setup_workspace(self):
        """Setup the workspace and ensure required directories exist."""
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("previews", exist_ok=True)
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information including GPU status."""
        try:
            # Check CUDA availability
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda_available else []
        except ImportError:
            cuda_available = False
            gpu_count = 0
            gpu_names = []
        
        return {
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "training_status": self.training_status,
            "workspace_ready": os.path.exists("src/infra/launch.py")
        }
    
    def start_training(self, dataset_path: str = "", total_kimg: int = 1000, resolution: int = 1024) -> Dict[str, Any]:
        """Start the training process."""
        if self.training_status == "running":
            return {"success": False, "message": "Training is already running"}
        
        if not dataset_path:
            dataset_path = os.environ.get("DATASET_DIR", "/workspace/dataset")
        
        try:
            # Use the existing launch script
            cmd = [
                sys.executable, "-m", "src.infra.launch",
                f"dataset.path={dataset_path}",
                f"dataset.resolution={resolution}",
                f"training.total_kimg={total_kimg}",
                "hydra.run.dir=logs",
                "exp_suffix=hf_space"
            ]
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/workspace"
            )
            
            self.training_status = "running"
            
            # Start log monitoring in background
            threading.Thread(target=self._monitor_training, daemon=True).start()
            
            return {
                "success": True,
                "message": "Training started successfully",
                "pid": self.training_process.pid,
                "status": self.training_status
            }
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return {"success": False, "message": f"Failed to start training: {str(e)}"}
    
    def _monitor_training(self):
        """Monitor training process and collect logs."""
        if not self.training_process:
            return
            
        try:
            for line in self.training_process.stdout:
                self.training_logs.append(line.strip())
                # Keep only last 1000 lines
                if len(self.training_logs) > 1000:
                    self.training_logs = self.training_logs[-1000:]
                    
            self.training_process.wait()
            self.training_status = "completed" if self.training_process.returncode == 0 else "failed"
            
        except Exception as e:
            logger.error(f"Error monitoring training: {e}")
            self.training_status = "error"
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop the training process."""
        if self.training_process and self.training_status == "running":
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=10)
                self.training_status = "stopped"
                return {"success": True, "message": "Training stopped"}
            except subprocess.TimeoutExpired:
                self.training_process.kill()
                self.training_status = "killed"
                return {"success": True, "message": "Training force killed"}
            except Exception as e:
                return {"success": False, "message": f"Failed to stop training: {str(e)}"}
        else:
            return {"success": False, "message": "No training process running"}
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and recent logs."""
        return {
            "status": self.training_status,
            "recent_logs": self.training_logs[-50:] if self.training_logs else [],
            "total_log_lines": len(self.training_logs),
            "process_alive": self.training_process.poll() is None if self.training_process else False
        }
    
    def list_checkpoints(self) -> Dict[str, Any]:
        """List available checkpoints."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return {"checkpoints": []}
        
        checkpoints = []
        for pattern in ["*.pkl", "*.pt", "*.ckpt", "*.pth"]:
            for ckpt in checkpoint_dir.glob(pattern):
                checkpoints.append({
                    "name": ckpt.name,
                    "size": ckpt.stat().st_size,
                    "modified": ckpt.stat().st_mtime
                })
        
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return {"checkpoints": checkpoints}

# Initialize the API
api = MonoXTrainingAPI()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="MonoX Training Interface") as demo:
        gr.Markdown("# MonoX StyleGAN-V Training Interface")
        
        with gr.Tab("System Status"):
            system_info_btn = gr.Button("Refresh System Info")
            system_info_output = gr.JSON(label="System Information")
            
            system_info_btn.click(
                fn=api.get_system_info,
                outputs=system_info_output
            )
        
        with gr.Tab("Training Control"):
            with gr.Row():
                dataset_path = gr.Textbox(
                    label="Dataset Path", 
                    value="/workspace/dataset",
                    placeholder="Path to training dataset"
                )
                total_kimg = gr.Number(
                    label="Total KImg", 
                    value=1000,
                    precision=0
                )
                resolution = gr.Number(
                    label="Resolution", 
                    value=1024,
                    precision=0
                )
            
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop")
                status_btn = gr.Button("Get Status")
            
            training_output = gr.JSON(label="Training Response")
            
            start_btn.click(
                fn=api.start_training,
                inputs=[dataset_path, total_kimg, resolution],
                outputs=training_output
            )
            
            stop_btn.click(
                fn=api.stop_training,
                outputs=training_output
            )
            
            status_btn.click(
                fn=api.get_training_status,
                outputs=training_output
            )
        
        with gr.Tab("Logs & Monitoring"):
            refresh_logs_btn = gr.Button("Refresh Logs")
            logs_output = gr.JSON(label="Training Status & Logs")
            
            refresh_logs_btn.click(
                fn=api.get_training_status,
                outputs=logs_output
            )
        
        with gr.Tab("Checkpoints"):
            refresh_ckpt_btn = gr.Button("List Checkpoints")
            ckpt_output = gr.JSON(label="Available Checkpoints")
            
            refresh_ckpt_btn.click(
                fn=api.list_checkpoints,
                outputs=ckpt_output
            )
        
        # Auto-refresh system info on load
        demo.load(fn=api.get_system_info, outputs=system_info_output)
    
    return demo

# API endpoints for direct JSON access
def create_api_routes():
    """Create API routes that return JSON responses."""
    
    def api_system_info():
        return api.get_system_info()
    
    def api_start_training(dataset_path: str = "", total_kimg: int = 1000, resolution: int = 1024):
        return api.start_training(dataset_path, total_kimg, resolution)
    
    def api_stop_training():
        return api.stop_training()
    
    def api_training_status():
        return api.get_training_status()
    
    def api_list_checkpoints():
        return api.list_checkpoints()
    
    # Create API interface
    api_demo = gr.Interface(
        fn=api_system_info,
        inputs=[],
        outputs=gr.JSON(),
        title="MonoX API - System Info",
        description="JSON API endpoint for system information"
    )
    
    return {
        "system_info": api_demo,
        "functions": {
            "system_info": api_system_info,
            "start_training": api_start_training,
            "stop_training": api_stop_training,
            "training_status": api_training_status,
            "list_checkpoints": api_list_checkpoints
        }
    }

if __name__ == "__main__":
    # Run startup checks
    try:
        from startup import main as startup_main
        startup_success = startup_main()
        if not startup_success:
            logger.error("Startup checks failed")
    except Exception as e:
        logger.warning(f"Startup checks failed: {e}")
    
    # Create the main interface
    demo = create_interface()
    
    # Launch with proper configuration for Hugging Face Spaces
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        quiet=False
    )