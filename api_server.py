from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import subprocess
import threading
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MonoX Training API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TrainingRequest(BaseModel):
    dataset_path: str = "/workspace/dataset"
    total_kimg: int = 1000
    resolution: int = 1024
    num_gpus: int = 1

class TrainingResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class MonoXTrainingManager:
    def __init__(self):
        self.training_process = None
        self.training_status = "idle"
        self.training_logs = []
        self.setup_workspace()
        
    def setup_workspace(self):
        """Setup the workspace and ensure required directories exist."""
        for directory in ["logs", "checkpoints", "previews", "dataset"]:
            os.makedirs(directory, exist_ok=True)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda_available else []
            
            # Check CUDA version
            cuda_version = torch.version.cuda if cuda_available else None
            
        except ImportError:
            cuda_available = False
            gpu_count = 0
            gpu_names = []
            cuda_version = None
        
        # Check workspace structure
        workspace_files = {
            "launch_script": os.path.exists("src/infra/launch.py"),
            "config_dir": os.path.exists("configs"),
            "requirements": os.path.exists("requirements.txt"),
            "dataset_dir": os.path.exists("dataset"),
            "logs_dir": os.path.exists("logs"),
            "checkpoints_dir": os.path.exists("checkpoints")
        }
        
        return {
            "timestamp": time.time(),
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "training_status": self.training_status,
            "workspace_files": workspace_files,
            "python_version": sys.version,
            "process_id": os.getpid(),
            "working_directory": os.getcwd()
        }
    
    def start_training(self, config: TrainingRequest) -> Dict[str, Any]:
        """Start the training process with given configuration."""
        if self.training_status == "running":
            return {"success": False, "message": "Training is already running"}
        
        # Validate dataset path
        if not os.path.exists(config.dataset_path):
            return {"success": False, "message": f"Dataset path does not exist: {config.dataset_path}"}
        
        try:
            # Prepare command
            cmd = [
                sys.executable, "-m", "src.infra.launch",
                f"dataset.path={config.dataset_path}",
                f"dataset.resolution={config.resolution}",
                f"training.total_kimg={config.total_kimg}",
                f"training.num_gpus={config.num_gpus}",
                "hydra.run.dir=logs",
                "exp_suffix=api_training"
            ]
            
            logger.info(f"Starting training with command: {' '.join(cmd)}")
            
            # Start process
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/workspace",
                env=dict(os.environ, PYTHONUNBUFFERED="1")
            )
            
            self.training_status = "running"
            self.training_logs = []  # Reset logs
            
            # Start monitoring thread
            threading.Thread(target=self._monitor_training, daemon=True).start()
            
            return {
                "success": True,
                "message": "Training started successfully",
                "data": {
                    "pid": self.training_process.pid,
                    "status": self.training_status,
                    "config": config.dict(),
                    "command": " ".join(cmd)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self.training_status = "error"
            return {"success": False, "message": f"Failed to start training: {str(e)}"}
    
    def _monitor_training(self):
        """Monitor training process and collect logs."""
        if not self.training_process:
            return
        
        try:
            for line in self.training_process.stdout:
                log_entry = {
                    "timestamp": time.time(),
                    "line": line.strip()
                }
                self.training_logs.append(log_entry)
                
                # Keep only last 2000 lines
                if len(self.training_logs) > 2000:
                    self.training_logs = self.training_logs[-2000:]
                
                logger.info(f"Training log: {line.strip()}")
            
            # Wait for process to complete
            return_code = self.training_process.wait()
            
            if return_code == 0:
                self.training_status = "completed"
                logger.info("Training completed successfully")
            else:
                self.training_status = "failed"
                logger.error(f"Training failed with return code: {return_code}")
                
        except Exception as e:
            logger.error(f"Error monitoring training: {e}")
            self.training_status = "error"
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop the current training process."""
        if not self.training_process or self.training_status != "running":
            return {"success": False, "message": "No training process is currently running"}
        
        try:
            logger.info("Stopping training process...")
            self.training_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.training_process.wait(timeout=15)
                self.training_status = "stopped"
                message = "Training stopped gracefully"
            except subprocess.TimeoutExpired:
                logger.warning("Training process didn't stop gracefully, force killing...")
                self.training_process.kill()
                self.training_process.wait()
                self.training_status = "killed"
                message = "Training process was force killed"
            
            return {"success": True, "message": message}
            
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return {"success": False, "message": f"Error stopping training: {str(e)}"}
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get detailed training status and logs."""
        recent_logs = self.training_logs[-100:] if self.training_logs else []
        
        return {
            "status": self.training_status,
            "process_alive": self.training_process.poll() is None if self.training_process else False,
            "total_log_lines": len(self.training_logs),
            "recent_logs": [log["line"] for log in recent_logs],
            "log_timestamps": [log["timestamp"] for log in recent_logs],
            "last_update": time.time()
        }
    
    def get_full_logs(self, lines: int = 500) -> Dict[str, Any]:
        """Get full training logs."""
        logs_to_return = self.training_logs[-lines:] if self.training_logs else []
        
        return {
            "total_lines": len(self.training_logs),
            "returned_lines": len(logs_to_return),
            "logs": logs_to_return
        }
    
    def list_checkpoints(self) -> Dict[str, Any]:
        """List all available checkpoints."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return {"checkpoints": [], "count": 0}
        
        checkpoints = []
        patterns = ["*.pkl", "*.pt", "*.ckpt", "*.pth"]
        
        for pattern in patterns:
            for ckpt_file in checkpoint_dir.glob(pattern):
                stat = ckpt_file.stat()
                checkpoints.append({
                    "name": ckpt_file.name,
                    "path": str(ckpt_file),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified_timestamp": stat.st_mtime,
                    "modified_readable": time.ctime(stat.st_mtime)
                })
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x["modified_timestamp"], reverse=True)
        
        return {"checkpoints": checkpoints, "count": len(checkpoints)}

# Initialize the training manager
training_manager = MonoXTrainingManager()

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MonoX Training API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/system/info",
            "/training/start",
            "/training/stop",
            "/training/status",
            "/training/logs",
            "/checkpoints/list"
        ]
    }

@app.get("/system/info")
async def get_system_info():
    """Get system information including GPU status."""
    return training_manager.get_system_info()

@app.post("/training/start")
async def start_training(config: TrainingRequest):
    """Start training with the given configuration."""
    result = training_manager.start_training(config)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/training/stop")
async def stop_training():
    """Stop the current training process."""
    result = training_manager.stop_training()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.get("/training/status")
async def get_training_status():
    """Get current training status and recent logs."""
    return training_manager.get_training_status()

@app.get("/training/logs")
async def get_training_logs(lines: int = 500):
    """Get training logs (default: last 500 lines)."""
    return training_manager.get_full_logs(lines)

@app.get("/checkpoints/list")
async def list_checkpoints():
    """List all available checkpoints."""
    return training_manager.list_checkpoints()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "training_status": training_manager.training_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")