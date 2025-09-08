#!/usr/bin/env python3
"""
Monitor MonoX Training Progress
Check training status and upload progress to lukua/monox model repo.
"""

import os
import sys
import time
import json
from pathlib import Path
import subprocess

def check_training_status():
    """Check if training is still running."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        training_processes = [
            line for line in result.stdout.split('\n')
            if 'simple_gan_training.py' in line and 'python' in line
        ]
        
        return len(training_processes) > 0, training_processes
    
    except Exception as e:
        print(f"Error checking training status: {e}")
        return False, []

def check_outputs():
    """Check training outputs."""
    outputs = {
        "checkpoints": len(list(Path("/workspace/checkpoints").glob("*.pt"))),
        "previews": len(list(Path("/workspace/previews").glob("*.png"))),
        "logs": len(list(Path("/workspace/logs").glob("*.log")))
    }
    
    return outputs

def upload_progress_report():
    """Create and upload progress report."""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("❌ No HF token for upload")
        return False
    
    try:
        from huggingface_hub import upload_file
        
        is_running, processes = check_training_status()
        outputs = check_outputs()
        
        report = {
            "timestamp": time.time(),
            "training_active": is_running,
            "outputs": outputs,
            "total_outputs": sum(outputs.values()),
            "processes": len(processes)
        }
        
        report_path = "/workspace/logs/training_progress_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        upload_file(
            path_or_fileobj=report_path,
            path_in_repo="reports/training_progress.json",
            repo_id="lukua/monox",
            token=hf_token,
            repo_type="model"
        )
        
        print(f"✅ Progress report uploaded")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main monitoring function."""
    print("🔍 MonoX Training Monitor")
    print("=" * 40)
    
    # Check training status
    is_running, processes = check_training_status()
    
    if is_running:
        print("✅ Training is running")
        for process in processes:
            print(f"  Process: {process.split()[1]} (PID: {process.split()[1]})")
    else:
        print("⚠️ No training process found")
    
    # Check outputs
    outputs = check_outputs()
    print(f"\\n📁 Current Outputs:")
    print(f"  Checkpoints: {outputs['checkpoints']}")
    print(f"  Previews: {outputs['previews']}")
    print(f"  Logs: {outputs['logs']}")
    print(f"  Total: {sum(outputs.values())} files")
    
    # Upload progress report
    if upload_progress_report():
        print("\\n📤 Progress report uploaded to lukua/monox")
    
    # Show latest files
    latest_files = []
    for directory in ["/workspace/checkpoints", "/workspace/previews"]:
        dir_path = Path(directory)
        if dir_path.exists():
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    latest_files.append((file_path.name, file_path.stat().st_mtime))
    
    if latest_files:
        latest_files.sort(key=lambda x: x[1], reverse=True)
        print(f"\\n📄 Latest Files:")
        for filename, mtime in latest_files[:5]:
            print(f"  {filename} ({time.ctime(mtime)})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())