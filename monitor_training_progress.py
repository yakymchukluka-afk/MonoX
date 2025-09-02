#!/usr/bin/env python3
"""
Real-time MonoX Training Monitor
"""

import os
import time
from pathlib import Path
import subprocess

def check_training_status():
    """Check if training is running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'simple_gan_training'], capture_output=True, text=True)
        return bool(result.stdout.strip())
    except:
        return False

def get_progress():
    """Get current training progress."""
    preview_dir = Path("previews")
    checkpoint_dir = Path("checkpoints")
    logs_dir = Path("logs")
    
    previews = len(list(preview_dir.glob("*.png"))) if preview_dir.exists() else 0
    checkpoints = len(list(checkpoint_dir.glob("*.pth"))) if checkpoint_dir.exists() else 0
    logs = len(list(logs_dir.glob("*.log"))) if logs_dir.exists() else 0
    
    return previews, checkpoints, logs

def monitor():
    """Monitor training progress."""
    print("🎯 MonoX Training Monitor")
    print("=" * 40)
    
    start_time = time.time()
    last_previews = 0
    
    while True:
        is_running = check_training_status()
        previews, checkpoints, logs = get_progress()
        elapsed = time.time() - start_time
        
        # Clear screen and show status
        os.system('clear' if os.name == 'posix' else 'cls')
        print("🎯 MonoX Training Monitor")
        print("=" * 40)
        print(f"⏰ Elapsed: {elapsed/60:.1f} minutes")
        print(f"🔄 Status: {'🟢 RUNNING' if is_running else '🔴 STOPPED'}")
        print(f"🖼️  Samples: {previews}")
        print(f"💾 Checkpoints: {checkpoints}")
        print(f"📝 Logs: {logs}")
        
        if previews > last_previews:
            print(f"🎉 NEW SAMPLE GENERATED! (Total: {previews})")
            last_previews = previews
        
        if previews > 0:
            progress_percent = (previews / 50) * 100
            print(f"📊 Progress: {progress_percent:.1f}% ({previews}/50 epochs)")
            
            if previews >= 50:
                print("🏆 TRAINING COMPLETE!")
                break
        
        if not is_running and previews == 0:
            print("⚠️  Training may have stopped unexpectedly")
        
        # Show latest files
        if Path("previews").exists():
            latest_files = sorted(Path("previews").glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
            if latest_files:
                print(f"📂 Latest: {latest_files[0].name}")
        
        print("\nPress Ctrl+C to stop monitoring...")
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")