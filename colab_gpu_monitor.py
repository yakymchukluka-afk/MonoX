#!/usr/bin/env python3
"""
Real-time GPU Monitoring for MonoX Training
==========================================

Monitors GPU usage, memory, and training progress in real-time.

Usage:
    !python colab_gpu_monitor.py &  # Run in background
    !python colab_gpu_monitor.py --interval 10  # Check every 10 seconds
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
import signal
import threading
from pathlib import Path

class GPUMonitor:
    def __init__(self, interval=5, log_file=None):
        self.interval = interval
        self.log_file = log_file
        self.running = True
        self.start_time = time.time()
        
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        try:
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_util, mem_util, mem_used, mem_total, temp, power = result.stdout.strip().split(", ")
                return {
                    "gpu_util": int(gpu_util),
                    "mem_util": int(mem_util),
                    "mem_used": int(mem_used),
                    "mem_total": int(mem_total),
                    "temperature": int(temp),
                    "power": float(power) if power != "[N/A]" else 0.0
                }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        
        return None
    
    def get_training_progress(self):
        """Check for training progress in log files"""
        logs_dir = "/content/MonoX/results/logs"
        if not os.path.exists(logs_dir):
            return None
        
        # Find the most recent log file
        log_files = list(Path(logs_dir).glob("*.log"))
        if not log_files:
            return None
        
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        
        try:
            # Read last few lines for progress info
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                
            # Look for training progress indicators
            for line in reversed(lines[-50:]):  # Check last 50 lines
                line = line.strip().lower()
                if "kimg" in line and ("tick" in line or "progress" in line):
                    return line
                elif "epoch" in line and "loss" in line:
                    return line
                elif "step" in line and ("loss" in line or "lr" in line):
                    return line
        except Exception:
            pass
        
        return None
    
    def format_stats(self, stats, progress=None):
        """Format statistics for display"""
        if not stats:
            return "❌ GPU stats unavailable"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # GPU utilization bar
        gpu_bar = "█" * (stats["gpu_util"] // 10) + "░" * (10 - stats["gpu_util"] // 10)
        
        # Memory usage
        mem_pct = (stats["mem_used"] / stats["mem_total"]) * 100
        mem_bar = "█" * (int(mem_pct) // 10) + "░" * (10 - int(mem_pct) // 10)
        
        runtime = time.time() - self.start_time
        runtime_str = f"{int(runtime // 3600):02d}:{int((runtime % 3600) // 60):02d}:{int(runtime % 60):02d}"
        
        status = f"""[{timestamp}] Runtime: {runtime_str}
🖥️  GPU:  {gpu_bar} {stats['gpu_util']:3d}%  |  🌡️  {stats['temperature']:2d}°C  |  ⚡ {stats['power']:5.1f}W
💾 MEM:  {mem_bar} {mem_pct:5.1f}%  |  {stats['mem_used']:,}MB / {stats['mem_total']:,}MB"""
        
        if progress:
            status += f"\n📈 Progress: {progress}"
        
        return status
    
    def log_to_file(self, message):
        """Log message to file if specified"""
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - {message}\n")
            except Exception:
                pass
    
    def run(self):
        """Main monitoring loop"""
        print("🚀 Starting GPU Monitor")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            while self.running:
                stats = self.get_gpu_stats()
                progress = self.get_training_progress()
                
                status = self.format_stats(stats, progress)
                
                # Clear previous output and print new status
                print("\033[H\033[J", end="")  # Clear screen
                print(status)
                
                # Log to file
                if stats:
                    self.log_to_file(f"GPU:{stats['gpu_util']}% MEM:{stats['mem_used']}MB TEMP:{stats['temperature']}°C")
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  GPU monitoring stopped")
        except Exception as e:
            print(f"\n💥 Monitoring error: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

def check_training_files():
    """Check if training output files are being created/updated"""
    print("\n🔍 Checking Training Output Files")
    print("-" * 40)
    
    paths_to_check = [
        ("/content/MonoX/results/logs", "log files", "*.log"),
        ("/content/MonoX/results/previews", "preview images", "*.png"),
        ("/content/MonoX/results/checkpoints", "checkpoints", "*.pkl")
    ]
    
    for dir_path, description, pattern in paths_to_check:
        if os.path.exists(dir_path):
            files = list(Path(dir_path).glob(pattern))
            if files:
                # Find most recent file
                latest_file = max(files, key=lambda p: p.stat().st_mtime)
                mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                time_ago = datetime.now() - mod_time
                
                if time_ago.total_seconds() < 300:  # Modified in last 5 minutes
                    print(f"✅ {description}: {len(files)} files (latest: {time_ago.seconds}s ago)")
                else:
                    print(f"⚠️  {description}: {len(files)} files (latest: {time_ago} ago)")
            else:
                print(f"❌ {description}: no files found")
        else:
            print(f"❌ {description}: directory not found ({dir_path})")

def quick_gpu_check():
    """Quick one-time GPU check"""
    print("⚡ Quick GPU Check")
    print("-" * 20)
    
    monitor = GPUMonitor()
    stats = monitor.get_gpu_stats()
    
    if stats:
        print(f"GPU Utilization: {stats['gpu_util']}%")
        print(f"Memory Used: {stats['mem_used']:,}MB / {stats['mem_total']:,}MB ({stats['mem_used']/stats['mem_total']*100:.1f}%)")
        print(f"Temperature: {stats['temperature']}°C")
        print(f"Power Draw: {stats['power']:.1f}W")
        
        if stats['gpu_util'] > 50:
            print("✅ GPU is actively being used!")
        elif stats['mem_used'] > 2000:  # More than 2GB
            print("✅ GPU memory is allocated (training likely active)")
        else:
            print("⚠️  Low GPU usage - training may not be running")
    else:
        print("❌ Could not get GPU statistics")

def main():
    parser = argparse.ArgumentParser(description="Monitor GPU usage during MonoX training")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--log-file", "-l", help="Log file to write stats to")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick one-time check")
    parser.add_argument("--check-files", "-f", action="store_true", help="Check training output files")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_gpu_check()
        return
    
    if args.check_files:
        check_training_files()
        return
    
    monitor = GPUMonitor(interval=args.interval, log_file=args.log_file)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        monitor.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor.run()

if __name__ == "__main__":
    main()