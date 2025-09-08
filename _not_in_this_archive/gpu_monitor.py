#!/usr/bin/env python3
"""
Simple GPU monitoring script for Colab
Monitors GPU usage without external dependencies.
"""

import subprocess
import time
import sys

def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        line = result.stdout.strip()
        if line:
            parts = line.split(', ')
            if len(parts) >= 4:
                gpu_util = int(parts[0])
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                temp = int(parts[3])
                mem_percent = (mem_used / mem_total) * 100
                return gpu_util, mem_percent, mem_used, mem_total, temp
    except:
        pass
    return None

def main():
    """Monitor GPU usage continuously."""
    print("🔥 GPU Monitor - Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            info = get_gpu_info()
            if info:
                gpu_util, mem_percent, mem_used, mem_total, temp = info
                print(f"🔥 GPU: {gpu_util:3d}% util | {mem_percent:5.1f}% mem ({mem_used:5d}/{mem_total:5d}MB) | {temp:2d}°C")
            else:
                print("❌ Unable to get GPU info")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 GPU monitoring stopped")

if __name__ == "__main__":
    main()