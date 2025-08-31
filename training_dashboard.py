#!/usr/bin/env python3
"""
MonoX Training Dashboard - Real-time progress tracking
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json

def get_training_status():
    """Get current training process status."""
    try:
        result = subprocess.run(['pgrep', '-f', 'simple_gan_training'], 
                              capture_output=True, text=True)
        pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return len(pids) > 0, pids
    except:
        return False, []

def get_file_info(filepath):
    """Get file size and modification time."""
    if not os.path.exists(filepath):
        return None
    
    stat = os.stat(filepath)
    size_mb = stat.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    return {
        'size_mb': round(size_mb, 2),
        'modified': mod_time.strftime('%H:%M:%S'),
        'age_minutes': round((time.time() - stat.st_mtime) / 60, 1)
    }

def analyze_progress():
    """Analyze training progress from outputs."""
    preview_dir = Path('previews')
    checkpoint_dir = Path('checkpoints')
    
    # Get all samples
    samples = []
    if preview_dir.exists():
        for sample_file in sorted(preview_dir.glob('samples_epoch_*.png')):
            epoch_num = int(sample_file.stem.split('_')[-1])
            info = get_file_info(sample_file)
            if info:
                samples.append({
                    'epoch': epoch_num,
                    'filename': sample_file.name,
                    **info
                })
    
    # Get checkpoints
    checkpoints = []
    if checkpoint_dir.exists():
        for checkpoint_file in sorted(checkpoint_dir.glob('*.pth')):
            info = get_file_info(checkpoint_file)
            if info:
                checkpoints.append({
                    'filename': checkpoint_file.name,
                    **info
                })
    
    return samples, checkpoints

def show_sample_preview(sample_path):
    """Show sample information and suggest viewing methods."""
    if not os.path.exists(sample_path):
        print(f"❌ Sample not found: {sample_path}")
        return
    
    info = get_file_info(sample_path)
    print(f"🖼️  Sample: {os.path.basename(sample_path)}")
    print(f"   Size: {info['size_mb']} MB")
    print(f"   Generated: {info['modified']} ({info['age_minutes']} min ago)")
    print(f"   Path: {os.path.abspath(sample_path)}")
    print(f"   📱 View methods:")
    print(f"      - Copy to local: scp user@server:{os.path.abspath(sample_path)} .")
    print(f"      - Open in browser: file://{os.path.abspath(sample_path)}")
    print(f"      - Base64 preview available")

def main():
    """Main dashboard function."""
    print("🎨 MonoX Training Dashboard")
    print("=" * 50)
    
    # Check training status
    is_running, pids = get_training_status()
    if is_running:
        print(f"✅ Training ACTIVE (PID: {', '.join(pids)})")
    else:
        print("⚠️  Training NOT running")
    
    print()
    
    # Analyze progress
    samples, checkpoints = analyze_progress()
    
    print(f"📈 Progress Summary:")
    print(f"   Samples: {len(samples)}")
    print(f"   Checkpoints: {len(checkpoints)}")
    
    if samples:
        print(f"\n🎨 Generated Samples:")
        for sample in samples:
            status = "🆕" if sample['age_minutes'] < 30 else "✅"
            print(f"   {status} Epoch {sample['epoch']:02d}: {sample['size_mb']}MB ({sample['age_minutes']}min ago)")
        
        # Show latest sample details
        latest = samples[-1]
        print(f"\n🖼️  Latest Sample Details:")
        show_sample_preview(f"previews/{latest['filename']}")
    
    if checkpoints:
        print(f"\n💾 Checkpoints:")
        for checkpoint in checkpoints:
            print(f"   📁 {checkpoint['filename']}: {checkpoint['size_mb']}MB")
    
    # Training rate analysis
    if len(samples) >= 2:
        time_diff = samples[-1]['age_minutes'] - samples[-2]['age_minutes']
        if time_diff > 0:
            print(f"\n⏱️  Training Rate: ~{abs(time_diff):.1f} minutes per epoch")
            estimated_total = abs(time_diff) * 50  # 50 epochs total
            print(f"   📅 Estimated total time: {estimated_total/60:.1f} hours")

if __name__ == "__main__":
    main()