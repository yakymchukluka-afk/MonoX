#!/usr/bin/env python3
"""
Real-time training progress tracker with live updates
"""

import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

def get_live_progress():
    """Get real-time training progress."""
    # Check processes
    try:
        result = subprocess.run(['pgrep', '-f', 'gan_training'], capture_output=True, text=True)
        training_active = len(result.stdout.strip()) > 0
    except:
        training_active = False
    
    # Count outputs
    previews = len(list(Path('previews').glob('*.png'))) if Path('previews').exists() else 0
    checkpoints = len(list(Path('checkpoints').glob('*.pth'))) if Path('checkpoints').exists() else 0
    logs = len(list(Path('.').glob('*.log'))) if Path('.').exists() else 0
    
    # Get latest sample info
    latest_sample = None
    if Path('previews').exists():
        samples = sorted(Path('previews').glob('samples_epoch_*.png'), key=lambda x: x.stat().st_mtime)
        if samples:
            latest = samples[-1]
            latest_sample = {
                'filename': latest.name,
                'epoch': int(latest.stem.split('_')[-1]),
                'size_mb': latest.stat().st_size / (1024*1024),
                'time_ago_min': (time.time() - latest.stat().st_mtime) / 60
            }
    
    return {
        'timestamp': time.time(),
        'training_active': training_active,
        'outputs': {
            'previews': previews,
            'checkpoints': checkpoints,
            'logs': logs
        },
        'latest_sample': latest_sample,
        'total_outputs': previews + checkpoints + logs
    }

def display_progress():
    """Display formatted progress information."""
    progress = get_live_progress()
    
    print("🎨 MonoX Training Progress")
    print("=" * 40)
    print(f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Training status
    status = "🟢 RUNNING" if progress['training_active'] else "🔴 STOPPED"
    print(f"📊 Status: {status}")
    
    # Output summary
    outputs = progress['outputs']
    print(f"📁 Outputs: {outputs['previews']} previews, {outputs['checkpoints']} checkpoints")
    
    # Latest sample
    if progress['latest_sample']:
        sample = progress['latest_sample']
        print(f"🖼️  Latest: Epoch {sample['epoch']} ({sample['size_mb']:.1f}MB, {sample['time_ago_min']:.0f}min ago)")
        
        # Show next checkpoint
        next_checkpoint = ((sample['epoch'] // 5) + 1) * 5
        epochs_to_checkpoint = next_checkpoint - sample['epoch']
        print(f"💾 Next checkpoint: Epoch {next_checkpoint} ({epochs_to_checkpoint} epochs away)")
    
    # Progress bar
    if progress['latest_sample']:
        current_epoch = progress['latest_sample']['epoch']
        total_epochs = 50
        progress_pct = (current_epoch / total_epochs) * 100
        
        bar_length = 30
        filled_length = int(bar_length * current_epoch // total_epochs)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"📈 Progress: [{bar}] {progress_pct:.1f}% ({current_epoch}/{total_epochs})")
    
    return progress

def save_progress_report(progress):
    """Save progress report for monitoring."""
    report_path = Path("reports/training_progress.json")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"📝 Progress saved: {report_path}")

def main():
    """Main tracking function."""
    progress = display_progress()
    save_progress_report(progress)
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if not progress['training_active']:
        print("   🔄 Restart training: python3 simple_gan_training.py")
        print("   🚀 Or upgrade to GPU: python3 gpu_gan_training.py")
    else:
        if progress['latest_sample'] and progress['latest_sample']['time_ago_min'] > 30:
            print("   ⚠️  Training seems stalled (no new samples in 30+ min)")
            print("   🔄 Consider restarting")

if __name__ == "__main__":
    main()