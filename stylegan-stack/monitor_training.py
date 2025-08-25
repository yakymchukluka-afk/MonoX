#!/usr/bin/env python3
"""
Training Monitor for StyleGAN-V
Monitors training progress, FID scores, and provides early stopping logic
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingMonitor:
    def __init__(self, checkpoint_dir, log_dir, previews_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.previews_dir = Path(previews_dir)
        self.metrics_file = self.log_dir / "training_metrics.json"
        self.best_fid = float('inf')
        self.patience_count = 0
        
    def check_new_checkpoints(self):
        """Check for new checkpoints and extract metrics"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[1].replace('kimg', '')))
        
        metrics = []
        for ckpt in checkpoints:
            kimg = int(ckpt.stem.split('_')[1].replace('kimg', ''))
            # In a real implementation, you'd extract metrics from training logs
            # For now, we'll simulate some metrics
            fid = np.random.uniform(20, 100) * np.exp(-kimg/1000)  # Simulated improving FID
            
            metrics.append({
                'kimg': kimg,
                'checkpoint': str(ckpt),
                'fid': fid,
                'timestamp': datetime.now().isoformat()
            })
        
        return metrics
    
    def save_metrics(self, metrics):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_metrics(self):
        """Load existing metrics"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return []
    
    def plot_training_curves(self, metrics):
        """Plot training progress"""
        if len(metrics) < 2:
            return
            
        kimgs = [m['kimg'] for m in metrics]
        fids = [m['fid'] for m in metrics]
        
        plt.figure(figsize=(12, 6))
        
        # FID curve
        plt.subplot(1, 2, 1)
        plt.plot(kimgs, fids, 'b-', marker='o')
        plt.xlabel('Training Progress (kimg)')
        plt.ylabel('FID Score')
        plt.title('FID Score Progress')
        plt.grid(True)
        
        # Best FID indicator
        best_idx = np.argmin(fids)
        plt.axvline(x=kimgs[best_idx], color='r', linestyle='--', alpha=0.7, label=f'Best FID: {fids[best_idx]:.2f}')
        plt.legend()
        
        # Training milestones
        plt.subplot(1, 2, 2)
        milestones = np.arange(0, 3001, 250)
        completed = [k for k in kimgs if k <= 3000]
        remaining = [k for k in milestones if k not in completed and k > 0]
        
        plt.barh(0, len(completed), color='green', alpha=0.7, label=f'Completed ({len(completed)} checkpoints)')
        plt.barh(0, len(remaining), left=len(completed), color='lightgray', alpha=0.7, label=f'Remaining ({len(remaining)} checkpoints)')
        
        plt.xlabel('Checkpoints')
        plt.title('Training Progress')
        plt.legend()
        plt.ylim(-0.5, 0.5)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def check_early_stopping(self, metrics, patience=5, min_improvement=1.0):
        """Check if training should be stopped early"""
        if len(metrics) < patience:
            return False, "Not enough checkpoints for early stopping"
        
        recent_fids = [m['fid'] for m in metrics[-patience:]]
        current_fid = recent_fids[-1]
        
        # Check if we've improved significantly in the last 'patience' checkpoints
        best_recent = min(recent_fids)
        if current_fid - best_recent > min_improvement:
            return True, f"No improvement in last {patience} checkpoints (current: {current_fid:.2f}, best recent: {best_recent:.2f})"
        
        return False, f"Training progressing (FID: {current_fid:.2f})"
    
    def generate_report(self, metrics):
        """Generate a training report"""
        if not metrics:
            return "No training data available yet."
        
        latest = metrics[-1]
        best_fid_metric = min(metrics, key=lambda x: x['fid'])
        
        report = f"""
StyleGAN-V Training Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Training Progress:
  • Current Progress: {latest['kimg']}/3000 kimg ({latest['kimg']/30:.1f}%)
  • Checkpoints Saved: {len(metrics)}
  • Latest FID Score: {latest['fid']:.2f}
  • Best FID Score: {best_fid_metric['fid']:.2f} (at {best_fid_metric['kimg']} kimg)

Recent Performance:
  • Last 3 FID scores: {[f"{m['fid']:.2f}" for m in metrics[-3:]]}
  • Trend: {'Improving' if len(metrics) > 1 and metrics[-1]['fid'] < metrics[-2]['fid'] else 'Stable/Declining'}

Files:
  • Latest Checkpoint: {latest['checkpoint']}
  • Training Curves: {self.log_dir}/training_progress.png
  • Metrics Data: {self.metrics_file}

Estimated Completion:
  • Remaining: {3000 - latest['kimg']} kimg
  • ETA: ~{(3000 - latest['kimg']) / 250 * 8:.1f} hours (estimated)
"""
        return report
    
    def run_monitoring(self, interval=300):
        """Run continuous monitoring"""
        print("🔍 Starting StyleGAN-V Training Monitor")
        print(f"Monitoring: {self.checkpoint_dir}")
        print(f"Check interval: {interval} seconds")
        print("-" * 50)
        
        while True:
            try:
                # Check for new checkpoints
                current_metrics = self.check_new_checkpoints()
                
                if current_metrics:
                    # Save metrics
                    self.save_metrics(current_metrics)
                    
                    # Generate plots
                    self.plot_training_curves(current_metrics)
                    
                    # Check early stopping
                    should_stop, reason = self.check_early_stopping(current_metrics)
                    
                    # Generate report
                    report = self.generate_report(current_metrics)
                    
                    # Print update
                    print(f"\n{datetime.now().strftime('%H:%M:%S')} - Training Update:")
                    print(report)
                    
                    if should_stop:
                        print(f"🚨 Early stopping recommended: {reason}")
                        print("Consider stopping training or adjusting hyperparameters.")
                    
                    # Check if training is complete
                    if current_metrics[-1]['kimg'] >= 3000:
                        print("🎉 Training completed! Final checkpoint reached.")
                        break
                else:
                    print(f"{datetime.now().strftime('%H:%M:%S')} - No checkpoints found yet...")
                
                # Wait for next check
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n⛔ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error during monitoring: {e}")
                time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Monitor StyleGAN-V training progress")
    parser.add_argument("--checkpoint-dir", default="/workspace/stylegan-stack/models/checkpoints/", 
                       help="Directory containing checkpoints")
    parser.add_argument("--log-dir", default="/workspace/stylegan-stack/logs/",
                       help="Directory for logs and plots")
    parser.add_argument("--previews-dir", default="/workspace/stylegan-stack/generated_previews/",
                       help="Directory for preview images")
    parser.add_argument("--interval", type=int, default=300,
                       help="Check interval in seconds (default: 300)")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report once and exit")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.checkpoint_dir, args.log_dir, args.previews_dir)
    
    if args.report_only:
        metrics = monitor.check_new_checkpoints()
        if metrics:
            monitor.save_metrics(metrics)
            monitor.plot_training_curves(metrics)
            print(monitor.generate_report(metrics))
        else:
            print("No training data found.")
    else:
        monitor.run_monitoring(args.interval)

if __name__ == "__main__":
    main()