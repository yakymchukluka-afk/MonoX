#!/usr/bin/env python3
"""
🔍🔥💥 REAL-TIME TRAINING MONITOR - CATCH THE EXACT ERROR! 💥🔥🔍
================================================================================
This script monitors training in real-time to catch exactly where it fails!
"""

import os
import subprocess
import sys
import time
import threading

def real_time_training_monitor():
    """Monitor training in real-time with full output capture."""
    print("🔍🔥💥 REAL-TIME TRAINING MONITOR - CATCH THE EXACT ERROR!")
    print("=" * 80)
    
    # Force to /content directory
    os.chdir("/content")
    
    # Set environment
    print("🌍 SETTING ENVIRONMENT...")
    env_vars = {
        'TORCH_EXTENSIONS_DIR': '/tmp/torch_extensions',
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTHONPATH': '/content/MonoX/.external/stylegan-v',
        'CUDA_LAUNCH_BLOCKING': '1'  # This will help catch CUDA errors
    }
    
    for var, val in env_vars.items():
        os.environ[var] = val
        print(f"✅ {var}={val}")
    
    # Change to StyleGAN-V directory
    stylegan_dir = "/content/MonoX/.external/stylegan-v"
    os.chdir(stylegan_dir)
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Training command with minimal settings to avoid issues
    print("\n🚀 LAUNCHING TRAINING WITH REAL-TIME MONITORING...")
    
    training_cmd = [
        'python3', '-m', 'src.infra.launch',
        'hydra.run.dir=logs',
        'exp_suffix=realtime_monitor',
        'dataset.path=/content/drive/MyDrive/MonoX/dataset',
        'dataset.resolution=256',
        'training.kimg=5',  # Very short for testing
        'training.snap=1',
        'num_gpus=1',
        '++training.gpus=1',
        '++training.batch_size=2',  # Small batch size
        '++training.fp32=false',
        '++training.nobench=false',
        '++training.allow_tf32=false',
        '++training.metrics=[fid50k_full]',
        '++training.seed=0',
        '++training.data=/content/drive/MyDrive/MonoX/dataset',
        '++model.loss_kwargs.source=StyleGAN2Loss',
        '++model.loss_kwargs.style_mixing_prob=0.0',
        '++model.discriminator.mbstd_group_size=2',
        '++model.discriminator.source=networks',
        '++model.generator.source=networks',
        '++model.generator.w_dim=512',
        '+model.optim.generator.lr=0.002',
        '+model.optim.discriminator.lr=0.002',
        '++training.num_workers=2',  # Reduced workers
        '++training.subset=null',
        '++training.mirror=true',
        '++training.cfg=auto',
        '++training.aug=ada',
        '++training.p=null',
        '++training.target=0.6',
        '++training.augpipe=bgc',
        '++training.freezed=0',
        '++training.dry_run=false',
        '++training.cond=false',
        '++training.nhwc=false',
        '++training.resume=null',
        '++training.outdir=/content/MonoX/results'
    ]
    
    print(f"🔥 Command: {' '.join(training_cmd[:10])}... [truncated]")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Launch with real-time output
        process = subprocess.Popen(
            training_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy()
        )
        
        print(f"🚀 Process started with PID: {process.pid}")
        print("📊 Monitoring output (will show ALL errors)...")
        print("=" * 40)
        
        # Monitor output in real-time
        line_count = 0
        last_activity = time.time()
        setup_complete = False
        training_started = False
        gpu_active = False
        error_detected = False
        
        while True:
            output = process.stdout.readline()
            
            if output == '' and process.poll() is not None:
                # Process ended
                break
                
            if output:
                line_count += 1
                last_activity = time.time()
                print(f"{line_count:3}: {output.strip()}")
                
                # Analyze the line for key indicators
                line_lower = output.lower()
                
                # Look for errors
                error_keywords = ['error', 'failed', 'exception', 'traceback', 'not found', 'cannot', 'unable']
                if any(keyword in line_lower for keyword in error_keywords):
                    error_detected = True
                    print(f"    🚨 *** ERROR DETECTED: {output.strip()} ***")
                
                # Look for success indicators
                if 'training config is located' in line_lower:
                    setup_complete = True
                    print(f"    ✅ *** SETUP COMPLETE! ***")
                
                if any(keyword in line_lower for keyword in ['loading training set', 'constructing networks', 'launching processes']):
                    training_started = True
                    print(f"    🚀 *** TRAINING PHASE STARTED! ***")
                
                if any(keyword in line_lower for keyword in ['cuda', 'gpu', 'device']):
                    gpu_active = True
                    print(f"    🔥 *** GPU ACTIVITY! ***")
                
                # Look for completion indicators
                if any(keyword in line_lower for keyword in ['tick 0', 'kimg 0', 'training tick']):
                    print(f"    🎯 *** TRAINING TICK DETECTED! ***")
                
                # Stop after reasonable amount of output or if we see actual training
                if line_count > 500:
                    print("⏹️ Stopping monitoring after 500 lines...")
                    break
            
            # Check if process has been idle too long
            if time.time() - last_activity > 30:
                print("⏰ No output for 30 seconds, checking process status...")
                if process.poll() is not None:
                    print("💀 Process has ended")
                    break
        
        # Get final status
        return_code = process.poll()
        runtime = time.time() - start_time
        
        print(f"\n📊 MONITORING COMPLETE")
        print(f"⏱️ Runtime: {runtime:.1f} seconds")
        print(f"📋 Return code: {return_code}")
        print(f"📄 Lines captured: {line_count}")
        
        # Analysis
        print(f"\n🔍 ANALYSIS:")
        print(f"✅ Setup Complete: {setup_complete}")
        print(f"🚀 Training Started: {training_started}")
        print(f"🔥 GPU Active: {gpu_active}")
        print(f"🚨 Error Detected: {error_detected}")
        
        if return_code == 0 and runtime < 60:
            print("\n⚠️ ISSUE: Process completed too quickly!")
            print("💡 This suggests a silent failure or missing component")
        elif return_code != 0:
            print(f"\n❌ ISSUE: Process failed with code {return_code}")
        elif training_started:
            print(f"\n✅ SUCCESS: Training appears to be working!")
        
        return return_code == 0 and training_started
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return False

def check_process_status():
    """Check what's actually running."""
    print("\n🔍 CHECKING CURRENT PROCESSES...")
    
    try:
        # Check for any Python training processes
        result = subprocess.run(['pgrep', '-f', 'python.*src.infra.launch'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training processes found:")
            for pid in result.stdout.strip().split('\n'):
                print(f"   PID: {pid}")
                # Get process details
                try:
                    details = subprocess.run(['ps', '-p', pid, '-o', 'pid,ppid,cmd'], 
                                           capture_output=True, text=True)
                    print(f"   Details: {details.stdout.strip()}")
                except:
                    pass
        else:
            print("ℹ️ No training processes currently running")
            
    except Exception as e:
        print(f"⚠️ Process check failed: {e}")

def check_recent_logs():
    """Check the most recent log files."""
    print("\n📋 CHECKING RECENT LOG FILES...")
    
    log_paths = [
        "/content/MonoX/.external/stylegan-v/logs/launch.log",
        "/content/MonoX/logs"
    ]
    
    for log_path in log_paths:
        if os.path.exists(log_path):
            if os.path.isfile(log_path):
                print(f"\n📄 {log_path}:")
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print("Last 10 lines:")
                            for line in lines[-10:]:
                                print(f"   {line.strip()}")
                        else:
                            print("   (empty)")
                except Exception as e:
                    print(f"   Error reading: {e}")
            elif os.path.isdir(log_path):
                try:
                    files = os.listdir(log_path)
                    if files:
                        print(f"\n📁 {log_path}: {files}")
                    else:
                        print(f"\n📂 {log_path}: empty")
                except Exception as e:
                    print(f"\n⚠️ {log_path}: {e}")
        else:
            print(f"\n❌ {log_path}: not found")

if __name__ == "__main__":
    print("🔍🔥💥 REAL-TIME TRAINING MONITOR - CATCH THE EXACT ERROR!")
    print("=" * 80)
    
    check_process_status()
    check_recent_logs()
    
    success = real_time_training_monitor()
    
    if success:
        print("\n🔍🔥💥 REAL-TIME MONITORING SUCCESS!")
        print("✅ Training appears to be working correctly!")
        print("🔥 Check GPU usage: !nvidia-smi")
    else:
        print("\n🔍🔥💥 REAL-TIME MONITORING COMPLETE!")
        print("🚨 Check the captured output above for the exact error")
        print("💡 Focus on any lines marked with 🚨 ERROR DETECTED")
    
    print("=" * 80)