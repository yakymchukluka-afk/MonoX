"""
Example Google Colab Cells for MonoX + StyleGAN-V Training
=========================================================

Copy and paste these cells into your Google Colab notebook.
"""

# ==========================================
# CELL 1: Initial Setup and GPU Check
# ==========================================
print("🚀 MonoX + StyleGAN-V Colab Setup")
print("=" * 40)

# Check GPU availability
!nvidia-smi

# Verify GPU in PyTorch
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# ==========================================
# CELL 2: Environment Setup (Run Once)
# ==========================================
print("🛠️ Setting up environment...")

# Change to content directory
import os
os.chdir('/content')

# Clone MonoX repository if not exists
if not os.path.exists('/content/MonoX'):
    !git clone https://github.com/your-username/MonoX.git
    os.chdir('/content/MonoX')
else:
    os.chdir('/content/MonoX')
    print("MonoX already cloned")

# Run environment setup
!python colab_environment_setup.py

# ==========================================  
# CELL 3: Upload Dataset (Optional)
# ==========================================
print("📁 Dataset Setup")

# Option A: Upload from local machine
from google.colab import files
import zipfile

print("Upload your dataset (zip file):")
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/MonoX/dataset/')
        print(f"✅ Extracted to /content/MonoX/dataset/")

# Option B: Use sample dataset (already created)
!ls -la /content/MonoX/dataset/sample_images/

# ==========================================
# CELL 4: Pre-Training Diagnostics  
# ==========================================
print("🔍 Running diagnostic checks...")

!python colab_debug_checklist.py

# ==========================================
# CELL 5: Start Training
# ==========================================
print("🚀 Starting training...")

# Basic training (adjust parameters as needed)
!python colab_training_launcher.py

# ==========================================
# CELL 6: Monitor Training (Run in Parallel)
# ==========================================
print("📊 Starting GPU monitoring...")

# Run this in a separate cell while training
!python colab_gpu_monitor.py --interval 10

# ==========================================
# CELL 7: Check Training Progress
# ==========================================
print("📈 Checking training progress...")

# Check latest log output
!tail -50 /content/MonoX/results/logs/*.log

# Check generated files
!ls -lt /content/MonoX/results/previews/
!ls -lt /content/MonoX/results/checkpoints/

# Quick GPU check
!python colab_gpu_monitor.py --quick

# ==========================================
# CELL 8: View Results
# ==========================================
print("🎨 Viewing generated images...")

import matplotlib.pyplot as plt
from PIL import Image
import glob

# Show latest preview images
preview_files = glob.glob('/content/MonoX/results/previews/*.png')
if preview_files:
    # Show most recent preview
    latest_preview = max(preview_files, key=os.path.getctime)
    
    img = Image.open(latest_preview)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Latest Generated Images: {os.path.basename(latest_preview)}')
    plt.show()
    
    print(f"📅 Generated: {os.path.basename(latest_preview)}")
    print(f"📂 Total previews: {len(preview_files)}")
else:
    print("❌ No preview images found yet")

# ==========================================
# CELL 9: Custom Training Configuration
# ==========================================
print("⚙️ Custom training examples...")

# Quick test run (5 minutes)
# !python colab_training_launcher.py training.total_kimg=100 training.snapshot_kimg=25

# Lower resolution for faster training
# !python colab_training_launcher.py dataset.resolution=512 training.batch_size=8

# High quality run
# !python colab_training_launcher.py dataset.resolution=1024 training.total_kimg=5000

# Resume from checkpoint
# !python colab_training_launcher.py training.resume=/content/MonoX/results/checkpoints/network-snapshot-000250.pkl

print("Uncomment the desired configuration above")

# ==========================================
# CELL 10: Save to Google Drive (Optional)
# ==========================================
print("💾 Save results to Google Drive...")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!mkdir -p /content/drive/MyDrive/MonoX_Results
!cp -r /content/MonoX/results/* /content/drive/MyDrive/MonoX_Results/

print("✅ Results saved to Google Drive: /content/drive/MyDrive/MonoX_Results/")

# Create symlink for future training
!rm -rf /content/MonoX/results
!ln -sf /content/drive/MyDrive/MonoX_Results /content/MonoX/results

print("✅ Future training will save directly to Google Drive")

# ==========================================
# CELL 11: Troubleshooting
# ==========================================
print("🔧 Troubleshooting tools...")

# Full diagnostic check
print("\n1. Full diagnostic check:")
print("!python colab_debug_checklist.py")

# Check file permissions
print("\n2. Check file permissions:")
print("!ls -la /content/MonoX/.external/stylegan-v/")

# Check Python path
print("\n3. Check Python environment:")
print("import sys; print('\\n'.join(sys.path))")

# Reset environment
print("\n4. Reset environment (if needed):")
print("!python colab_environment_setup.py")

# Manual PYTHONPATH setup
print("\n5. Manual PYTHONPATH (if needed):")
print("import os")
print("os.environ['PYTHONPATH'] = '/content/MonoX/.external/stylegan-v:/content/MonoX'")

print("\nRun these commands if you encounter issues.")

# ==========================================
# USAGE SUMMARY
# ==========================================
"""
🎯 USAGE SUMMARY:

1. Run Cell 1: Check GPU
2. Run Cell 2: Environment setup (once)  
3. Run Cell 3: Upload dataset (if needed)
4. Run Cell 4: Pre-training checks
5. Run Cell 5: Start training
6. Run Cell 6: Monitor (parallel to training)
7. Run Cell 7: Check progress
8. Run Cell 8: View results
9. Use Cell 9: For custom configs
10. Use Cell 10: Save to Drive
11. Use Cell 11: If troubleshooting needed

🔥 EXPECTED RESULTS:
- GPU utilization 80-95%
- Preview images every 50 kimg
- Checkpoints every 250 kimg
- High-quality generated images
"""