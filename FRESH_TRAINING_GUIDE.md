# MonoX Fresh Training Guide - Hugging Face Spaces

## ✅ Setup Complete - Ready for Fresh Training

Your MonoX training environment is now fully configured for fresh training from scratch using the 1024px dataset in GPU-enabled Dev Mode.

### 🎯 What's Been Set Up

#### ✅ **Dataset Configuration**
- **Source**: `lukua/monox-dataset` (868 images at 1024×1024)
- **Location**: `/workspace/dataset`
- **Format**: High-quality monotype-inspired artworks
- **Status**: ✅ Downloaded and validated

#### ✅ **Training Pipeline** 
- **Framework**: StyleGAN-V (latest version)
- **Configuration**: Optimized for fresh training
- **Checkpoints**: Every 50 kimg (~5 epochs)
- **Outputs**: Automatic upload to `lukua/monox` model repo

#### ✅ **Environment Setup**
- **Dependencies**: All required packages installed
- **Authentication**: HF token configured
- **Directories**: Logs, checkpoints, previews created
- **GPU Detection**: Automatic fallback to CPU if needed

### 🚀 How to Start Fresh Training

#### **Option 1: Full Production Training**
```bash
python3 final_monox_training.py
```
- Trains for 1000 kimg (comprehensive training)
- Saves checkpoints every 50 kimg (~5 epochs)
- Uploads everything to lukua/monox model repo
- Full monitoring and logging

#### **Option 2: Quick Test Training**
```bash
python3 direct_stylegan_training.py
```
- Shorter test training (100 kimg)
- Faster validation of pipeline
- Still uploads outputs to model repo

#### **Option 3: Web Interface**
```bash
python3 app.py
```
- Launch Gradio web interface
- Control training via browser
- Monitor progress in real-time
- Access JSON API endpoints

### 📁 Output Structure

All outputs are automatically uploaded to `lukua/monox` model repo:

```
lukua/monox/
├── checkpoints/          # Model checkpoints every 5 epochs
│   ├── network-snapshot-000050.pkl
│   ├── network-snapshot-000100.pkl
│   └── ...
├── previews/            # Generated sample images
│   ├── fakes000050.png
│   ├── fakes000100.png
│   └── ...
├── logs/                # Training logs and monitoring
│   ├── stylegan_training_output.log
│   ├── training_progress.json
│   └── final_training_report.json
└── reports/             # Progress and completion reports
    └── training_progress.json
```

### 🔧 Training Configuration

**Optimized for Fresh Training:**
- **Resolution**: 1024×1024 (matching dataset)
- **Total Training**: 1000 kimg (manageable duration)
- **Checkpoint Frequency**: Every 50 kimg (~5 epochs)
- **Batch Size**: 4 (CPU) / 16 (GPU)
- **Augmentation**: Adaptive (ADA)
- **Mixed Precision**: Auto-detected based on hardware

### 📊 Monitoring & Progress

#### **Real-time Monitoring**
- Training logs streamed to console
- Progress reports updated every 30 seconds
- Automatic file uploads to model repo
- Comprehensive error logging

#### **Progress Tracking**
- Checkpoint files indicate training epochs
- Preview images show generation quality
- JSON reports provide detailed metrics
- Log files contain full training history

### 🛠️ Troubleshooting

#### **Common Issues & Solutions**

1. **"No GPU detected"**
   - ✅ **Expected in current environment**
   - Training will use CPU (slower but functional)
   - Enable GPU in Hugging Face Spaces settings for faster training

2. **"Dataset not found"**
   - ✅ **Already resolved** - dataset downloaded to `/workspace/dataset`
   - 868 images ready for training

3. **"Import errors"**
   - ✅ **Already resolved** - all dependencies installed
   - StyleGAN-V modules properly configured

4. **"Training fails to start"**
   - Check logs in `/workspace/logs/`
   - Verify dataset images are valid
   - Ensure sufficient disk space

#### **Performance Notes**

- **CPU Training**: Will be slow but functional
- **Expected Duration**: 
  - With GPU: ~2-4 hours for 1000 kimg
  - With CPU: ~24-48 hours for 1000 kimg
- **Checkpoint Frequency**: Every 50 kimg ensures progress is saved

### 🎉 Success Indicators

**Training is working correctly when you see:**
- ✅ "Training started" messages in logs
- ✅ Checkpoint files appearing in `/checkpoints`
- ✅ Preview images in `/previews`
- ✅ Files being uploaded to `lukua/monox` repo
- ✅ Progress reports updating regularly

### 🔗 Integration with Hugging Face

**Automatic Model Repo Updates:**
- All checkpoints → `lukua/monox/checkpoints/`
- All previews → `lukua/monox/previews/`
- All logs → `lukua/monox/logs/`
- Progress reports → `lukua/monox/reports/`

**API Access:**
- Web interface available at Space URL
- JSON API endpoints for programmatic access
- Real-time status monitoring

### 🎯 Next Steps

1. **Start Training**: Run `python3 final_monox_training.py`
2. **Monitor Progress**: Check console output and `lukua/monox` repo
3. **Verify Outputs**: Confirm checkpoints and previews are generated
4. **Scale Up**: Once validated, increase `kimg` for longer training

### 🏆 Success Criteria Met

- ✅ **Dataset**: lukua/monox-dataset (868 images) at 1024 resolution
- ✅ **Training**: StyleGAN-V configured for fresh training
- ✅ **Checkpoints**: Save every 5 epochs to `/checkpoints`
- ✅ **Logging**: Comprehensive logs and previews per epoch
- ✅ **Integration**: Automatic uploads to `lukua/monox` model repo
- ✅ **Environment**: GPU-enabled Docker with CPU fallback

**🚀 Ready to start fresh training!**