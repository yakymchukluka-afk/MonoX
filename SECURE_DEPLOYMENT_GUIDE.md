# 🔒 Secure MonoX Training Deployment Guide

## ⚠️ Important Security Notice

**NEVER include actual HF tokens in source code or documentation!**

All previous token exposures have been resolved. This guide shows how to deploy securely.

## 🎯 Fresh Training Setup Complete

Your MonoX training pipeline is ready for fresh training from scratch:

- ✅ **Dataset**: lukua/monox-dataset (868 images at 1024×1024)
- ✅ **StyleGAN-V**: Configured and tested
- ✅ **Security**: All hardcoded tokens removed
- ✅ **Pipeline**: Fresh training with checkpoint saving every 5 epochs

## 🔐 Secure Token Configuration

### **For Hugging Face Spaces:**

1. **Go to Space Settings**:
   ```
   https://huggingface.co/spaces/lukua/monox/settings
   ```

2. **Add Repository Secret**:
   - Navigate to "Repository secrets"
   - Click "Add a new secret"
   - **Name**: `HF_TOKEN`
   - **Value**: `[PASTE_YOUR_NEW_TOKEN_HERE]`
   - **Save the secret**

3. **Restart Space**:
   - Go back to your Space
   - Click "Restart this Space"

### **For Dev Mode:**
```bash
# Set token as environment variable (replace with your actual token)
export HF_TOKEN="[YOUR_ACTUAL_TOKEN_HERE]"
```

## 🚀 Start Training Commands

### **Secure Training (Recommended)**
```bash
# Token should already be set via Space secrets or environment
python3 secure_training.py
```

### **Full Pipeline**
```bash
# Token should already be set via Space secrets or environment  
python3 final_monox_training.py
```

### **Web Interface**
```bash
# Token should already be set via Space secrets or environment
python3 app.py
```

## 📊 Training Configuration

**Fresh Training Parameters:**
- **Dataset Source**: lukua/monox-dataset
- **Image Count**: 868 images
- **Resolution**: 1024×1024
- **Training Duration**: 1000 kimg
- **Checkpoint Frequency**: Every 50 kimg (~5 epochs)
- **Output Target**: lukua/monox model repo

## 📁 Expected Outputs

Training will create these outputs in `lukua/monox`:

```
lukua/monox/
├── checkpoints/          # Model checkpoints every 5 epochs
├── previews/            # Generated sample images  
├── logs/                # Training logs and progress
└── reports/             # Progress summaries
```

## ✅ Security Validation

**Before starting training, verify:**

```bash
# Check authentication (without exposing token)
python3 -c "
import os
from huggingface_hub import whoami

token = os.environ.get('HF_TOKEN')
if token and token.startswith('hf_'):
    try:
        user_info = whoami(token=token)
        print(f'✅ Authenticated as: {user_info[\"name\"]}')
    except Exception as e:
        print(f'❌ Authentication failed: {e}')
else:
    print('❌ HF_TOKEN not properly configured')
"
```

## 🛡️ Security Best Practices Applied

- ✅ **No hardcoded tokens** in any source files
- ✅ **Environment variable authentication** only
- ✅ **Secure documentation** without token exposure
- ✅ **Authentication validation** before operations
- ✅ **Error handling** for missing tokens

## 🎉 Ready for Production

Your MonoX training pipeline is now:
- 🔒 **Completely Secure**: No token exposure anywhere
- 🎯 **Fully Configured**: Fresh training from scratch ready
- 📊 **Monitored**: Real-time progress tracking
- 📤 **Integrated**: Auto-upload to model repo

**Start training securely with proper token configuration!**

---

**🚨 Remember: Always set HF_TOKEN via environment variables or Spaces secrets - never in code!**