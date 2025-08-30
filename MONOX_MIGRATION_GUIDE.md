# 🚀 MonoX Complete Migration Guide: Google Drive → Hugging Face

This guide walks you through migrating your entire MonoX project from Google Drive to Hugging Face, creating a fully integrated ecosystem.

## 📋 Migration Overview

**What we're migrating:**
- 🎨 **Training Dataset**: 800+ monotype images → HF Dataset
- 🧠 **Model Checkpoints**: All training checkpoints → HF Model Repository  
- 🌐 **Interactive Space**: Already set up at `lukua/monox`

**Final Result:**
- Fully integrated Hugging Face ecosystem
- No more Google Drive dependencies
- Seamless training continuation
- Public access to datasets and models

## 🔧 Prerequisites

1. **Hugging Face Account** with write access
2. **Google Drive** with MonoX_training folder
3. **Local Environment** with Python and git

## 📝 Step-by-Step Migration

### Step 1: Authenticate with Hugging Face

```bash
# Install HF CLI (if not already installed)
pip install --break-system-packages -U "huggingface_hub[cli]"

# Login to Hugging Face
hf auth login
# Enter your HF token when prompted
```

### Step 2: Download Your Google Drive Data

```bash
# Run the download helper script
python download_from_gdrive.py
```

**You'll be prompted for:**
- Dataset folder URL (your 800+ monotype images)
- Checkpoints folder URL (all your .pth files)
- Samples folder URL (generated images - optional)

**Google Drive URLs formats supported:**
- `https://drive.google.com/file/d/FILE_ID/view`
- `https://drive.google.com/open?id=FILE_ID`
- `https://drive.google.com/drive/folders/FOLDER_ID`

### Step 3: Create Hugging Face Repositories

```bash
# Run the migration script to create repositories
python migrate_monox_to_hf.py
```

This creates:
- `lukua/monox-dataset` (dataset repository)
- `lukua/monox-models` (model repository)

### Step 4: Upload Dataset

```bash
# Upload your training images
hf upload lukua/monox-dataset ./monox_gdrive_download/monox_dataset --repo-type=dataset

# Or upload individual folders:
hf upload lukua/monox-dataset ./path/to/images --repo-type=dataset
```

### Step 5: Upload Model Checkpoints

```bash
# Upload all checkpoints
hf upload lukua/monox-models ./monox_gdrive_download/monox_checkpoints

# Or upload individual checkpoints:
hf upload lukua/monox-models ./path/to/checkpoint.pth
```

### Step 6: Verify Migration

Check your repositories:
- 🎨 **Dataset**: https://huggingface.co/datasets/lukua/monox-dataset
- 🧠 **Models**: https://huggingface.co/lukua/monox-models  
- 🌐 **Space**: https://huggingface.co/spaces/lukua/monox

## 🔄 Alternative: Manual Upload via Web Interface

If CLI upload fails, use the web interface:

### For Dataset:
1. Go to https://huggingface.co/datasets/lukua/monox-dataset
2. Click "Add file" → "Upload files"
3. Drag and drop your images

### For Models:
1. Go to https://huggingface.co/lukua/monox-models
2. Click "Add file" → "Upload files"  
3. Upload your .pth checkpoint files

## 🎯 Post-Migration Benefits

### Integrated Training
- Resume training directly in the Space
- Automatic checkpoint detection
- No Google Drive dependencies

### Public Access
- Share datasets and models with the community
- Reproducible research
- Easy collaboration

### Seamless Workflow
```bash
# Everything in one place:
Dataset: lukua/monox-dataset
Models: lukua/monox-models  
Space: lukua/monox
```

## 🚨 Important Notes

### File Size Limits
- **Git LFS**: Automatically handles large files
- **Individual files**: Up to 50GB per file
- **Repository**: No total size limit

### Naming Convention
Keep your checkpoint naming consistent:
- `monox_generator_00050.pth` (epoch 50)
- `monox_generator_01000.pth` (epoch 1000)
- etc.

### Training Continuation
After migration, training will:
1. Auto-detect latest checkpoint from HF models
2. Resume from the exact epoch
3. Save new checkpoints to HF models
4. Generate samples accessible via the Space

## 🔧 Troubleshooting

### Large File Upload Issues
```bash
# Install git-lfs if needed
git lfs install

# For very large files, use chunked upload
hf upload lukua/monox-models large_checkpoint.pth --chunk-size=100MB
```

### Authentication Issues
```bash
# Re-authenticate
hf auth logout
hf auth login
```

### Repository Access
Make sure repositories are public or you have access rights.

## 🎉 Success Indicators

✅ **Dataset uploaded**: Images visible at lukua/monox-dataset  
✅ **Models uploaded**: Checkpoints visible at lukua/monox-models  
✅ **Space updated**: Training works in lukua/monox  
✅ **Integration working**: Can resume training from HF checkpoints

## 📞 Support

If you encounter issues:
1. Check the error logs
2. Verify file formats and sizes
3. Ensure proper authentication
4. Try manual web interface upload as fallback

Your MonoX project will be fully integrated in the Hugging Face ecosystem! 🎨✨