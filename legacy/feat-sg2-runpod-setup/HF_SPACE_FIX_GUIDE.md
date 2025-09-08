# MonoX HF Space Fix Guide

## Problem Analysis

Based on the error logs, you're experiencing two main issues:

### 1. Authentication Error (401)
```
401 Client Error. (Request ID: Root=1-68b8b0d2-1446202a290b5d86582f9740;fa0b3eeb-c40d-4e1c-abab-2e628821a0e1)
Repository Not Found for url: https://huggingface.co/api/models/lukua/monox-model/preupload/main.
Invalid username or password.
```

### 2. Repository Path Issues
- The error references `lukua/monox-model/preupload/main` (doesn't exist)
- Your code targets `lukua/monox` (different repository)
- The "preupload" path is HF's internal upload mechanism

## Root Causes

1. **Model Repository Changed to Private**: You changed `lukua/monox-model` from public to private
2. **Missing Authentication**: The HF Space lacks proper authentication to access the private repo
3. **Repository Mismatch**: Code targets wrong repository (`lukua/monox` vs `lukua/monox-model`)

## Solutions

### Step 1: Fix Authentication

1. **Generate HF Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create new token with "Write" permissions
   - Copy the token

2. **Add Token to Space**:
   - Go to your HF Space settings
   - Add secret: `HF_TOKEN` = `your_token_here`
   - Restart the Space

### Step 2: Fix Repository Paths

The current code uploads to `lukua/monox` but should upload to `lukua/monox-model`.

**Files to update**:
- `simple_gan_training.py` (line 139)
- `monitor_training.py` (line 71)

Change:
```python
repo_id="lukua/monox"
```
To:
```python
repo_id="lukua/monox-model"
```

### Step 3: Use Fixed Training Script

I've created `fixed_training_script.py` that:
- ✅ Handles authentication properly
- ✅ Uses correct repository (`lukua/monox-model`)
- ✅ Uploads to correct paths (`samples/`, `checkpoints/`, `logs/`)
- ✅ Provides proper error handling

### Step 4: Run Setup Script

```bash
python setup_hf_space.py
```

This will:
- Set up proper directories
- Configure HF authentication
- Test upload functionality
- Create proper configuration files

## Quick Fix Commands

```bash
# 1. Run the setup
python setup_hf_space.py

# 2. Test the fixed training
python fixed_training_script.py

# 3. Check uploads
ls -la samples/ checkpoints/ logs/
```

## Verification

After applying fixes, you should see:
- ✅ No more 401 errors
- ✅ Successful uploads to `lukua/monox-model`
- ✅ Files uploaded to correct paths:
  - `samples/monox_epoch_XXXXX.png`
  - `checkpoints/monox_checkpoint_epoch_XXXXX.pth`
  - `logs/training_log_epoch_XXXXX.txt`

## Files Created

1. `fix_hf_authentication.py` - Authentication fix
2. `fixed_training_script.py` - Working training script
3. `setup_hf_space.py` - Complete setup script
4. `HF_SPACE_FIX_GUIDE.md` - This guide

## Next Steps

1. Add `HF_TOKEN` secret to your Space
2. Restart the Space
3. Run `python setup_hf_space.py`
4. Run `python fixed_training_script.py`
5. Check `lukua/monox-model` repository for uploaded files

The training should now work properly with automatic uploads to your private model repository!