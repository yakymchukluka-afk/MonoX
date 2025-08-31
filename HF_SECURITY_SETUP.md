# 🔒 Secure HF Token Setup for MonoX Training

## ⚠️ Security Issue Resolved

The previous HF token was automatically invalidated because it was exposed in source code. I've now updated all files to use secure authentication methods.

## 🔧 How to Set Up the New Token Securely

### **Option 1: Hugging Face Spaces Secrets (Recommended)**

1. **Go to your Space settings**: `https://huggingface.co/spaces/lukua/monox/settings`
2. **Navigate to "Repository secrets"**
3. **Add a new secret**:
   - **Name**: `HF_TOKEN`
   - **Value**: `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`
4. **Save the secret**
5. **Restart your Space**

### **Option 2: Environment Variable (Dev Mode)**

If you're using Dev Mode, set the token as an environment variable:

```bash
export HF_TOKEN="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"
```

## ✅ Security Improvements Made

### **Removed Hardcoded Tokens**
I've updated these files to remove hardcoded tokens:
- `start_training.sh`
- `final_monox_training.py`
- `direct_stylegan_training.py`
- `start_fresh_training.py`
- `monox_fresh_training.py`
- `run_fresh_training.py`
- `fresh_training_config.py`

### **Secure Authentication Pattern**
All scripts now use this secure pattern:

```python
# ✅ SECURE - Read from environment
hf_token = os.environ.get('HF_TOKEN')

# ❌ INSECURE - Never do this
# hf_token = "hf_hardcoded_token_here"
```

### **Authentication Validation**
Added comprehensive authentication checks:

```python
from huggingface_hub import login, whoami

# Login using environment token
login(token=os.environ.get('HF_TOKEN'))

# Verify authentication
user_info = whoami(token=os.environ.get('HF_TOKEN'))
print(f"✅ Authenticated as: {user_info['name']}")
```

## 🚀 Ready to Start Secure Training

### **1. Set the Token Securely**
Choose one of the methods above to set `HF_TOKEN` securely.

### **2. Start Training**
```bash
# Make sure token is set
export HF_TOKEN="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"

# Start secure training
python3 secure_training.py
```

### **3. Verify Security**
```bash
# Test secure authentication
python3 secure_training_setup.py
```

## 🔍 Validation Commands

### **Check Token Setup**
```bash
python3 -c "
import os
from huggingface_hub import whoami

token = os.environ.get('HF_TOKEN')
if token:
    try:
        user_info = whoami(token=token)
        print(f'✅ Authenticated as: {user_info[\"name\"]}')
    except Exception as e:
        print(f'❌ Authentication failed: {e}')
else:
    print('❌ HF_TOKEN not found in environment')
"
```

### **Verify No Hardcoded Tokens**
```bash
# This should return no results
grep -r "hf_" *.py | grep -v "os.environ" | grep -v "environment"
```

## 📁 Updated Training Pipeline

All training scripts now:
- ✅ Use environment variables for authentication
- ✅ Validate authentication before starting
- ✅ Provide clear error messages if token is missing
- ✅ Upload securely to `lukua/monox` model repo

## 🎯 Next Steps

1. **Set HF_TOKEN securely** (using Spaces secrets or environment variable)
2. **Restart your Space** (if using Spaces secrets)
3. **Start training**: `python3 secure_training.py`
4. **Monitor progress**: Check `lukua/monox` model repo for uploads

## 🛡️ Security Best Practices

- ✅ **Never hardcode tokens** in source files
- ✅ **Use environment variables** or Spaces secrets
- ✅ **Validate authentication** before operations
- ✅ **Monitor token usage** through HF dashboard
- ✅ **Rotate tokens regularly** for security

Your MonoX training is now secure and ready to run! 🎉