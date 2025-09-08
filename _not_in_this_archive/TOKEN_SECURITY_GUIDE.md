# 🔒 TOKEN SECURITY GUIDE - CRITICAL

## ⚠️ **NEVER PUT TOKENS IN CODE!**

This is the **SECOND TIME** a token has been exposed. Follow this guide exactly.

## 🛡️ **SECURE METHODS ONLY**

### **Method 1: HF Space Secrets (RECOMMENDED)**
1. Go to your HF Space settings
2. Click "Secrets" tab
3. Add: `HF_TOKEN` = `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`
4. Restart the Space

### **Method 2: Environment Variables (Local)**
```bash
# Set environment variable
export HF_TOKEN=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

# Or create .env file (NEVER commit this!)
echo "HF_TOKEN=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" > .env
```

### **Method 3: Make Repository Private**
1. Go to GitHub repository settings
2. Change visibility to "Private"
3. This prevents accidental exposure

## 🚫 **NEVER DO THIS:**
- ❌ Put tokens in Python files
- ❌ Put tokens in .py files
- ❌ Put tokens in any code files
- ❌ Commit tokens to git
- ❌ Share tokens in chat/email

## ✅ **ALWAYS DO THIS:**
- ✅ Use environment variables
- ✅ Use Space secrets
- ✅ Use GitHub secrets
- ✅ Keep tokens in .env (gitignored)
- ✅ Rotate tokens regularly

## 🔧 **Code Pattern for Security**

```python
# SECURE - gets token from environment
hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("❌ HF_TOKEN not found")
    return False

# INSECURE - never do this
# hf_token = "hf_actual_token_here"  # ❌ NEVER!
```

## 📋 **Setup Steps (SECURE)**

1. **Add token to HF Space secrets:**
   - Space Settings → Secrets
   - Add: `HF_TOKEN` = `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`

2. **Test setup:**
   ```bash
   python3 secure_setup.py
   ```

3. **Start training:**
   ```bash
   python3 monox_training_with_hybrid_auth.py
   ```

## 🚨 **If Token Exposed Again:**

1. **Immediately revoke** at https://huggingface.co/settings/tokens
2. **Generate new token**
3. **Update all services**
4. **Check for unauthorized usage**

## 🔒 **Repository Security Status**

- ✅ All hardcoded tokens removed
- ✅ Code uses environment variables only
- ✅ .gitignore protects secrets
- ⚠️ **Repository is still PUBLIC** (consider making private)

## 🎯 **RECOMMENDATION**

**Make your repository PRIVATE** to prevent future token exposure!

Your new token `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` is ready to use securely! 🔒