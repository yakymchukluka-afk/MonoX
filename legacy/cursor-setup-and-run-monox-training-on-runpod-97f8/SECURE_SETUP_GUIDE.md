# 🔒 SECURE SETUP GUIDE - MonoX Training

## ⚠️ CRITICAL: Token Security

**NEVER put tokens directly in code!** Always use environment variables or secrets.

## 🛡️ Proper Token Management

### 1. **For Local Development**
```bash
# Create .env file (NEVER commit this!)
cp .env.example .env

# Edit .env with your token
echo "HF_TOKEN=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" >> .env

# Load environment variables
export $(cat .env | xargs)
```

### 2. **For Hugging Face Spaces**
1. Go to your Space settings
2. Click "Secrets" tab
3. Add secret: `HF_TOKEN` = `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`
4. Restart the Space

### 3. **For GitHub Actions**
```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

## 🔐 Security Best Practices

### ✅ **DO:**
- Use environment variables
- Use Space secrets
- Use GitHub secrets
- Keep tokens in `.env` files (gitignored)
- Rotate tokens regularly

### ❌ **DON'T:**
- Put tokens in code
- Commit `.env` files
- Share tokens in chat/email
- Use tokens in public repositories

## 🚨 **If Token is Exposed:**

1. **Immediately revoke** the exposed token
2. **Generate new token** at https://huggingface.co/settings/tokens
3. **Update all services** with new token
4. **Check logs** for unauthorized usage

## 🔧 **Updated Code Structure**

All code now uses:
```python
# Secure - gets token from environment
hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("❌ HF_TOKEN not found")
    return False
```

## 📋 **Setup Steps**

1. **Add token to Space secrets:**
   - Space Settings → Secrets
   - Add: `HF_TOKEN` = `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`

2. **Test authentication:**
   ```bash
   python3 monox_hybrid_auth.py
   ```

3. **Start training:**
   ```bash
   python3 monox_training_with_hybrid_auth.py
   ```

## 🛡️ **Repository Security**

- ✅ All tokens removed from code
- ✅ `.gitignore` updated to protect secrets
- ✅ Environment variable usage implemented
- ✅ Secure setup guide created

Your repository is now secure! 🔒