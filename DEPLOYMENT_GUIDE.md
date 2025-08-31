# MonoX Training - Deployment Guide

## Problem Resolution Summary

The original issue was that your Hugging Face Space was serving HTML instead of JSON API responses, causing the error: `"Unexpected token '<', DOCTYPE..." is not valid JSON`.

## Root Cause
- No proper web application (`app.py`) was configured for Hugging Face Spaces
- The space was auto-generating a basic HTML interface
- No JSON API endpoints were available for programmatic access

## Solution Implemented

### 1. Created Main Application (`app.py`)
- **Full Gradio Interface**: Complete web UI with tabs for system status, training control, logs, and checkpoints
- **JSON API Support**: Built-in API endpoints that return proper JSON responses
- **Training Management**: Integration with your existing `src/infra/launch.py` training launcher
- **Real-time Monitoring**: Live training logs and status updates

### 2. Created Fallback Application (`minimal_app.py`)
- **Works Without Dependencies**: Functions even if Gradio/PyTorch aren't available
- **Simple HTTP Server**: Basic web interface with JSON API endpoints
- **Diagnostic Information**: Clear status reporting and troubleshooting info
- **Graceful Degradation**: Falls back to basic functionality when full stack isn't available

### 3. Updated Configuration Files
- **Updated `requirements.txt`**: Added Gradio, FastAPI, and Uvicorn dependencies
- **Created `README.md`**: Proper Hugging Face Space configuration with metadata
- **Added `startup.py`**: Environment setup and diagnostic checks

### 4. API Endpoints Available

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | Web interface (HTML) | HTML page |
| `/system/info` | GET | System status and GPU info | JSON |
| `/training/start` | POST | Start training with config | JSON |
| `/training/stop` | POST | Stop current training | JSON |
| `/training/status` | GET | Current training status | JSON |
| `/training/logs` | GET | Training logs | JSON |
| `/checkpoints/list` | GET | Available checkpoints | JSON |
| `/health` | GET | Health check | JSON |

## Deployment Steps

### For Hugging Face Spaces:

1. **Upload Files**: Ensure these files are in your space:
   - `app.py` (main application)
   - `minimal_app.py` (fallback)
   - `requirements.txt` (updated with web dependencies)
   - `README.md` (with proper HF metadata)

2. **Space Configuration**: Your `README.md` now includes:
   ```yaml
   ---
   title: MonoX StyleGAN-V Training
   sdk: gradio
   sdk_version: 4.44.0
   app_file: app.py
   ---
   ```

3. **Dependencies**: The space will automatically install:
   - `gradio>=4.0.0`
   - `fastapi>=0.100.0` 
   - `uvicorn>=0.20.0`
   - Plus your existing ML dependencies

### Testing the Fix:

#### ✅ Web Interface Access:
```
https://your-space-url/
```

#### ✅ JSON API Access:
```bash
# System information
curl https://your-space-url/system/info

# Training status  
curl https://your-space-url/training/status

# Start training
curl -X POST https://your-space-url/training/start \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "/workspace/dataset", "total_kimg": 1000}'
```

## GPU Detection Fix

The warning `"NVIDIA Driver was not detected"` is addressed by:

1. **Environment Variables**: Set proper CUDA debugging flags
2. **Graceful Degradation**: System works with or without GPU
3. **Clear Status Reporting**: API shows exact GPU availability
4. **Diagnostic Information**: Detailed system info for troubleshooting

## Troubleshooting

### If you still get HTML instead of JSON:

1. **Check URL**: Use `/system/info` not `/` for JSON
2. **Check Headers**: Send `Content-Type: application/json` for POST requests
3. **Check Method**: Use correct HTTP method (GET/POST)

### If training doesn't start:

1. **Check Dependencies**: Ensure PyTorch is available
2. **Check Dataset**: Upload training images to `/dataset` directory
3. **Check Logs**: Use `/training/logs` endpoint to see detailed errors
4. **Check GPU**: Use `/system/info` to verify GPU availability

### If the space fails to start:

1. **Use Fallback**: The `minimal_app.py` works without dependencies
2. **Check Requirements**: Ensure `requirements.txt` is valid
3. **Check Syntax**: Validate Python syntax in uploaded files

## Next Steps

1. **Deploy to HF Spaces**: Upload the files and restart your space
2. **Test API Endpoints**: Use the provided curl commands or the web interface
3. **Upload Training Data**: Add your images to the `/dataset` directory
4. **Start Training**: Use either the web UI or API endpoints

The API will now return proper JSON responses instead of HTML, resolving your original error.