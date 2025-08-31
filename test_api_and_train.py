#!/usr/bin/env python3
"""
Test the FastAPI endpoints and start training on HF Space
"""

import requests
import time

# HF Space URL
BASE_URL = "https://lukua-monox.hf.space"

def test_endpoints():
    """Test all FastAPI endpoints."""
    print("🧪 Testing FastAPI endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ GET / - Status: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"❌ GET / failed: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ GET /health - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   CUDA Available: {data.get('cuda_available')}")
            print(f"   Device Count: {data.get('device_count')}")
    except Exception as e:
        print(f"❌ GET /health failed: {e}")
    
    # Test training status
    try:
        response = requests.get(f"{BASE_URL}/training/status")
        print(f"✅ GET /training/status - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Training Status: {data.get('status')}")
            print(f"   Previews: {data.get('previews')}")
            print(f"   Checkpoints: {data.get('checkpoints')}")
    except Exception as e:
        print(f"❌ GET /training/status failed: {e}")

def start_training():
    """Start GPU training if available, otherwise CPU."""
    print("\n🚀 Starting training...")
    
    # Try GPU training first
    try:
        response = requests.post(f"{BASE_URL}/training/start/gpu")
        print(f"✅ POST /training/start/gpu - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Result: {data.get('message')}")
            return True
    except Exception as e:
        print(f"❌ GPU training failed: {e}")
    
    # Fallback to CPU training
    try:
        response = requests.post(f"{BASE_URL}/training/start/cpu")
        print(f"✅ POST /training/start/cpu - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Result: {data.get('message')}")
            return True
    except Exception as e:
        print(f"❌ CPU training failed: {e}")
    
    return False

def monitor_progress():
    """Monitor training progress."""
    print("\n📊 Monitoring training progress...")
    
    for i in range(10):  # Check 10 times
        try:
            response = requests.get(f"{BASE_URL}/training/status")
            if response.status_code == 200:
                data = response.json()
                print(f"   Check {i+1}: Status: {data.get('status')}, Previews: {data.get('previews')}, Progress: {data.get('progress_percent')}%")
            time.sleep(30)  # Wait 30 seconds between checks
        except Exception as e:
            print(f"   Check {i+1} failed: {e}")

if __name__ == "__main__":
    print("🎨 MonoX Training Test & Launch")
    print("=" * 50)
    
    # Test endpoints
    test_endpoints()
    
    # Start training
    if start_training():
        print("\n✅ Training started successfully!")
        monitor_progress()
    else:
        print("\n❌ Failed to start training")