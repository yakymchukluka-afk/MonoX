#!/usr/bin/env python3
"""
Trigger training via FastAPI endpoints locally
"""

import subprocess
import time
import json
from pathlib import Path

def call_fastapi_endpoint(endpoint, method="GET", data=None):
    """Call FastAPI endpoint using curl."""
    url = f"http://localhost:7860{endpoint}"
    
    if method == "GET":
        cmd = ["curl", "-s", url]
    else:
        cmd = ["curl", "-s", "-X", method, url]
        if data:
            cmd.extend(["-H", "Content-Type: application/json", "-d", json.dumps(data)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"error": result.stderr}
    except Exception as e:
        return {"error": str(e)}

def main():
    """Test FastAPI and start training."""
    print("🧪 Testing FastAPI endpoints...")
    
    # Test root endpoint
    response = call_fastapi_endpoint("/")
    print(f"GET / : {response}")
    
    # Test health endpoint  
    response = call_fastapi_endpoint("/health")
    print(f"GET /health : {response}")
    
    # Test training status
    response = call_fastapi_endpoint("/training/status")
    print(f"GET /training/status : {response}")
    
    # Start GPU training
    print("\n🚀 Starting GPU training...")
    response = call_fastapi_endpoint("/training/start/gpu", "POST")
    print(f"POST /training/start/gpu : {response}")
    
    # If GPU failed, try CPU
    if response.get("status") != "success":
        print("🔄 Trying CPU training...")
        response = call_fastapi_endpoint("/training/start/cpu", "POST")
        print(f"POST /training/start/cpu : {response}")
    
    # Monitor for a bit
    print("\n📊 Monitoring progress...")
    for i in range(5):
        time.sleep(10)
        response = call_fastapi_endpoint("/training/status")
        print(f"Status check {i+1}: {response}")

if __name__ == "__main__":
    main()