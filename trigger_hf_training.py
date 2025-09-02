#!/usr/bin/env python3
"""
Trigger training on HuggingFace Space
"""

import requests
import time
from huggingface_hub import HfApi
import os

# HF Space details
SPACE_NAME = "lukua/monox"
HF_TOKEN = "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"
SPACE_URL = "https://lukua-monox.hf.space"

def check_space_status():
    """Check if the HF Space is running."""
    try:
        # Try different endpoints
        endpoints_to_try = [
            f"{SPACE_URL}",
            f"{SPACE_URL}/",
            f"{SPACE_URL}/docs",
            f"{SPACE_URL}/health",
            "https://huggingface.co/spaces/lukua/monox"
        ]
        
        for endpoint in endpoints_to_try:
            print(f"🔍 Checking: {endpoint}")
            try:
                response = requests.get(endpoint, timeout=10)
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    print(f"   ✅ Success! Content length: {len(response.text)}")
                    if "gradio" in response.text.lower() or "monox" in response.text.lower():
                        return True, endpoint
                else:
                    print(f"   ❌ Failed with status {response.status_code}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return False, None
    except Exception as e:
        print(f"❌ Error checking space: {e}")
        return False, None

def trigger_training_via_api():
    """Try to trigger training via API calls."""
    print("🚀 Attempting to trigger training via API...")
    
    # Try different API endpoints that might exist
    api_endpoints = [
        f"{SPACE_URL}/training/start/gpu",
        f"{SPACE_URL}/api/training/start",
        f"{SPACE_URL}/start_gpu_training",
        f"{SPACE_URL}/gpu_training",
        f"{SPACE_URL}/train"
    ]
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    for endpoint in api_endpoints:
        print(f"🔍 Trying: {endpoint}")
        try:
            # Try POST
            response = requests.post(endpoint, headers=headers, timeout=30)
            print(f"   POST Status: {response.status_code}")
            if response.status_code in [200, 201, 202]:
                print(f"   ✅ Success! Response: {response.text}")
                return True
            
            # Try GET
            response = requests.get(endpoint, headers=headers, timeout=30)
            print(f"   GET Status: {response.status_code}")
            if response.status_code in [200, 201, 202]:
                print(f"   ✅ Success! Response: {response.text}")
                return True
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def use_hf_api():
    """Use HuggingFace API to interact with the space."""
    print("🔧 Using HuggingFace API...")
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Get space info
        space_info = api.space_info(SPACE_NAME)
        print(f"📊 Space Status: {space_info.runtime.stage if space_info.runtime else 'Unknown'}")
        print(f"📊 Space Hardware: {space_info.runtime.hardware if space_info.runtime else 'Unknown'}")
        
        # Try to restart the space to trigger any startup scripts
        print("🔄 Attempting to restart space...")
        api.restart_space(SPACE_NAME)
        print("✅ Space restart requested")
        
        return True
    except Exception as e:
        print(f"❌ HF API Error: {e}")
        return False

def main():
    """Main function to trigger training."""
    print("🎯 MonoX Training Launcher")
    print("=" * 50)
    
    # Check space status
    is_running, working_url = check_space_status()
    
    if is_running:
        print(f"✅ Space is running at: {working_url}")
        
        # Try to trigger training
        if trigger_training_via_api():
            print("🎉 Training triggered successfully!")
            return True
    else:
        print("⚠️  Space appears to be down or not accessible")
    
    # Try HF API approach
    if use_hf_api():
        print("🔄 Attempted to restart space via HF API")
        print("⏳ Waiting 30 seconds for restart...")
        time.sleep(30)
        
        # Check again after restart
        is_running, working_url = check_space_status()
        if is_running:
            print(f"✅ Space is now running at: {working_url}")
            return trigger_training_via_api()
    
    print("❌ Could not trigger training via HF Space")
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Recommendation: Try running training locally instead")
        print("   Command: python3 gpu_gan_training.py")