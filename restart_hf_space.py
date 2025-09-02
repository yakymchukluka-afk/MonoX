#!/usr/bin/env python3
"""
Restart HuggingFace Space and check status
"""

from huggingface_hub import HfApi, login
import os
import time
import requests

def restart_space():
    """Restart the HF Space."""
    
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found!")
        return False
    
    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        print("✅ HuggingFace authentication successful")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    repo_id = "lukua/monox"
    
    try:
        # Get space info
        space_info = api.space_info(repo_id)
        print(f"📊 Space Status: {space_info.runtime.stage if space_info.runtime else 'Unknown'}")
        print(f"📊 Space Hardware: {space_info.runtime.hardware if space_info.runtime else 'Unknown'}")
        
        # Restart the space
        print("🔄 Restarting space...")
        api.restart_space(repo_id)
        print("✅ Space restart requested")
        
        # Wait and check status
        for i in range(6):  # Check for 3 minutes
            time.sleep(30)
            print(f"⏳ Waiting... ({i+1}/6)")
            
            try:
                response = requests.get("https://lukua-monox.hf.space/", timeout=10)
                if response.status_code == 200:
                    print("🎉 Space is now accessible!")
                    return True
                else:
                    print(f"📊 Status: {response.status_code}")
            except Exception as e:
                print(f"📊 Still not ready: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Space restart failed: {e}")
        return False

if __name__ == "__main__":
    success = restart_space()
    if success:
        print("\n🚀 Space is ready!")
        print("🔗 Visit: https://huggingface.co/spaces/lukua/monox")
    else:
        print("\n⚠️ Space may need manual intervention")
        print("💡 Try visiting the Space directly and check the logs")