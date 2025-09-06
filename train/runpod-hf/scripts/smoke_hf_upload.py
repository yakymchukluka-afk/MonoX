#!/usr/bin/env python3
"""
Smoke test for Hugging Face upload functionality
"""
import os
from huggingface_hub import HfApi

def main():
    print("[smoke] Testing Hugging Face authentication...")
    
    # Check if HF_TOKEN is set
    if not os.getenv('HF_TOKEN'):
        print("ERROR: HF_TOKEN not set. Please set RUNPOD_SECRET_HF_token environment variable.")
        return False
    
    try:
        # Test authentication
        api = HfApi()
        user = api.whoami()
        print(f"[smoke] Successfully authenticated as: {user['name']}")
        
        # Test repository access
        print("[smoke] Testing repository access...")
        
        # Check if we can access the dataset repo
        try:
            api.repo_info("lukua/monox-dataset", repo_type="model")
            print("[smoke] Dataset repository access: OK")
        except Exception as e:
            print(f"[smoke] Dataset repository access failed: {e}")
            return False
        
        # Check if we can access the model repo
        try:
            api.repo_info("lukua/monox-model", repo_type="model")
            print("[smoke] Model repository access: OK")
        except Exception as e:
            print(f"[smoke] Model repository access failed: {e}")
            return False
        
        print("[smoke] All tests passed! Ready for training.")
        return True
        
    except Exception as e:
        print(f"[smoke] Authentication failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)