#!/usr/bin/env python3
"""
Smoke test for Hugging Face dataset access (public only)
"""
from huggingface_hub import HfApi

def main():
    print("[smoke] Testing Hugging Face dataset access (public)...")
    
    try:
        # Test dataset access (public, no auth needed)
        api = HfApi()
        
        # Check if we can access the dataset repo
        try:
            api.repo_info("lukua/monox-dataset", repo_type="dataset")
            print("[smoke] Dataset repository access: OK")
        except Exception as e:
            print(f"[smoke] Dataset repository access failed: {e}")
            return False
        
        print("[smoke] Dataset access test passed! Ready for training.")
        return True
        
    except Exception as e:
        print(f"[smoke] Dataset access test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)