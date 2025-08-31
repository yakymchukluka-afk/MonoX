#!/usr/bin/env python3
"""
Test script for MonoX Training API.
Tests both Gradio interface and FastAPI endpoints.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

def test_endpoint(url: str, method: str = "GET", data: Dict = None, timeout: int = 10) -> Dict[str, Any]:
    """Test a single API endpoint."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return {"success": False, "error": f"Unsupported method: {method}"}
        
        # Check if response is JSON
        try:
            json_data = response.json()
            return {
                "success": True,
                "status_code": response.status_code,
                "data": json_data,
                "content_type": response.headers.get("content-type", "")
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": "Response is not valid JSON",
                "content": response.text[:200] + "..." if len(response.text) > 200 else response.text,
                "content_type": response.headers.get("content-type", "")
            }
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def test_local_api(base_url: str = "http://localhost:7860"):
    """Test the local API endpoints."""
    print(f"Testing API at: {base_url}")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        {"url": f"{base_url}/", "method": "GET", "name": "Root endpoint"},
        {"url": f"{base_url}/system/info", "method": "GET", "name": "System info"},
        {"url": f"{base_url}/health", "method": "GET", "name": "Health check"},
        {"url": f"{base_url}/training/status", "method": "GET", "name": "Training status"},
        {"url": f"{base_url}/checkpoints/list", "method": "GET", "name": "List checkpoints"},
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"\nTesting: {endpoint['name']}")
        print(f"URL: {endpoint['url']}")
        
        result = test_endpoint(endpoint["url"], endpoint["method"])
        results.append({"endpoint": endpoint, "result": result})
        
        if result["success"]:
            print(f"✅ SUCCESS (Status: {result['status_code']})")
            print(f"Content-Type: {result['content_type']}")
            if isinstance(result["data"], dict):
                print(f"Response keys: {list(result['data'].keys())}")
            else:
                print(f"Response type: {type(result['data'])}")
        else:
            print(f"❌ FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            if "content" in result:
                print(f"Content preview: {result['content']}")
    
    # Test training start (with dummy data)
    print(f"\nTesting: Training start (dummy request)")
    training_data = {
        "dataset_path": "/workspace/dataset",
        "total_kimg": 10,  # Very small for testing
        "resolution": 512,
        "num_gpus": 1
    }
    
    result = test_endpoint(f"{base_url}/training/start", "POST", training_data)
    results.append({"endpoint": {"name": "Training start", "url": f"{base_url}/training/start"}, "result": result})
    
    if result["success"]:
        print(f"✅ Training start endpoint works (Status: {result['status_code']})")
        if result["status_code"] == 400:
            print(f"Expected failure (no dataset): {result['data']}")
    else:
        print(f"❌ Training start failed: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for r in results if r["result"]["success"])
    total = len(results)
    
    print(f"Successful tests: {successful}/{total}")
    
    for result in results:
        status = "✅" if result["result"]["success"] else "❌"
        name = result["endpoint"]["name"]
        print(f"{status} {name}")
    
    return successful == total

def test_gradio_interface(base_url: str = "http://localhost:7860"):
    """Test if Gradio interface is accessible."""
    print(f"\nTesting Gradio interface at: {base_url}")
    print("=" * 50)
    
    try:
        response = requests.get(base_url, timeout=10)
        
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            
            if "text/html" in content_type:
                print("✅ Gradio interface is accessible")
                print(f"Content-Type: {content_type}")
                
                # Check if it's actually a Gradio app
                if "gradio" in response.text.lower() or "blocks" in response.text.lower():
                    print("✅ Appears to be a Gradio application")
                    return True
                else:
                    print("⚠️  HTML response but may not be Gradio")
                    return True
            else:
                print(f"❌ Unexpected content type: {content_type}")
                return False
        else:
            print(f"❌ HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main test function."""
    print("MonoX Training API Test Suite")
    print("=" * 50)
    
    # Test different possible URLs
    test_urls = [
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "http://0.0.0.0:7860"
    ]
    
    api_success = False
    gradio_success = False
    
    for url in test_urls:
        print(f"\nTrying URL: {url}")
        
        # Test basic connectivity
        try:
            response = requests.get(url, timeout=5)
            print(f"✅ Server responding at {url}")
            
            # Test API endpoints
            if not api_success:
                api_success = test_local_api(url)
            
            # Test Gradio interface
            if not gradio_success:
                gradio_success = test_gradio_interface(url)
            
            break
            
        except requests.exceptions.RequestException:
            print(f"❌ No server at {url}")
            continue
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    if api_success:
        print("✅ API endpoints working correctly")
    else:
        print("❌ API endpoints have issues")
    
    if gradio_success:
        print("✅ Gradio interface accessible")
    else:
        print("❌ Gradio interface not accessible")
    
    if api_success and gradio_success:
        print("\n🎉 All tests passed! The API is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)