#!/usr/bin/env python3
"""
📥 MonoX Google Drive Bulk Download Script
==========================================

This script helps download your MonoX training data from Google Drive
for migration to Hugging Face.

Usage:
1. Get your Google Drive folder/file IDs
2. Run this script to download everything locally
3. Use the migration script to upload to Hugging Face
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json

def download_gdrive_file(file_id, output_path):
    """Download a single file from Google Drive."""
    try:
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        print(f"📥 Downloading: {output_path}")
        urllib.request.urlretrieve(url, output_path)
        
        # Check if it's a zip file and extract
        if output_path.endswith('.zip'):
            print(f"📂 Extracting: {output_path}")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                extract_dir = output_path.replace('.zip', '')
                zip_ref.extractall(extract_dir)
            os.remove(output_path)  # Remove zip after extraction
            return extract_dir
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error downloading {file_id}: {e}")
        return None

def extract_file_id_from_url(url):
    """Extract file ID from various Google Drive URL formats."""
    if "/file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    elif "id=" in url:
        return url.split("id=")[1].split("&")[0]
    else:
        raise ValueError("Invalid Google Drive URL format")

def download_monox_data():
    """Interactive download of MonoX data from Google Drive."""
    print("🎨 MonoX Google Drive Download Helper")
    print("=" * 50)
    
    # Create download directory
    download_dir = Path("./monox_gdrive_download")
    download_dir.mkdir(exist_ok=True)
    
    print(f"📁 Download directory: {download_dir.absolute()}")
    
    # Download categories
    categories = {
        "dataset": {
            "description": "Training dataset (800+ monotype images)",
            "default_filename": "monox_dataset"
        },
        "checkpoints": {
            "description": "Model checkpoints (.pth files)",
            "default_filename": "monox_checkpoints"
        },
        "samples": {
            "description": "Generated samples/results",
            "default_filename": "monox_samples"
        }
    }
    
    downloaded_items = {}
    
    for category, info in categories.items():
        print(f"\n📂 {category.upper()}: {info['description']}")
        
        while True:
            url = input(f"Enter Google Drive URL for {category} (or 'skip'): ").strip()
            
            if url.lower() == 'skip':
                print(f"⏭️ Skipping {category}")
                break
                
            if not url:
                continue
                
            try:
                file_id = extract_file_id_from_url(url)
                output_path = download_dir / f"{info['default_filename']}.zip"
                
                result = download_gdrive_file(file_id, str(output_path))
                if result:
                    downloaded_items[category] = result
                    print(f"✅ {category} downloaded successfully")
                    break
                else:
                    print(f"❌ Failed to download {category}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please check the URL format")
    
    # Create summary
    summary = {
        "download_directory": str(download_dir.absolute()),
        "downloaded_items": downloaded_items,
        "next_steps": [
            "1. Verify downloaded content",
            "2. Run migrate_monox_to_hf.py to create HF repositories",
            "3. Upload content to Hugging Face using HF CLI or web interface"
        ]
    }
    
    summary_file = download_dir / "download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Download Summary:")
    print(f"📁 Location: {download_dir.absolute()}")
    for category, path in downloaded_items.items():
        print(f"✅ {category}: {path}")
    
    print(f"\n📄 Summary saved to: {summary_file}")
    print("\n🚀 Next Steps:")
    print("1. Verify your downloaded content")
    print("2. Run: python migrate_monox_to_hf.py")
    print("3. Upload to HF using the migration instructions")
    
    return downloaded_items

if __name__ == "__main__":
    try:
        download_monox_data()
    except KeyboardInterrupt:
        print("\n⏹️ Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)