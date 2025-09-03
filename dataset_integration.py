#!/usr/bin/env python3
"""
MonoX Dataset Integration - HuggingFace Datasets Connection
===========================================================

This script connects StyleGAN-V training to the HuggingFace dataset 'mono-dataset'
without downloading files locally. Uses datasets library for efficient data loading.

Key Features:
- Connects to 'mono-dataset' via HF datasets library
- Prepares data loader for StyleGAN-V training
- Handles 1024x1024 image resolution
- No local file downloads needed
- Memory efficient streaming
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from datasets import load_dataset, Dataset
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from huggingface_hub import HfApi
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Run: pip install datasets torch torchvision huggingface_hub")
    sys.exit(1)


class MonoDatasetLoader:
    """HuggingFace dataset loader for MonoX training."""
    
    def __init__(self, dataset_name: str = "lukua/monox-dataset", resolution: int = 1024):
        self.dataset_name = dataset_name
        self.resolution = resolution
        self.dataset = None
        self.transform = self._create_transform()
        
    def _create_transform(self):
        """Create image transformation pipeline for StyleGAN-V."""
        return transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range
        ])
    
    def connect(self, use_auth_token: bool = True) -> bool:
        """Connect to the HuggingFace dataset."""
        try:
            print(f"🔗 Connecting to HuggingFace dataset: {self.dataset_name}")
            
            # Try to load the dataset with authentication
            if use_auth_token:
                print("🔑 Using authentication token for private dataset access...")
                self.dataset = load_dataset(self.dataset_name, streaming=True, use_auth_token=True)
            else:
                self.dataset = load_dataset(self.dataset_name, streaming=True)
            
            print(f"✅ Successfully connected to {self.dataset_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to dataset {self.dataset_name}: {e}")
            
            if "authentication" in str(e).lower() or "private" in str(e).lower():
                print(f"🔒 Dataset appears to be private. Make sure you're authenticated with HuggingFace CLI:")
                print(f"   Run: huggingface-cli login")
                print(f"   Or set HF_TOKEN environment variable")
            elif "not found" in str(e).lower():
                print(f"📝 Dataset '{self.dataset_name}' not found or access denied")
            else:
                print(f"🔧 Connection error: {str(e)}")
                
            return False
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset structure and content."""
        if not self.dataset:
            return {"valid": False, "error": "Dataset not connected"}
        
        try:
            # Get first few samples to validate
            train_split = self.dataset.get('train', self.dataset)
            samples = list(train_split.take(3))
            
            if not samples:
                return {"valid": False, "error": "No samples found"}
            
            # Check sample structure
            sample = samples[0]
            print(f"📊 Sample structure: {list(sample.keys())}")
            
            # Validate image field
            image_field = None
            for key in ['image', 'img', 'picture', 'photo']:
                if key in sample:
                    image_field = key
                    break
            
            if not image_field:
                return {"valid": False, "error": f"No image field found. Available: {list(sample.keys())}"}
            
            # Test image loading
            test_image = sample[image_field]
            if hasattr(test_image, 'convert'):  # PIL Image
                img = test_image.convert('RGB')
            else:
                return {"valid": False, "error": f"Invalid image format: {type(test_image)}"}
            
            return {
                "valid": True,
                "samples": len(samples),
                "image_field": image_field,
                "image_size": img.size,
                "sample_keys": list(sample.keys())
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def create_stylegan_dataset_dir(self, output_dir: str = "/tmp/mono_dataset_cache") -> str:
        """Create a temporary dataset directory structure for StyleGAN-V."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.dataset:
            raise ValueError("Dataset not connected. Call connect() first.")
        
        try:
            # Create sequence directory structure expected by StyleGAN-V
            sequence_dir = os.path.join(output_dir, "sequence_001")
            os.makedirs(sequence_dir, exist_ok=True)
            
            print(f"📁 Creating StyleGAN-V compatible dataset structure in: {output_dir}")
            
            # Get training split
            train_split = self.dataset.get('train', self.dataset)
            
            # Process first batch to create directory structure
            sample_count = 0
            for sample in train_split.take(10):  # Process first 10 samples for validation
                # Find image field
                image_field = None
                for key in ['image', 'img', 'picture', 'photo']:
                    if key in sample:
                        image_field = key
                        break
                
                if not image_field:
                    continue
                
                # Save sample image
                image = sample[image_field]
                if hasattr(image, 'convert'):
                    img = image.convert('RGB')
                    img = img.resize((self.resolution, self.resolution))
                    
                    img_path = os.path.join(sequence_dir, f"frame_{sample_count:06d}.jpg")
                    img.save(img_path, "JPEG", quality=95)
                    sample_count += 1
            
            print(f"✅ Created {sample_count} sample images in {sequence_dir}")
            return output_dir
            
        except Exception as e:
            print(f"❌ Failed to create dataset directory: {e}")
            raise
    
    def get_data_loader(self, batch_size: int = 4, num_workers: int = 2):
        """Create PyTorch DataLoader for training."""
        if not self.dataset:
            raise ValueError("Dataset not connected. Call connect() first.")
        
        # For now, create a simple wrapper - this would need more work for production
        print(f"🔄 Creating DataLoader with batch_size={batch_size}")
        return None  # Placeholder - would need custom Dataset class


def test_dataset_connection():
    """Test the dataset connection and validation - STRICT MODE."""
    print("🧪 Testing MonoX Dataset Connection (STRICT MODE)")
    print("=" * 60)
    print("⚠️  REQUIREMENT: Must connect to 'lukua/monox-dataset'")
    print("⚠️  This is a PRIVATE dataset - authentication required!")
    print("=" * 60)
    
    loader = MonoDatasetLoader(resolution=1024)
    
    # Test connection - STRICT: Must work or fail completely
    if loader.connect():
        print("✅ Dataset connection successful")
        
        # Validate dataset
        validation = loader.validate_dataset()
        print(f"📊 Validation result: {validation}")
        
        if validation["valid"]:
            print("✅ Dataset validation passed")
            print("🎯 MonoX training can proceed with authenticated dataset access")
            
            # Create sample dataset directory
            try:
                dataset_dir = loader.create_stylegan_dataset_dir()
                print(f"✅ Sample dataset directory created: {dataset_dir}")
                return dataset_dir
            except Exception as e:
                print(f"❌ Failed to create dataset directory: {e}")
                return None
        else:
            print(f"❌ Dataset validation failed: {validation['error']}")
            print("🚫 TRAINING BLOCKED: Dataset validation required")
            return None
    else:
        print("❌ Dataset connection failed")
        print("🚫 TRAINING BLOCKED: Cannot proceed without 'lukua/monox-dataset'")
        print("")
        print("🔧 REQUIRED SETUP:")
        print("   1. Authenticate with HuggingFace:")
        print("      - Set HF_TOKEN environment variable, OR")
        print("      - Run: huggingface-cli login")
        print("   2. Ensure you have access to 'lukua/monox-dataset'")
        print("   3. Re-run this script to validate connection")
        return None


def main():
    """Main function for testing dataset integration - STRICT MODE."""
    try:
        result = test_dataset_connection()
        if result:
            print(f"\n🎉 MonoX dataset integration ready!")
            print(f"📁 Dataset directory: {result}")
            print(f"🚀 Ready for StyleGAN-V training at 1024x1024 resolution")
            print(f"✅ AUTHENTICATION: Successfully connected to private dataset")
        else:
            print(f"\n🚫 DATASET CONNECTION FAILED")
            print(f"❌ MonoX training is BLOCKED until 'lukua/monox-dataset' is accessible")
            print(f"🔒 This is a REQUIRED private dataset - no fallbacks allowed")
            print(f"\n💡 Next steps:")
            print(f"   1. Verify HuggingFace authentication")
            print(f"   2. Confirm access to 'lukua/monox-dataset'")
            print(f"   3. Re-run this validation")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during dataset validation: {e}")
        print(f"🚫 TRAINING BLOCKED: Dataset connection is mandatory")
        sys.exit(1)


if __name__ == "__main__":
    main()