"""
Data loading utilities for MonoX training
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder structure"""
    
    def __init__(self, path, resolution=1024, use_labels=False, transform=None):
        self.path = path
        self.resolution = resolution
        self.use_labels = use_labels
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.image_files.extend(Path(path).rglob(f'*{ext}'))
            self.image_files.extend(Path(path).rglob(f'*{ext.upper()}'))
        
        if not self.image_files:
            raise ValueError(f"No image files found in {path}")
        
        print(f"Found {len(self.image_files)} images")
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            if self.use_labels:
                # For now, return dummy label
                return image, 0
            else:
                return image, None
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.resolution, self.resolution), None if not self.use_labels else 0