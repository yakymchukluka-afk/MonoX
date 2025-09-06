"""
MonoX Training Module
"""
from .data import ImageFolderDataset
from .train import main

__all__ = ['ImageFolderDataset', 'main']