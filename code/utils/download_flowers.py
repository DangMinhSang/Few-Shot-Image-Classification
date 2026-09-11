import os
from pathlib import Path
import torch
from torchvision.datasets import Flowers102
from torchvision import transforms

# Define where to store raw data
data_root = Path(__file__).resolve().parents[2] / "data" / "raw"
os.makedirs(data_root, exist_ok=True)

# Flowers102 will download and extract automatically
# We use split="train" just to trigger download; the dataset provides train/val/test splits
 Flowers102(root=str(data_root), split="train", download=True, transform=transforms.ToTensor())
 Flowers102(root=str(data_root), split="val", download=True, transform=transforms.ToTensor())
 Flowers102(root=str(data_root), split="test", download=True, transform=transforms.ToTensor())

print("Download completed. Raw data stored at:", data_root)
