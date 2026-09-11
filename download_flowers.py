import os
from pathlib import Path
from torchvision.datasets import Flowers102

# Destination directory (we have write permission at project root)
root_dir = Path.cwd() / "flowers_data"
os.makedirs(root_dir, exist_ok=True)

# Download all splits
for split in ["train", "val", "test"]:
    Flowers102(root=str(root_dir), split=split, download=True)
print("Oxford Flowers 102 downloaded to", root_dir)
