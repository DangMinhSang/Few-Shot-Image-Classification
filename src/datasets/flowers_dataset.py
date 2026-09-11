"""
Oxford Flowers 102 Dataset Loader
"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FlowersDataset(Dataset):
    """
    Oxford Flowers 102 dataset.
    
    Supports both standard classification and few-shot episode sampling.
    """
    
    FLOWER_NAMES = [
        'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells',
        'sweet pea', 'english marigold', 'tiger lily', 'moon orchid',
        'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
        "colt's foot", 'king protea', 'spear thistle', 'yellow iris',
        'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
        'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
        'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
        'stemless gentian', 'artichoke', 'sweet william', 'carnation',
        'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
        'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
        'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
        'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
        'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
        'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
        'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
        'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
        'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea',
        'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower',
        'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis',
        'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia',
        'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm',
        'pink quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
        'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily'
    ]
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[T.Compose] = None,
        image_size: int = 224,
    ):
        """
        Args:
            root: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Optional torchvision transforms
            image_size: Target image size
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform if transform is not None else self._default_transform(split)
        
        # Load dataset index
        self.samples: List[Tuple[str, int]] = []  # (image_path, label)
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_samples: Dict[int, List[str]] = {}
        
        self._load_dataset()
    
    def _default_transform(self, split: str) -> T.Compose:
        """Default transforms for each split."""
        if split == 'train':
            return T.Compose([
                T.Resize((self.image_size + 32, self.image_size + 32)),
                T.RandomCrop(self.image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def _load_dataset(self):
        """Load dataset from directory structure or torchvision."""
        split_dir = self.root / self.split
        
        if split_dir.exists():
            # Load from local directory structure
            self._load_from_directory(split_dir)
        else:
            # Try loading via torchvision
            self._load_from_torchvision()
    
    def _load_from_directory(self, split_dir: Path):
        """Load from folder structure: split/class_name/image.jpg"""
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            self.class_to_samples[idx] = []
            
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), idx))
                    self.class_to_samples[idx].append(str(img_path))
    
    def _load_from_torchvision(self):
        """Load Oxford Flowers 102 via torchvision."""
        import torchvision.datasets as dsets
        
        split_map = {'train': 'train', 'val': 'val', 'test': 'test'}
        tv_split = split_map[self.split]
        
        # Download if needed
        tv_dataset = dsets.Flowers102(
            root=str(self.root),
            split=tv_split,
            download=True
        )
        
        # Build index
        for img_path, label in zip(tv_dataset._image_files, tv_dataset._labels):
            img_path_str = str(img_path)
            if label not in self.class_to_samples:
                self.class_to_samples[label] = []
                class_name = self.FLOWER_NAMES[label] if label < len(self.FLOWER_NAMES) else f'class_{label}'
                self.class_to_idx[class_name] = label
                self.idx_to_class[label] = class_name
            self.samples.append((img_path_str, label))
            self.class_to_samples[label].append(img_path_str)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label
    
    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
    
    def get_class_samples(self, class_idx: int) -> List[str]:
        """Get all image paths for a given class."""
        return self.class_to_samples.get(class_idx, [])
    
    def get_all_classes(self) -> List[int]:
        """Get list of all class indices."""
        return list(self.class_to_samples.keys())
