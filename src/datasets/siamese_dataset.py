"""
Siamese Dataset for pairwise training.

Generates positive pairs (same class) and negative pairs (different class).
"""
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class SiameseDataset(Dataset):
    """
    Dataset for Siamese Network training.
    
    Returns pairs of images with a label:
    - 0: same class (positive pair)
    - 1: different class (negative pair)
    """
    
    def __init__(
        self,
        class_to_samples: Dict[int, List[str]],
        transform: Optional[T.Compose] = None,
        n_pairs: int = 10000,
        positive_ratio: float = 0.5,
        image_size: int = 224,
    ):
        """
        Args:
            class_to_samples: Dict mapping class_idx -> list of image paths
            transform: Optional image transform
            n_pairs: Total number of pairs per epoch
            positive_ratio: Fraction of positive pairs
            image_size: Target image size
        """
        self.class_to_samples = class_to_samples
        self.classes = list(class_to_samples.keys())
        self.n_pairs = n_pairs
        self.positive_ratio = positive_ratio
        self.image_size = image_size
        
        self.transform = transform if transform is not None else self._default_transform()
        
        # Pre-generate pairs for consistency
        self.pairs = self._generate_pairs()
    
    def _default_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _generate_pairs(self) -> List[Tuple[str, str, int]]:
        """Pre-generate (img1_path, img2_path, label) pairs."""
        pairs = []
        n_positive = int(self.n_pairs * self.positive_ratio)
        n_negative = self.n_pairs - n_positive
        
        # Positive pairs (same class)
        for _ in range(n_positive):
            cls = random.choice(self.classes)
            samples = self.class_to_samples[cls]
            if len(samples) >= 2:
                img1, img2 = random.sample(samples, 2)
            else:
                img1 = img2 = samples[0]
            pairs.append((img1, img2, 0))  # 0 = same class
        
        # Negative pairs (different classes)
        for _ in range(n_negative):
            cls1, cls2 = random.sample(self.classes, 2)
            img1 = random.choice(self.class_to_samples[cls1])
            img2 = random.choice(self.class_to_samples[cls2])
            pairs.append((img1, img2, 1))  # 1 = different class
        
        random.shuffle(pairs)
        return pairs
    
    def __len__(self) -> int:
        return self.n_pairs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1_path, img2_path, label = self.pairs[idx]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)
