"""
Few-Shot Episode Sampler for Prototypical Network training.

Generates N-way K-shot episodes with Q query samples per class.
"""
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


@dataclass
class FewShotEpisode:
    """
    A single few-shot episode.
    
    Attributes:
        support_images: (n_way, k_shot, C, H, W)
        support_labels: (n_way, k_shot)  - local labels 0..n_way-1
        query_images: (n_way * n_query, C, H, W)
        query_labels: (n_way * n_query,) - local labels 0..n_way-1
        class_names: list of actual class names in this episode
    """
    support_images: torch.Tensor
    support_labels: torch.Tensor
    query_images: torch.Tensor
    query_labels: torch.Tensor
    class_names: List[str]


class EpisodeSampler(Dataset):
    """
    Generates few-shot episodes for meta-learning.
    
    Each episode contains:
    - N classes (N-way)
    - K support images per class (K-shot)
    - Q query images per class
    """
    
    def __init__(
        self,
        class_to_samples: Dict[int, List[str]],
        idx_to_class: Dict[int, str],
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        n_episodes: int = 100,
        transform: Optional[T.Compose] = None,
        augment_support: bool = False,
        image_size: int = 224,
    ):
        """
        Args:
            class_to_samples: Dict mapping class_idx -> list of image paths
            idx_to_class: Dict mapping class_idx -> class name
            n_way: Number of classes per episode
            k_shot: Number of support images per class
            n_query: Number of query images per class
            n_episodes: Number of episodes per epoch
            transform: Image transform pipeline
            augment_support: Whether to augment support images
            image_size: Target image size
        """
        self.class_to_samples = class_to_samples
        self.idx_to_class = idx_to_class
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.image_size = image_size
        
        # Filter classes with enough samples
        self.valid_classes = [
            cls for cls, samples in class_to_samples.items()
            if len(samples) >= k_shot + n_query
        ]
        
        if len(self.valid_classes) < n_way:
            # Relax constraint - allow reusing samples
            self.valid_classes = [
                cls for cls, samples in class_to_samples.items()
                if len(samples) >= max(k_shot, n_query)
            ]
            print(f"Warning: Using {len(self.valid_classes)} classes with relaxed sampling")
        
        assert len(self.valid_classes) >= n_way, \
            f"Not enough classes: have {len(self.valid_classes)}, need {n_way}"
        
        self.transform = transform if transform is not None else self._default_transform()
        self.support_transform = self._augment_transform() if augment_support else self.transform
    
    def _default_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _augment_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.image_size + 32, self.image_size + 32)),
            T.RandomCrop(self.image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomRotation(30),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_image(self, path: str, transform: T.Compose) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return transform(img)
    
    def __len__(self) -> int:
        return self.n_episodes
    
    def __getitem__(self, idx: int) -> FewShotEpisode:
        """Generate a random few-shot episode."""
        # Sample N classes
        episode_classes = random.sample(self.valid_classes, self.n_way)
        class_names = [self.idx_to_class[c] for c in episode_classes]
        
        support_images_list = []
        query_images_list = []
        support_labels_list = []
        query_labels_list = []
        
        for local_label, global_class in enumerate(episode_classes):
            samples = self.class_to_samples[global_class]
            
            # Sample support + query
            n_needed = self.k_shot + self.n_query
            if len(samples) >= n_needed:
                chosen = random.sample(samples, n_needed)
            else:
                # Sample with replacement if not enough images
                chosen = random.choices(samples, k=n_needed)
            
            support_paths = chosen[:self.k_shot]
            query_paths = chosen[self.k_shot:]
            
            # Load support images
            support_imgs = torch.stack([
                self._load_image(p, self.support_transform) for p in support_paths
            ])  # (k_shot, C, H, W)
            
            # Load query images
            query_imgs = torch.stack([
                self._load_image(p, self.transform) for p in query_paths
            ])  # (n_query, C, H, W)
            
            support_images_list.append(support_imgs)
            query_images_list.append(query_imgs)
            support_labels_list.append(torch.full((self.k_shot,), local_label, dtype=torch.long))
            query_labels_list.append(torch.full((self.n_query,), local_label, dtype=torch.long))
        
        support_images = torch.stack(support_images_list)   # (n_way, k_shot, C, H, W)
        support_labels = torch.stack(support_labels_list)   # (n_way, k_shot)
        query_images = torch.cat(query_images_list, dim=0)  # (n_way*n_query, C, H, W)
        query_labels = torch.cat(query_labels_list, dim=0)  # (n_way*n_query,)
        
        return FewShotEpisode(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
            query_labels=query_labels,
            class_names=class_names,
        )


def collate_episodes(batch: List[FewShotEpisode]) -> FewShotEpisode:
    """Collate a batch of episodes (usually batch_size=1 for meta-learning)."""
    # For simplicity, we typically use batch_size=1 for episodes
    return batch[0]
