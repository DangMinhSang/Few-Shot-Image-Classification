import os
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from . import config

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transform(train: bool = True):
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    if train:
        base = [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4)] + base
    return transforms.Compose(base)

def get_image_folder(split: str = "train"):
    path = config.DATA_ROOT / split
    return ImageFolder(root=str(path), transform=get_transform(train=(split=="train")))

# Episodic sampler (used for Siamese & ProtoNet)
def create_episode(dataset: ImageFolder, n_way: int = 5, k_shot: int = 5, q_query: int = 15):
    class_idx = {c: [] for c in range(len(dataset.classes))}
    for i, (_, label) in enumerate(dataset.samples):
        class_idx[label].append(i)
    chosen = np.random.choice(list(class_idx.keys()), n_way, replace=False)
    support_idxs, query_idxs = [], []
    support_labels, query_labels = [], []
    for cls_id, cls in enumerate(chosen):
        idxs = np.random.choice(class_idx[cls], k_shot + q_query, replace=False)
        support = idxs[:k_shot]
        query = idxs[k_shot:]
        support_idxs.extend(support)
        query_idxs.extend(query)
        support_labels.extend([cls_id]*k_shot)
        query_labels.extend([cls_id]*q_query)
    support_set = Subset(dataset, support_idxs)
    query_set = Subset(dataset, query_idxs)
    support_loader = DataLoader(support_set, batch_size=len(support_set), shuffle=False)
    query_loader = DataLoader(query_set, batch_size=len(query_set), shuffle=False)
    return support_loader, query_loader, support_labels, query_labels
