"""
Level 2: Siamese Network

Learns a similarity metric between pairs of images.
Supports both Contrastive Loss and Triplet Loss.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_backbone


class SiameseNetwork(nn.Module):
    """
    Siamese Network: two identical branches sharing weights.
    
    Architecture:
        Image A ---> Encoder ---\n                                  +---> Distance ---> Same/Different
        Image B ---> Encoder ---/
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.backbone, feature_dim = get_backbone(backbone_name, pretrained)
        
        # Embedding projection
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized embedding for a single image batch."""
        features = self.backbone(x)              # (B, feature_dim)
        embedding = self.embedding_head(features) # (B, embedding_dim)
        return F.normalize(embedding, p=2, dim=1) # L2 normalize
    
    def forward(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of images.
        
        Returns:
            emb1: embedding of img1 (B, embedding_dim)
            emb2: embedding of img2 (B, embedding_dim)
            distance: L2 distance between embeddings (B,)
        """
        emb1 = self.encode(img1)
        emb2 = self.encode(img2)
        distance = F.pairwise_distance(emb1, emb2, p=2)
        return emb1, emb2, distance
    
    def predict_same(self, img1: torch.Tensor, img2: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict if two images belong to same class (1) or not (0)."""
        _, _, distance = self.forward(img1, img2)
        return (distance < threshold).float()
