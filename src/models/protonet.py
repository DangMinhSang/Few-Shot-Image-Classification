"""
Level 3: Prototypical Network

Core few-shot learning model that classifies by proximity to class prototypes.

Reference: Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_backbone


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Classification.
    
    Algorithm:
    1. Encode support images -> embeddings
    2. Average embeddings per class -> prototypes
    3. Classify query by nearest prototype
    
    Loss: Cross-entropy over softmax(-distances)
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        distance: str = 'euclidean',
    ):
        """
        Args:
            backbone_name: Name of backbone architecture
            pretrained: Use ImageNet pretrained weights
            embedding_dim: Dimension of embedding space
            dropout: Dropout rate
            distance: 'euclidean' or 'cosine'
        """
        super().__init__()
        
        self.backbone, feature_dim = get_backbone(backbone_name, pretrained)
        self.distance_metric = distance
        self.embedding_dim = embedding_dim
        
        # Embedding projection head
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to embedding space."""
        features = self.backbone(x)               # (B, feature_dim)
        embeddings = self.embedding_head(features) # (B, embedding_dim)
        return embeddings
    
    def compute_prototypes(
        self, 
        support_images: torch.Tensor,   # (n_way, k_shot, C, H, W)
        support_labels: torch.Tensor,   # (n_way, k_shot)  
    ) -> torch.Tensor:                  # (n_way, embedding_dim)
        """
        Compute class prototypes by averaging support embeddings.
        
        p_c = (1/|S_c|) * sum_{x in S_c} f(x)
        """
        n_way, k_shot = support_images.shape[:2]
        
        # Flatten for batch encoding
        support_flat = support_images.view(n_way * k_shot, *support_images.shape[2:])  # (n_way*k_shot, C, H, W)
        support_emb = self.encode(support_flat)        # (n_way*k_shot, embed_dim)
        support_emb = support_emb.view(n_way, k_shot, -1)  # (n_way, k_shot, embed_dim)
        
        # Average per class
        prototypes = support_emb.mean(dim=1)           # (n_way, embed_dim)
        return prototypes
    
    def compute_distances(
        self,
        query_emb: torch.Tensor,   # (n_query, embed_dim)
        prototypes: torch.Tensor,  # (n_way, embed_dim)
    ) -> torch.Tensor:             # (n_query, n_way)
        """Compute distances from each query to each prototype."""
        if self.distance_metric == 'euclidean':
            # ||q - p||^2 using broadcasting
            # query_emb: (n_query, 1, embed_dim)
            # prototypes: (1, n_way, embed_dim)
            diff = query_emb.unsqueeze(1) - prototypes.unsqueeze(0)  # (n_query, n_way, embed_dim)
            distances = (diff ** 2).sum(dim=-1)  # (n_query, n_way)
        
        elif self.distance_metric == 'cosine':
            query_norm = F.normalize(query_emb, p=2, dim=-1)       # (n_query, embed_dim)
            proto_norm = F.normalize(prototypes, p=2, dim=-1)       # (n_way, embed_dim)
            # cosine similarity -> distance = 1 - similarity
            similarity = query_norm @ proto_norm.T                  # (n_query, n_way)
            distances = 1 - similarity
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(
        self,
        support_images: torch.Tensor,  # (n_way, k_shot, C, H, W)
        support_labels: torch.Tensor,  # (n_way, k_shot)
        query_images: torch.Tensor,    # (n_query_total, C, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            log_probs: (n_query_total, n_way) log probabilities
            prototypes: (n_way, embedding_dim)
        """
        # Compute prototypes
        prototypes = self.compute_prototypes(support_images, support_labels)  # (n_way, embed_dim)
        
        # Encode queries
        query_emb = self.encode(query_images)  # (n_query_total, embed_dim)
        
        # Compute distances
        distances = self.compute_distances(query_emb, prototypes)  # (n_query_total, n_way)
        
        # Negative distances -> log probabilities
        log_probs = F.log_softmax(-distances, dim=-1)  # (n_query_total, n_way)
        
        return log_probs, prototypes
    
    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Predict class labels for query images."""
        log_probs, _ = self.forward(support_images, support_labels, query_images)
        return log_probs.argmax(dim=-1)
