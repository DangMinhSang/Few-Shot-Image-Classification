"""
Contrastive Loss for Siamese Network training.

Loss = (1-y) * 0.5 * D^2 + y * 0.5 * max(margin - D, 0)^2

Where:
    y = 0: same class (positive pair)
    y = 1: different class (negative pair)
    D: Euclidean distance between embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (Hadsell et al., 2006).
    
    Args:
        margin: Margin for negative pairs (default: 1.0)
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        emb1: torch.Tensor,   # (B, embed_dim)
        emb2: torch.Tensor,   # (B, embed_dim)  
        labels: torch.Tensor, # (B,) - 0: same class, 1: different class
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            emb1, emb2: L2-normalized embeddings
            labels: 0 for same class, 1 for different class
        """
        distance = F.pairwise_distance(emb1, emb2, p=2)  # (B,)
        
        # Positive pairs (same class): minimize distance
        pos_loss = (1 - labels) * 0.5 * distance.pow(2)
        
        # Negative pairs (different class): push apart up to margin
        neg_loss = labels * 0.5 * F.relu(self.margin - distance).pow(2)
        
        loss = (pos_loss + neg_loss).mean()
        return loss
    
    def get_metrics(self, emb1, emb2, labels):
        """Compute additional metrics for logging."""
        with torch.no_grad():
            distance = F.pairwise_distance(emb1, emb2, p=2)
            pos_dist = distance[labels == 0].mean().item() if (labels == 0).any() else 0
            neg_dist = distance[labels == 1].mean().item() if (labels == 1).any() else 0
        return {'pos_dist': pos_dist, 'neg_dist': neg_dist}
