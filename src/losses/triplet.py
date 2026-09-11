"""
Triplet Loss for metric learning.

Loss = max(d(a, p) - d(a, n) + margin, 0)

Where:
    a: anchor
    p: positive (same class as anchor)
    n: negative (different class from anchor)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss with online hard mining.
    
    Args:
        margin: Margin between positive and negative distances
        mining: 'hard', 'semi-hard', or 'none'
    """
    
    def __init__(self, margin: float = 0.3, mining: str = 'hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, reduction='mean')
    
    def forward(
        self,
        anchor: torch.Tensor,    # (B, embed_dim)
        positive: torch.Tensor,  # (B, embed_dim)
        negative: torch.Tensor,  # (B, embed_dim)
    ) -> torch.Tensor:
        """Compute triplet loss."""
        return self.triplet_loss(anchor, positive, negative)
    
    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,  # (B, embed_dim)
        labels: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        """
        Compute triplet loss with online hard mining.
        
        Args:
            embeddings: Batch of embeddings
            labels: Corresponding class labels
        """
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        if self.mining == 'hard':
            return self._hard_triplet_loss(embeddings, labels, pairwise_dist)
        else:
            return self._batch_hard_triplet_loss(embeddings, labels, pairwise_dist)
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared Euclidean distances."""
        dot = torch.mm(embeddings, embeddings.t())
        sq_norm = dot.diag()
        distances = sq_norm.unsqueeze(0) - 2 * dot + sq_norm.unsqueeze(1)
        return F.relu(distances).sqrt()
    
    def _hard_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        pairwise_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Hard triplet mining: hardest positive + hardest negative."""
        B = embeddings.size(0)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        
        # Hardest positive: max distance within same class
        pos_dist = pairwise_dist * labels_equal.float()
        hardest_pos = pos_dist.max(dim=1).values  # (B,)
        
        # Hardest negative: min distance across different classes
        neg_dist = pairwise_dist + labels_equal.float() * 1e9  # mask positives
        hardest_neg = neg_dist.min(dim=1).values  # (B,)
        
        triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return triplet_loss.mean()
    
    def _batch_hard_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        pairwise_dist: torch.Tensor,
    ) -> torch.Tensor:
        """All valid triplets averaged."""
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        
        # All (i, j) distances where i != j
        anchor_pos = pairwise_dist.unsqueeze(2)  # (B, B, 1)
        anchor_neg = pairwise_dist.unsqueeze(1)  # (B, 1, B)
        
        triplet_loss = F.relu(anchor_pos - anchor_neg + self.margin)  # (B, B, B)
        
        # Valid triplets: (i, j) same class, (i, k) different class, i!=j, i!=k
        valid_mask = labels_equal.unsqueeze(2) & (~labels_equal.unsqueeze(1))
        valid_mask &= torch.eye(B, device=embeddings.device, dtype=torch.bool).unsqueeze(2).logical_not()
        
        return (triplet_loss * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8)
