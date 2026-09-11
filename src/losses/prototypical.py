"""
Prototypical Loss for Prototypical Network.

Classification via nearest prototype in embedding space,
using NLL loss over softmax(-distances).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalLoss(nn.Module):
    """
    Cross-entropy loss over prototype distances.
    
    The model outputs log_probs via softmax(-distances),
    this loss just applies NLL with the query labels.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        log_probs: torch.Tensor,  # (n_query_total, n_way)
        query_labels: torch.Tensor,  # (n_query_total,)
    ) -> torch.Tensor:
        """NLL loss over log probabilities."""
        return F.nll_loss(log_probs, query_labels)
    
    def accuracy(self, log_probs: torch.Tensor, query_labels: torch.Tensor) -> float:
        """Compute episode accuracy."""
        predictions = log_probs.argmax(dim=-1)
        correct = (predictions == query_labels).float().sum()
        total = query_labels.size(0)
        return (correct / total).item()
