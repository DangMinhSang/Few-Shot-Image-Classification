"""Evaluation metrics for few-shot learning."""
from typing import List, Tuple
import numpy as np
import torch


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute mean and 95% confidence interval.
    
    Returns:
        (mean, ci): mean and half-width of CI
    """
    values = np.array(values)
    n = len(values)
    mean = values.mean()
    std = values.std()
    
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    ci = z * std / np.sqrt(n)
    
    return mean, ci


def compute_few_shot_accuracy(
    model,
    sampler,
    n_episodes: int = 600,
    device: torch.device = torch.device('cpu'),
) -> Tuple[float, float]:
    """
    Compute few-shot accuracy over multiple episodes.
    
    Returns:
        (mean_acc, ci95): mean accuracy and 95% CI
    """
    model.eval()
    accs = []
    
    with torch.no_grad():
        for i in range(n_episodes):
            episode = sampler[i % len(sampler)]
            
            support_images = episode.support_images.to(device)
            support_labels = episode.support_labels.to(device)
            query_images = episode.query_images.to(device)
            query_labels = episode.query_labels.to(device)
            
            predictions = model.predict(support_images, support_labels, query_images)
            acc = (predictions == query_labels).float().mean().item()
            accs.append(acc)
    
    mean_acc, ci95 = compute_confidence_interval(accs)
    return mean_acc, ci95


def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute top-k accuracy."""
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(labels.view(-1, 1).expand_as(topk))
    return correct.any(dim=1).float().mean().item()
