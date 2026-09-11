import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        # label: 1 = same class, 0 = different
        d = F.pairwise_distance(out1, out2)
        loss = torch.mean((1 - label) * d.pow(2) +
                         label * F.relu(self.margin - d).pow(2))
        return loss
