"""
Level 1: Baseline Classifier

Standard ResNet18 supervised classification as baseline reference.
"""
from typing import Optional

import torch
import torch.nn as nn
from .backbone import get_backbone


class BaselineClassifier(nn.Module):
    """
    Standard supervised classifier using a pretrained backbone.
    
    Architecture:
        Backbone (ResNet18) -> Projection Head -> Classifier
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        dropout: float = 0.3,
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.backbone, self.feature_dim = get_backbone(backbone_name, pretrained)
        self.embedding_dim = embedding_dim or self.feature_dim
        
        # Projection head (optional)
        if embedding_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.projector = nn.Identity()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.backbone(x)          # (B, feature_dim)
        embeddings = self.projector(features) # (B, embedding_dim)
        logits = self.classifier(embeddings)  # (B, num_classes)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without classification head."""
        features = self.backbone(x)
        return self.projector(features)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning head only."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
