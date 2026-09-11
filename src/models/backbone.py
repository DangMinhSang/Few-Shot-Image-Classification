"""
Backbone feature extractors.

Supports: ResNet18, ResNet50, EfficientNet-B0, ViT-B/16
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str = 'resnet18', pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Get backbone encoder.
    
    Returns:
        (backbone, feature_dim): backbone module and output feature dimension
    """
    name = name.lower()
    
    if name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Remove classification head
        return backbone, feature_dim
    
    elif name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    
    elif name in ['efficientnet_b0', 'efficientnet-b0']:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim
    
    elif name in ['efficientnet_b2', 'efficientnet-b2']:
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim
    
    elif name in ['mobilenet_v3', 'mobilenetv3']:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim
    
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose from: resnet18, resnet50, efficientnet_b0")


class ConvBlock(nn.Module):
    """Standard conv block: Conv2d -> BN -> ReLU -> MaxPool."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv4(nn.Module):
    """
    Simple 4-layer Conv backbone, classic for few-shot learning.
    Input: (B, 3, 84, 84) -> Output: (B, 64*5*5) for 84x84 input
    """
    
    def __init__(self, hidden_size: int = 64, out_size: int = 1600):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            nn.Flatten(),
        )
        self.out_size = out_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
