import torch.nn as nn
from torchvision import models

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone
    def forward_one(self, x):
        return self.backbone(x)
    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)
