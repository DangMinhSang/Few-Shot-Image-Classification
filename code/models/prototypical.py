import torch.nn as nn
from torchvision import models

class ProtoNet(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone
    def embed(self, x):
        return self.backbone(x)
