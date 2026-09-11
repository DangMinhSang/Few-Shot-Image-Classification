import torch.nn as nn
from torchvision import models

class BaselineResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone
    def forward(self, x):
        return self.backbone(x)
