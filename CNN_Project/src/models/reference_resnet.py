from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ResNet18ReferenceCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.experiment_name = "model_3_resnet18_reference"

        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except TypeError:
            backbone = models.resnet18(pretrained=True)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

        frozen_blocks = [self.stem, self.layer1, self.layer2, self.layer3]
        for block in frozen_blocks:
            for parameter in block.parameters():
                parameter.requires_grad = False
            block.eval()

        for parameter in self.layer4.parameters():
            parameter.requires_grad = True
        for parameter in self.fc.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.stem.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
