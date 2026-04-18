from __future__ import annotations

import torch
from torch import nn


class LeNetBaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.experiment_name = "model_1_lenet_baseline"
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.output(x)
