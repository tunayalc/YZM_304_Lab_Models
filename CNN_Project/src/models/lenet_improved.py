from __future__ import annotations

import torch
from torch import nn


class LeNetImprovedCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.30) -> None:
        super().__init__()
        self.experiment_name = "model_2_lenet_improved"
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(self.relu(self.bn3(self.fc1(x))))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.output(x)
