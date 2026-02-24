"""A lightweight CNN domain discriminator.

This discriminator is used to estimate how *source-like* an image is.
We train it with domain labels only:
  - source images => label 1
  - target images => label 0

At inference, p_source(x) is used as a "style alignment" score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        c = base_channels
        self.fea = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(2*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(4*c, 8*c, 3, padding=1), nn.BatchNorm2d(8*c), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8*c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        h = self.fea(x)
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        logit = self.fc(h)
        return logit

    @staticmethod
    def prob_source(logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit)
