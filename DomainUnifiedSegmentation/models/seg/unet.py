"""A simple UNet implementation for 2D medical image segmentation.

- Pure PyTorch (no torchvision)
- Supports binary or multi-class segmentation

This is intentionally minimal so it can be dropped into most research codebases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth

        # Encoder
        enc_blocks = []
        pools = []
        ch = in_channels
        for d in range(depth):
            out_ch = base_channels * (2**d)
            enc_blocks.append(ConvBlock(ch, out_ch))
            pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = out_ch
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.pools = nn.ModuleList(pools)

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        # Decoder
        upconvs = []
        dec_blocks = []
        for d in reversed(range(depth)):
            out_ch = base_channels * (2**d)
            upconvs.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2))
            dec_blocks.append(ConvBlock(ch, out_ch))
            ch = out_ch
        self.upconvs = nn.ModuleList(upconvs)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)
        h = self.dropout(h)

        for up, dec in zip(self.upconvs, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            # pad if needed due to odd sizes
            if h.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - h.shape[-2]
                dw = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        return self.head(h)
