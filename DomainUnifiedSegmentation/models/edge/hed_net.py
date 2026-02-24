"""
HED-like edge detector for ultrasound.

Adapted from ControlNet annotator HED, with:
- deep supervision (side outputs + fused)
- forward returns FULL-RES side logits + fused logits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, layer_number: int):
        super().__init__()
        convs = []
        convs.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1))
        for _ in range(1, layer_number):
            convs.append(nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1))
        self.convs = nn.ModuleList(convs)
        self.proj = nn.Conv2d(output_channel, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, down_sampling: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        if down_sampling:
            h = F.max_pool2d(h, kernel_size=2, stride=2)
        for conv in self.convs:
            h = F.relu(conv(h), inplace=True)
        p = self.proj(h)  # (B,1,h,w) logits
        return h, p


class HEDNet(nn.Module):
    def __init__(self):
        super().__init__()
        # learnable mean norm (as in ControlNet HED annotator)
        self.norm = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.block1 = DoubleConvBlock(3, 64, 2)
        self.block2 = DoubleConvBlock(64, 128, 2)
        self.block3 = DoubleConvBlock(128, 256, 3)
        self.block4 = DoubleConvBlock(256, 512, 3)
        self.block5 = DoubleConvBlock(512, 512, 3)
        self.conv = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, padding=0)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            outputs: [p1_up, p2_up, p3_up, p4_up, p5_up, fused]
            all are logits of shape (B,1,H,W) at input resolution
        """
        H, W = x.shape[-2], x.shape[-1]

        h = x - self.norm
        h, p1 = self.block1(h)                    # (B,1,H,W)
        h, p2 = self.block2(h, down_sampling=True)  # (B,1,H/2,W/2)
        h, p3 = self.block3(h, down_sampling=True)  # (B,1,H/4,W/4)
        h, p4 = self.block4(h, down_sampling=True)  # (B,1,H/8,W/8)
        h, p5 = self.block5(h, down_sampling=True)  # (B,1,H/16,W/16)

        projections = [p1, p2, p3, p4, p5]

        # upsample all side logits to full-res
        ups = [F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False) for p in projections]
        cat = torch.cat([ups[0], ups[1], ups[2], ups[3], ups[4]], dim=1)

        # fused full-res logits
        fused = self.conv(cat)

        return ups[0].sigmoid(),ups[1].sigmoid(),ups[2].sigmoid(),ups[3].sigmoid(),ups[4].sigmoid(),fused.sigmoid()

    @staticmethod
    def fuse_logits(projections: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        """Fuse side outputs into a single logit map at out_size."""
        H, W = out_size
        ups = [F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False) for p in projections]
        stacked = torch.stack(ups, dim=0)  # (S,B,1,H,W)
        mean_logits = stacked.mean(dim=0)  # (B,1,H,W)
        return mean_logits

    @staticmethod
    def fuse_prob(projections: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        """Fuse side outputs into a single probability edge map at out_size."""
        return torch.sigmoid(HEDNet.fuse_logits(projections, out_size=out_size))


@dataclass
class HEDLossConfig:
    side_weight: float = 0.5
    fused_weight: float = 1.0


def hed_deep_supervision_loss(
    projections: List[torch.Tensor],
    gt_edge: torch.Tensor,
    cfg: HEDLossConfig,
) -> torch.Tensor:
    """
    BCE loss for side outputs + fused output.

    Args:
        projections: list of logits. Accepts either:
            - original multi-scale: [p1..p5] at different sizes
            - full-res outputs from forward: [p1_up..p5_up,fused]
        gt_edge: (B,1,H,W) float {0,1}
    """
    bce = nn.BCEWithLogitsLoss()

    H, W = gt_edge.shape[-2:]

    # If caller passes [p1..p5,fused] full-res, split accordingly.
    if len(projections) >= 6:
        side = projections[:-1]
        fused_logits = projections[-1]
        loss_fused = bce(fused_logits, gt_edge)
        loss_side = 0.0
        for p in side:
            # p is already full-res
            loss_side = loss_side + bce(p, gt_edge)
        loss_side = loss_side / max(1, len(side))
        return cfg.side_weight * loss_side + cfg.fused_weight * loss_fused

    # Otherwise treat as multi-scale [p1..p5]
    mean_logits = HEDNet.fuse_logits(projections, out_size=(H, W))
    loss_fused = bce(mean_logits, gt_edge)

    loss_side = 0.0
    for p in projections:
        gt_s = F.interpolate(gt_edge, size=p.shape[-2:], mode="nearest")
        loss_side = loss_side + bce(p, gt_s)
    loss_side = loss_side / max(1, len(projections))

    return cfg.side_weight * loss_side + cfg.fused_weight * loss_fused
