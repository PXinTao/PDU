"""Loss functions for segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss.

    Args:
        logits: (B,C,H,W)
        target: (B,H,W) long class indices
    """
    # one-hot
    b, c, h, w = logits.shape
    probs = F.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()
    dims = (0,2,3)
    inter = (probs * target_1h).sum(dims)
    denom = (probs + target_1h).sum(dims)
    dice = (2*inter + eps) / (denom + eps)
    # average over classes (excluding background optional? keep all for now)
    return 1.0 - dice.mean()


@dataclass
class SegLossConfig:
    ce_weight: float = 1.0
    dice_weight: float = 1.0


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, cfg: SegLossConfig) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    dice = soft_dice_loss(logits, target, num_classes=num_classes)
    return cfg.ce_weight * ce + cfg.dice_weight * dice
