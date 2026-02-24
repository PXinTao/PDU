# DomainUnifiedSegmentation/stage3_hypersphere/models_byol.py
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sphere_pair_loss(x: torch.Tensor, y: torch.Tensor, m: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Pairwise hypersphere angle loss (1-to-1), NOT x @ y.T
    x,y: (B,D)
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    cos = (x * y).sum(dim=-1).clamp(-1 + eps, 1 - eps)  # (B,)
    theta = torch.acos(cos)                               # (B,)
    theta_scaled = m * theta
    return (theta_scaled ** 2).mean()


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


class EMA:
    def __init__(self, beta: float):
        self.beta = beta

    @torch.no_grad()
    def update(self, ma_model: nn.Module, cur_model: nn.Module):
        for ma_p, p in zip(ma_model.parameters(), cur_model.parameters()):
            ma_p.data = ma_p.data * self.beta + p.data * (1 - self.beta)


class EncoderWrapper(nn.Module):
    """
    Wrap a torchvision backbone to output a feature vector (before classifier).
    For resnet50, we take avgpool output.
    """
    def __init__(self, backbone: nn.Module, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet forward up to avgpool:
        # backbone should be resnet with fc replaced by Identity.
        z = self.backbone(x)  # (B,feat_dim)
        return z


@dataclass
class BYOLConfig:
    proj_dim: int = 256
    proj_hidden: int = 2048
    pred_hidden: int = 2048
    ema: float = 0.99
    sphere_m: float = 4.0


class BYOLHypersphere(nn.Module):
    """
    BYOL with hypersphere pair loss.
    """
    def __init__(self, encoder: nn.Module, feat_dim: int, cfg: BYOLConfig):
        super().__init__()
        self.cfg = cfg

        self.online_encoder = EncoderWrapper(encoder, feat_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.projector = MLP(feat_dim, cfg.proj_hidden, cfg.proj_dim)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.predictor = MLP(cfg.proj_dim, cfg.pred_hidden, cfg.proj_dim)

        self.ema = EMA(cfg.ema)

    @torch.no_grad()
    def update_target(self):
        self.ema.update(self.target_encoder, self.online_encoder)
        self.ema.update(self.target_projector, self.projector)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        v1,v2: (B,3,H,W) in [0,1]
        """
        # online
        o1 = self.online_encoder(v1)
        o2 = self.online_encoder(v2)
        z1 = F.normalize(self.projector(o1), dim=-1)
        z2 = F.normalize(self.projector(o2), dim=-1)
        p1 = F.normalize(self.predictor(z1), dim=-1)
        p2 = F.normalize(self.predictor(z2), dim=-1)

        # target
        with torch.no_grad():
            t1 = self.target_encoder(v1)
            t2 = self.target_encoder(v2)
            tz1 = F.normalize(self.target_projector(t1), dim=-1)
            tz2 = F.normalize(self.target_projector(t2), dim=-1)

        # hypersphere pair losses
        loss1 = sphere_pair_loss(p1, tz2, m=self.cfg.sphere_m)
        loss2 = sphere_pair_loss(p2, tz1, m=self.cfg.sphere_m)
        return loss1 + loss2
