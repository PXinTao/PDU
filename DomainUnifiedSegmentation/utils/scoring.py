"""Scoring utilities.

We compute two families of scores:

1) Structural preservation score (S_struct)
   - computed as F1 between edge maps of raw and generated images

2) Style alignment score (S_style)
   - default: probability of being source domain from a domain discriminator
   - fallback: histogram similarity if no discriminator is provided

The final fusion weight w is a monotonic function of these scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import cv2
import torch

from .metrics import edge_f1, clamp


@dataclass
class ScoreConfig:
    # Edge score
    edge_thr: float = 0.2
    edge_dilation: int = 1

    # Style score via domain discriminator
    style_weight: float = 0.6
    struct_weight: float = 0.4

    # Fusion weight shaping
    gamma_style: float = 1.0
    gamma_struct: float = 1.0

    # optional routing thresholds
    tau_low: float = 0.35
    tau_high: float = 0.75


def histogram_style_score(img: np.ndarray, ref_hist: np.ndarray) -> float:
    """Compute a simple histogram similarity score in [0,1].

    Works best for grayscale ultrasound.
    """
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        g = img
    hist = cv2.calcHist([g], [0], None, [256], [0,256]).astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    # Chi-square distance -> similarity
    chi = 0.5 * np.sum(((hist - ref_hist) ** 2) / (hist + ref_hist + 1e-8))
    sim = float(np.exp(-chi * 10.0))
    return clamp(sim, 0.0, 1.0)


def compute_scores(
    raw_img: np.ndarray,
    gen_img: np.ndarray,
    raw_edge: np.ndarray,
    gen_edge: np.ndarray,
    cfg: ScoreConfig,
    domain_discriminator: Optional[torch.nn.Module] = None,
    ref_hist: Optional[np.ndarray] = None,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Compute S_struct, S_style, S_total and recommended fusion weight w."""
    s_struct = edge_f1(raw_edge, gen_edge, thr=cfg.edge_thr, dilation=cfg.edge_dilation)

    # Style score
    s_style: float
    if domain_discriminator is not None:
        domain_discriminator.eval()
        # convert to tensor
        x = gen_img.astype(np.float32) / 255.0
        if x.ndim == 3:
            x = x.mean(axis=2, keepdims=True)  # 1-channel
        x_t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = domain_discriminator(x_t)
            s_style = float(torch.sigmoid(logit).squeeze().item())
    else:
        if ref_hist is None:
            # neutral
            s_style = 0.5
        else:
            s_style = histogram_style_score(gen_img, ref_hist)

    s_total = clamp(cfg.style_weight * s_style + cfg.struct_weight * s_struct, 0.0, 1.0)

    # weight for generated image in fusion
    w = (s_style ** cfg.gamma_style) * (s_struct ** cfg.gamma_struct)
    w = clamp(w, 0.0, 1.0)

    return {
        's_struct': float(s_struct),
        's_style': float(s_style),
        's_total': float(s_total),
        'w': float(w),
    }
