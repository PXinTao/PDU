"""Image fusion strategies.

Goal: keep target-domain geometry/structure while injecting source-domain style.

We implement 3 strategies:
1) alpha blend: x_f = w*x_gen + (1-w)*x_raw
2) frequency blend: low-freq from gen + high-freq from raw
3) edge-guided overlay: keep raw around edges, gen elsewhere

All functions accept and return uint8 RGB images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np


@dataclass
class FusionConfig:
    method: Literal['alpha', 'frequency', 'edge_overlay'] = 'frequency'
    # For frequency blending
    sigma: float = 3.0
    # For edge_overlay
    edge_dilation: int = 2
    edge_thr: float = 0.2


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    # ksize from sigma; ensure odd
    ksize = int(max(3, round(sigma * 6)))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def alpha_blend(x_raw: np.ndarray, x_gen: np.ndarray, w: float) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    out = (w * x_gen.astype(np.float32) + (1.0 - w) * x_raw.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def frequency_blend(x_raw: np.ndarray, x_gen: np.ndarray, w: float, sigma: float = 3.0) -> np.ndarray:
    """Blend low-frequency component from x_gen with high-frequency from x_raw.

    The output can be further mixed with x_raw using weight w.
    """
    x_raw_f = x_raw.astype(np.float32)
    x_gen_f = x_gen.astype(np.float32)

    low_raw = _gaussian_blur(x_raw_f, sigma)
    high_raw = x_raw_f - low_raw

    low_gen = _gaussian_blur(x_gen_f, sigma)
    fused = low_gen + high_raw

    # mix back to raw depending on w
    out = w * fused + (1.0 - w) * x_raw_f
    return np.clip(out, 0, 255).astype(np.uint8)


def edge_overlay(x_raw: np.ndarray, x_gen: np.ndarray, edge_map: np.ndarray, w: float, edge_thr: float = 0.2, dilation: int = 2) -> np.ndarray:
    """Use gen globally, but override neighborhoods around edges with raw.

    w controls overall alpha blend as well.
    """
    base = alpha_blend(x_raw, x_gen, w)
    e = edge_map.astype(np.float32)
    if e.max() > 1.0:
        e = e / 255.0
    mask = (e >= edge_thr).astype(np.uint8)
    if dilation and dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k, iterations=dilation)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = base.copy()
    out[mask3 == 1] = x_raw[mask3 == 1]
    return out


def fuse(x_raw: np.ndarray, x_gen: np.ndarray, w: float, cfg: FusionConfig, edge_map: Optional[np.ndarray] = None) -> np.ndarray:
    if cfg.method == 'alpha':
        return alpha_blend(x_raw, x_gen, w)
    if cfg.method == 'frequency':
        return frequency_blend(x_raw, x_gen, w, sigma=cfg.sigma)
    if cfg.method == 'edge_overlay':
        if edge_map is None:
            raise ValueError('edge_map is required for edge_overlay fusion')
        return edge_overlay(x_raw, x_gen, edge_map=edge_map, w=w, edge_thr=cfg.edge_thr, dilation=cfg.edge_dilation)
    raise ValueError(f'Unknown fusion method: {cfg.method}')
