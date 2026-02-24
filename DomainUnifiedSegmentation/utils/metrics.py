"""Metrics used for scoring and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import cv2


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Dice coefficient for binary masks.

    Args:
        pred: HxW bool/uint8
        gt: HxW bool/uint8
    """
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    inter = float((p * g).sum())
    return (2.0 * inter + eps) / (float(p.sum() + g.sum()) + eps)


def edge_f1(a: np.ndarray, b: np.ndarray, thr: float = 0.2, dilation: int = 1) -> float:
    """F1 score between two edge probability maps.

    Args:
        a, b: edge maps in [0,1] or [0,255]
        thr: threshold to binarize
        dilation: dilate edges to tolerate small misalignments
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    if b.max() > 1.0:
        b = b / 255.0

    A = (a >= thr).astype(np.uint8)
    B = (b >= thr).astype(np.uint8)

    if dilation and dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        A_d = cv2.dilate(A, k, iterations=dilation)
        B_d = cv2.dilate(B, k, iterations=dilation)
    else:
        A_d, B_d = A, B

    tp = int(((A_d == 1) & (B == 1)).sum())
    fp = int(((A_d == 0) & (B == 1)).sum())
    fn = int(((A == 1) & (B_d == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))
