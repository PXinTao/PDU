"""Edge related utilities.

- Canny edge extraction
- Boundary-from-mask edge extraction
- Simple morphological ops

All returned edge maps are float32 in [0, 1] unless stated otherwise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def to_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    if img.shape[2] == 1:
        g = img[:, :, 0]
        return np.clip(g, 0, 255).astype(np.uint8)
    # assume RGB
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return g.astype(np.uint8)


def canny_edge(
    img: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    L2gradient: bool = False,
    blur_ksize: int = 3,
) -> np.ndarray:
    """Compute Canny edges.

    Args:
        img: RGB uint8 or grayscale uint8.

    Returns:
        edge: HxW float32 in [0,1].
    """
    g = to_grayscale_uint8(img)
    if blur_ksize and blur_ksize > 1:
        g = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 0)
    e = cv2.Canny(g, threshold1=low_threshold, threshold2=high_threshold, apertureSize=aperture_size, L2gradient=L2gradient)
    return (e.astype(np.float32) / 255.0)


def boundary_from_mask(mask: np.ndarray, dilation: int = 1) -> np.ndarray:
    """Extract a 1-pixel boundary map from a segmentation mask.

    Works for binary or multi-class masks. Non-zero is treated as foreground.

    Args:
        mask: HxW uint8/int, where each pixel is a class index.
        dilation: dilate boundary thickness.

    Returns:
        edge: HxW float32 in [0,1]
    """
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    m = mask.astype(np.int32)
    # Treat all non-zero as FG for boundary
    fg = (m > 0).astype(np.uint8)
    # Morphological gradient: dilate - erode
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(fg, k, iterations=1)
    ero = cv2.erode(fg, k, iterations=1)
    edge = (dil - ero).clip(0, 1)
    if dilation and dilation > 1:
        edge = cv2.dilate(edge, k, iterations=int(dilation))
        edge = (edge > 0).astype(np.uint8)
    return edge.astype(np.float32)


def dilate(edge: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    e = edge
    if e.dtype != np.uint8:
        e = (e > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    d = cv2.dilate(e, k, iterations=iterations)
    return d


def as_controlnet_hint(edge: np.ndarray) -> np.ndarray:
    """Convert an edge map to ControlNet hint format: HxWx3 float32 in [0,1]."""
    if edge.ndim == 3:
        if edge.shape[2] == 3:
            hint = edge.astype(np.float32)
        elif edge.shape[2] == 1:
            hint = np.repeat(edge, 3, axis=2).astype(np.float32)
        else:
            raise ValueError(f"Unsupported edge shape: {edge.shape}")
    else:
        hint = np.repeat(edge[:, :, None], 3, axis=2).astype(np.float32)
    # normalize if input looks like uint8
    if hint.max() > 1.0:
        hint = hint / 255.0
    hint = np.clip(hint, 0.0, 1.0).astype(np.float32)
    return hint


def stack_multi_hint(ch1: np.ndarray, ch2: Optional[np.ndarray] = None, ch3: Optional[np.ndarray] = None) -> np.ndarray:
    """Stack up to 3 single-channel maps into a 3-channel hint.

    Each channel is normalized to [0,1] if necessary.
    """
    def norm(x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float32)
        if y.ndim == 3:
            y = y[:, :, 0]
        if y.max() > 1.0:
            y = y / 255.0
        return np.clip(y, 0.0, 1.0)

    c1 = norm(ch1)
    c2 = norm(ch2) if ch2 is not None else np.zeros_like(c1)
    c3 = norm(ch3) if ch3 is not None else np.zeros_like(c1)
    hint = np.stack([c1, c2, c3], axis=2).astype(np.float32)
    return hint
