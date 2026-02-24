"""I/O helpers.

All image I/O is implemented with OpenCV to avoid optional dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def is_image_file(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in IMG_EXTS


def read_image(path: str | os.PathLike, to_rgb: bool = True, grayscale: bool = False) -> np.ndarray:
    """Read an image.

    Returns:
        np.ndarray:
            - grayscale=True: HxW, uint8
            - else: HxWx3, uint8
    """
    path = str(path)
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_image(path: str | os.PathLike, img: np.ndarray, from_rgb: bool = True) -> None:
    """Write an image (uint8 expected)."""
    path = str(path)
    ensure_dir(Path(path).parent)
    out = img
    if out.ndim == 3 and out.shape[2] == 3 and from_rgb:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, out)
    if not ok:
        raise IOError(f"Failed to write image to: {path}")


def list_images(root: str | os.PathLike) -> list[str]:
    root = str(root)
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if is_image_file(p):
                paths.append(p)
    paths.sort()
    return paths


def resize_keep_aspect(img: np.ndarray, long_side: int, interp: int = cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) == long_side:
        return img
    if h >= w:
        new_h = long_side
        new_w = int(round(w * (long_side / h)))
    else:
        new_w = long_side
        new_h = int(round(h * (long_side / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def pad_to_square(img: np.ndarray, pad_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad to square.

    Returns:
        padded_img, (top, bottom, left, right)
    """
    h, w = img.shape[:2]
    if h == w:
        return img, (0, 0, 0, 0)
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    if img.ndim == 2:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return padded, (top, bottom, left, right)


def unpad(img: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    if top == bottom == left == right == 0:
        return img
    h, w = img.shape[:2]
    return img[top : h - bottom, left : w - right]
