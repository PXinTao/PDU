"""Lightweight numpy-based augmentations.

Designed to avoid torchvision dependency.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def random_flip(img: np.ndarray, mask: np.ndarray, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        img = np.flip(img, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if random.random() < p:
        img = np.flip(img, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    return img, mask


def random_rotate90(img: np.ndarray, mask: np.ndarray, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        k = random.randint(0, 3)
        img = np.rot90(img, k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
    return img, mask


def compose_default(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img, mask = random_flip(img, mask, p=0.5)
    img, mask = random_rotate90(img, mask, p=0.5)
    return img, mask
