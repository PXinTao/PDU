"""Dataset for domain classifier.

We only need domain labels (source vs target), no segmentation labels.

Expected input:
- source_dir: folder with images
- target_dir: folder with images

The dataset returns (image_tensor, domain_label, path), where domain_label:
    1 = source, 0 = target
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import list_images, read_image


class DomainFolderDataset(Dataset):
    def __init__(self, source_dir: str, target_dir: str, resize: Optional[Tuple[int, int]] = (512, 512), to_grayscale: bool = True):
        super().__init__()
        self.source_paths = list_images(source_dir)
        self.target_paths = list_images(target_dir)
        if len(self.source_paths) == 0:
            raise FileNotFoundError(f'No images under source_dir: {source_dir}')
        if len(self.target_paths) == 0:
            raise FileNotFoundError(f'No images under target_dir: {target_dir}')
        self.resize = resize
        self.to_grayscale = to_grayscale

        # Keep roughly balanced by sampling from both in __getitem__
        self.n = max(len(self.source_paths), len(self.target_paths))

    def __len__(self) -> int:
        return self.n * 2

    def __getitem__(self, idx: int):
        # alternate source/target
        is_source = (idx % 2 == 0)
        if is_source:
            p = random.choice(self.source_paths)
            y = 1
        else:
            p = random.choice(self.target_paths)
            y = 0

        img = read_image(p, to_rgb=True, grayscale=False)
        if self.resize is not None:
            import cv2
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        if self.to_grayscale:
            x = (img.astype(np.float32) / 255.0).mean(axis=2, keepdims=True)
        else:
            x = (img.astype(np.float32) / 255.0)

        x_t = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y_t = torch.tensor(y, dtype=torch.float32)
        return x_t, y_t, p
