"""Generic image/mask dataset.

Expected layout (recommended):

root/
  images/
    xxx.png
  masks/
    xxx.png

Masks can be uint8 class index maps. For binary tasks, 0=background, 1=foreground.

This dataset does not assume any specific dataset (EchoNet, CAMUS, etc.).
You can prepare frames from videos externally and place them into this structure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.io import list_images, read_image


@dataclass
class ImageMaskSample:
    img_path: str
    mask_path: str


class ImageMaskFolderDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        resize: Optional[Tuple[int, int]] = (512, 512),
        to_grayscale: bool = True,
        mask_values_to_ignore: Optional[List[int]] = None,
        transform: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        super().__init__()
        self.images_dir = str(images_dir)
        self.masks_dir = str(masks_dir)
        self.resize = resize
        self.to_grayscale = to_grayscale
        self.transform = transform
        self.mask_values_to_ignore = set(mask_values_to_ignore or [])

        img_paths = list_images(self.images_dir)
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No images found under: {self.images_dir}")

        samples: list[ImageMaskSample] = []
        for ip in img_paths:
            rel = os.path.relpath(ip, self.images_dir)
            mp = os.path.join(self.masks_dir, rel)
            if not os.path.exists(mp):
                # also try same stem with .png
                stem = Path(rel).with_suffix('.png')
                mp2 = os.path.join(self.masks_dir, str(stem))
                if os.path.exists(mp2):
                    mp = mp2
                else:
                    continue
            samples.append(ImageMaskSample(img_path=ip, mask_path=mp))

        if len(samples) == 0:
            raise FileNotFoundError(
                f"No matched (image,mask) pairs. images_dir={self.images_dir}, masks_dir={self.masks_dir}"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = read_image(s.img_path, to_rgb=True, grayscale=False)
        mask = read_image(s.mask_path, to_rgb=False, grayscale=True)

        if self.resize is not None:
            import cv2
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)

        if self.to_grayscale:
            # convert RGB->1 channel by averaging (keep as 1 channel tensor)
            img_f = img.astype(np.float32) / 255.0
            img_f = img_f.mean(axis=2, keepdims=True)
        else:
            img_f = img.astype(np.float32) / 255.0

        mask_i = mask.astype(np.int64)

        # optionally ignore some labels by setting them to background
        if self.mask_values_to_ignore:
            for v in self.mask_values_to_ignore:
                mask_i[mask_i == v] = 0

        if self.transform is not None:
            img_f, mask_i = self.transform(img_f, mask_i)

        # to torch
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_i).long()

        return img_t, mask_t, s.img_path
