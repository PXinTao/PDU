# DomainUnifiedSegmentation/stage3_hypersphere/dataset_pairs.py
from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def imread_rgb01(path: str, size: int) -> torch.Tensor:
    """
    Returns: (3, size, size) float in [0,1]
    """
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
    return t


class RawGenPairDataset(Dataset):
    """
    BYOL dataset that can optionally align (raw, gen) by name.

    raw_dir: /.../EchoNet_Merged/train/imgs
    gen_dir: /.../unified_cache_r256/train/gen

    If pair_gen=True:
      - only keep items where raw stem exists
      - return dict(raw=..., gen=..., name=stem)
    Else:
      - just return raw images (gen ignored)
    """
    def __init__(
        self,
        raw_dir: str,
        image_size: int = 256,
        gen_dir: Optional[str] = None,
        pair_gen: bool = True,
        max_items: int = -1,
    ):
        self.raw_dir = str(raw_dir)
        self.gen_dir = str(gen_dir) if gen_dir else None
        self.image_size = int(image_size)
        self.pair_gen = bool(pair_gen)

        raw_paths = list_images(self.raw_dir)
        if len(raw_paths) == 0:
            raise FileNotFoundError(f"No images in raw_dir: {self.raw_dir}")

        if self.gen_dir and self.pair_gen:
            gen_paths = list_images(self.gen_dir)
            gen_map = {Path(p).stem: p for p in gen_paths}

            items = []
            for rp in raw_paths:
                stem = Path(rp).stem
                gp = gen_map.get(stem, None)
                if gp is None:
                    continue
                items.append((rp, gp, stem))
            if len(items) == 0:
                raise RuntimeError(
                    f"pair_gen=True but found 0 aligned items.\n"
                    f"raw_dir={self.raw_dir}\n"
                    f"gen_dir={self.gen_dir}\n"
                    "Make sure gen filenames match raw stems (same stem)."
                )
        else:
            items = [(rp, None, Path(rp).stem) for rp in raw_paths]

        if max_items > 0:
            items = items[:max_items]

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rp, gp, stem = self.items[idx]
        raw = imread_rgb01(rp, self.image_size)
        out = {"raw": raw, "name": stem}
        if gp is not None:
            gen = imread_rgb01(gp, self.image_size)
            out["gen"] = gen
        return out
