"""Dataset for ControlNet training (Stage1: train unifier on SOURCE domain).

This matches the data interface expected by the ControlNet training code in
`Stage1&2.Diffusion Model`:
  return dict(jpg=image, txt=prompt, hint=hint)

- jpg: HxWx3 float32 in [-1,1]
- hint: HxWx3 float32 in [0,1]
- txt: str

We support two modes:
1) JSON list mode (recommended):
   Each item = {"source": "/abs/path/to/img.png", "prompt": "...", "hint": "<optional edge image>"}
2) Folder mode: read all images from images_dir and compute hint online (CPU Canny only).

Notes:
- If you want HED hints, we strongly recommend precomputing them to disk and referencing via JSON, because GPU inference inside DataLoader workers is error-prone.

Extra:
- If return_paths=True, we ALSO return:
    path: original source path (string)
    name: Path(source).stem (string)
  This is useful at inference time to save generated images with exactly the same filename
  as the original image, enabling strict 1:1 alignment for contrastive learning / scoring.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..utils.io import list_images, read_image
from ..utils.edge_ops import canny_edge, as_controlnet_hint


@dataclass
class ControlNetItem:
    source: str
    prompt: str
    hint: Optional[str] = None


class ControlNetDataset(Dataset):
    def __init__(
        self,
        json_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        hints_dir: Optional[str] = None,
        resolution: int = 512,
        canny_low: int = 50,
        canny_high: int = 150,
        drop_prompt_prob: float = 0.2,
        drop_hint_prob: float = 0.2,
        return_paths: bool = False,   # NEW
        strict_hints: bool = False,   # optional: if True, require hint exists for every item
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.drop_prompt_prob = float(drop_prompt_prob)
        self.drop_hint_prob = float(drop_hint_prob)
        self.return_paths = bool(return_paths)
        self.strict_hints = bool(strict_hints)

        items: List[ControlNetItem] = []
        if json_path:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for it in data:
                items.append(
                    ControlNetItem(
                        source=it['source'],
                        prompt=it.get('prompt', ''),
                        hint=it.get('hint', None),
                    )
                )
        elif images_dir:
            img_paths = list_images(images_dir)
            for p in img_paths:
                hint_path = None
                if hints_dir:
                    rel = os.path.relpath(p, images_dir)
                    cand = os.path.join(hints_dir, rel)
                    if os.path.exists(cand):
                        hint_path = cand
                items.append(ControlNetItem(source=p, prompt='', hint=hint_path))
        else:
            raise ValueError('Either json_path or images_dir must be provided')

        if len(items) == 0:
            raise FileNotFoundError('No items found for ControlNetDataset')

        # If strict_hints, verify all have hints
        if self.strict_hints:
            missing = [it.source for it in items if (it.hint is None or (not os.path.exists(it.hint)))]
            if len(missing) > 0:
                raise FileNotFoundError(
                    f"strict_hints=True but {len(missing)} items are missing hint files. "
                    f"Example missing source: {missing[0]}"
                )

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = read_image(it.source, to_rgb=True, grayscale=False)
        img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # jpg: [-1,1]
        jpg = (img.astype(np.float32) / 127.5) - 1.0

        # hint: either load from path or compute canny
        if it.hint is not None and os.path.exists(it.hint):
            h = read_image(it.hint, to_rgb=True, grayscale=False)
            h = cv2.resize(h, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            hint = h.astype(np.float32) / 255.0
        else:
            e = canny_edge(img, low_threshold=self.canny_low, high_threshold=self.canny_high, blur_ksize=3)
            hint = as_controlnet_hint(e)

        prompt = it.prompt

        # drop modes (as used in original ControlNet training)
        if self.drop_prompt_prob > 0 and random.random() < self.drop_prompt_prob:
            prompt = ""
        if self.drop_hint_prob > 0 and random.random() < self.drop_hint_prob:
            hint = np.zeros_like(hint)

        out = dict(jpg=jpg, txt=prompt, hint=hint)

        # NEW: return original path/name for strict 1:1 alignment (useful for inference)
        if self.return_paths:
            out["path"] = it.source
            out["name"] = Path(it.source).stem

        return out
