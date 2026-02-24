"""Edge extractor abstraction.

Supported methods:
- canny: OpenCV Canny
- hed_controlnet: use ControlNet annotator HEDdetector (pretrained, good for ControlNet)
- hed_finetuned: our HEDNet with your fine-tuned weights
- multi: stack up to 3 channels (e.g., canny_fine, canny_coarse, hed)

Output:
- edge maps are returned as float32 in [0,1] (HxW)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

from ...utils.edge_ops import canny_edge


EdgeMethod = Literal['canny', 'hed_controlnet', 'hed_finetuned', 'multi']


@dataclass
class EdgeExtractorConfig:
    method: EdgeMethod = 'hed_controlnet'

    # Canny params
    canny_low: int = 50
    canny_high: int = 150
    canny_blur: int = 3

    # Multi-hint setup
    multi_canny_low_1: int = 50
    multi_canny_high_1: int = 150
    multi_canny_low_2: int = 30
    multi_canny_high_2: int = 100

    # HED fine-tuned weights
    hed_weights: Optional[str] = None

    # Stage1 folder (required for hed_controlnet)
    stage1_root: Optional[str] = None

    device: str = 'cuda'


class EdgeExtractor:
    def __init__(self, cfg: EdgeExtractorConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self._hed_detector = None
        self._hed_net = None

        if cfg.method == 'hed_controlnet':
            if cfg.stage1_root is None:
                raise ValueError('stage1_root is required for hed_controlnet')
            import sys
            stage_root = Path(cfg.stage1_root).resolve()
            if not stage_root.exists():
                raise FileNotFoundError(f'Stage1 root not found: {stage_root}')
            sys.path.insert(0, str(stage_root))
            from annotator.hed import HEDdetector
            self._hed_detector = HEDdetector()

        if cfg.method == 'hed_finetuned':
            if cfg.hed_weights is None:
                raise ValueError('hed_weights is required for hed_finetuned')
            from .hed_net import HEDNet
            net = HEDNet().to(self.device)
            state = torch.load(cfg.hed_weights, map_location=self.device)
            # allow both full state dict and nested
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            # strip possible prefixes
            cleaned = {}
            for k, v in state.items():
                nk = k
                for prefix in ['module.', 'net.', 'model.']:
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                cleaned[nk] = v
            net.load_state_dict(cleaned, strict=False)
            net.eval()
            self._hed_net = net

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        cfg = self.cfg

        if cfg.method == 'canny':
            e = canny_edge(img_rgb, low_threshold=cfg.canny_low, high_threshold=cfg.canny_high, blur_ksize=cfg.canny_blur)
            return e

        if cfg.method == 'hed_controlnet':
            # ControlNet HED expects RGB uint8
            edge_u8 = self._hed_detector(img_rgb)
            return (edge_u8.astype(np.float32) / 255.0)

        if cfg.method == 'hed_finetuned':
            import cv2
            # Use RGB float tensor
            x = img_rgb.astype(np.float32) / 255.0
            if x.ndim == 2:
                x = np.repeat(x[:, :, None], 3, axis=2)
            if x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)
            x_t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                proj = self._hed_net(x_t)
                prob = torch.sigmoid(self._hed_net.fuse_logits(proj, out_size=(x_t.shape[-2], x_t.shape[-1])))
            e = prob.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
            return np.clip(e, 0.0, 1.0)

        if cfg.method == 'multi':
            # Return a 3-channel hint map (HxWx3) for ControlNet.
            import cv2
            # channel 1: fine canny
            e1 = canny_edge(img_rgb, low_threshold=cfg.multi_canny_low_1, high_threshold=cfg.multi_canny_high_1, blur_ksize=cfg.canny_blur)
            # channel 2: coarse canny
            e2 = canny_edge(img_rgb, low_threshold=cfg.multi_canny_low_2, high_threshold=cfg.multi_canny_high_2, blur_ksize=cfg.canny_blur)
            # channel 3: hed
            if self._hed_detector is None:
                if cfg.stage1_root is None:
                    raise ValueError('stage1_root required for multi (hed_controlnet part)')
                import sys
                stage_root = Path(cfg.stage1_root).resolve()
                sys.path.insert(0, str(stage_root))
                from annotator.hed import HEDdetector
                self._hed_detector = HEDdetector()
            e3 = (self._hed_detector(img_rgb).astype(np.float32) / 255.0)
            hint = np.stack([e1, e2, e3], axis=2).astype(np.float32)
            return hint

        raise ValueError(f'Unknown edge method: {cfg.method}')
