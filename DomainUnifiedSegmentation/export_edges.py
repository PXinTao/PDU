"""Export edge maps for a folder of images.

This is recommended if you want to use learned HED edges during ControlNet training,
because computing GPU edges inside DataLoader workers is unstable.

Examples:
# Using ControlNet HED annotator (pretrained)
python DomainUnifiedSegmentation/export_edges.py \
  --stage1_root "/path/to/Stage1&2.Diffusion Model" \
  --images_dir  /path/to/source/images \
  --out_dir     /path/to/source/edges_hed \
  --edge_method hed_controlnet

# Using your fine-tuned HEDNet
python DomainUnifiedSegmentation/export_edges.py \
  --images_dir /path/to/target/images \
  --out_dir    /path/to/target/edges_hed \
  --edge_method hed_finetuned \
  --hed_weights /path/to/hed_ultrasound.pth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from DomainUnifiedSegmentation.models.edge.extractor import EdgeExtractor, EdgeExtractorConfig
from DomainUnifiedSegmentation.utils.io import ensure_dir, list_images, read_image, write_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--edge_method', type=str, default='hed_controlnet', choices=['canny','hed_controlnet','hed_finetuned','multi'])
    p.add_argument('--stage1_root', type=str, default='')
    p.add_argument('--hed_weights', type=str, default='')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--resize', type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    cfg = EdgeExtractorConfig(
        method=args.edge_method,
        stage1_root=args.stage1_root if args.stage1_root else None,
        hed_weights=args.hed_weights if args.hed_weights else None,
        device=args.device,
    )
    extractor = EdgeExtractor(cfg)

    paths = list_images(args.images_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f'No images under: {args.images_dir}')

    for p in paths:
        img = read_image(p, to_rgb=True, grayscale=False)
        import cv2
        img_r = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
        e = extractor(img_r)

        rel = os.path.relpath(p, args.images_dir)
        out_path = Path(args.out_dir) / Path(rel).with_suffix('.png')
        ensure_dir(out_path.parent)

        if e.ndim == 3:
            # multi: save as 3-channel visualization (0-255)
            vis = np.clip(e * 255.0, 0, 255).astype(np.uint8)
            write_image(str(out_path), vis)
        else:
            vis = np.clip(e * 255.0, 0, 255).astype(np.uint8)
            write_image(str(out_path), vis, from_rgb=False)

        print(f"{p} -> {out_path}")


if __name__ == '__main__':
    main()
