"""Train a segmentation model (UNet) on the SOURCE domain.

Usage:
python DomainUnifiedSegmentation/train_seg.py \
  --images_dir /path/to/source/images \
  --masks_dir  /path/to/source/masks \
  --out_ckpt   /path/to/seg_unet.pth \
  --num_classes 2 --epochs 100 --batch_size 8

Then run inference-time domain unification on a target domain using:
python DomainUnifiedSegmentation/infer_seg_unify.py ...

Notes:
- This is a minimal baseline UNet trainer. Replace with your own backbone if needed.
- Masks are expected to be class-index maps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from DomainUnifiedSegmentation.datasets.image_mask_dataset import ImageMaskFolderDataset
from DomainUnifiedSegmentation.models.seg.unet import UNet
from DomainUnifiedSegmentation.utils.augment import compose_default
from DomainUnifiedSegmentation.utils.io import ensure_dir
from DomainUnifiedSegmentation.utils.losses import SegLossConfig, segmentation_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--masks_dir', type=str, required=True)
    p.add_argument('--out_ckpt', type=str, required=True)

    p.add_argument('--resize', type=int, default=512)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')

    p.add_argument('--in_channels', type=int, default=1)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--base_channels', type=int, default=32)

    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--dice_weight', type=float, default=1.0)

    # optional init
    p.add_argument('--init_weights', type=str, default='')

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ds = ImageMaskFolderDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        resize=(args.resize, args.resize),
        to_grayscale=(args.in_channels == 1),
        transform=lambda img, mask: compose_default(img, mask),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    net = UNet(in_channels=args.in_channels, num_classes=args.num_classes, base_channels=args.base_channels).to(device)

    if args.init_weights:
        state = torch.load(args.init_weights, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ['module.', 'net.', 'model.']:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        net.load_state_dict(cleaned, strict=False)
        print(f"Loaded init weights from: {args.init_weights}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    cfg = SegLossConfig(ce_weight=args.ce_weight, dice_weight=args.dice_weight)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        n = 0
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = net(x)
            loss = segmentation_loss(logits, y, num_classes=args.num_classes, cfg=cfg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1

        avg = running / max(1, n)
        print(f"Epoch {epoch}/{args.epochs} - loss={avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            out_path = Path(args.out_ckpt)
            ensure_dir(out_path.parent)
            torch.save({'state_dict': net.state_dict(), 'epoch': epoch, 'loss': best_loss}, str(out_path))
            print(f"Saved best checkpoint to: {out_path} (loss={best_loss:.6f})")


if __name__ == '__main__':
    main()
