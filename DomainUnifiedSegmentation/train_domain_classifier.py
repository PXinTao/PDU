"""Train a domain discriminator (source vs target).

This model provides a *style alignment score* p_source(x).
It only requires domain labels (source vs target), no segmentation labels.

Usage:
python DomainUnifiedSegmentation/train_domain_classifier.py \
  --source_dir /path/to/source/images \
  --target_dir /path/to/target/images \
  --out_ckpt   /path/to/domain_disc.pth \
  --epochs 10 --batch_size 16 --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DomainUnifiedSegmentation.datasets.domain_dataset import DomainFolderDataset
from DomainUnifiedSegmentation.models.domain.simple_cnn import DomainDiscriminator
from DomainUnifiedSegmentation.utils.io import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source_dir', type=str, required=True)
    p.add_argument('--target_dir', type=str, required=True)
    p.add_argument('--out_ckpt', type=str, required=True)

    p.add_argument('--resize', type=int, default=512)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--in_channels', type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ds = DomainFolderDataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        resize=(args.resize, args.resize),
        to_grayscale=(args.in_channels == 1),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    net = DomainDiscriminator(in_channels=args.in_channels).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        n = 0
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            logit = net(x)
            loss = bce(logit, y)

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
