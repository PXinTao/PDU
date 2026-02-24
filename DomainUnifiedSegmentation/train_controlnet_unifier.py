"""Stage1: Train a structure-guided domain unifier (ControlNet diffusion) on SOURCE domain.

This script reuses the ControlNet code shipped in:
  Stage1&2.Diffusion Model/

Unlike the original `tutorial_dataset.py`, this script is:
- configurable (no hard-coded paths)
- compatible with edge hints produced by your learned HED edge extractor

Workflow:
1) Prepare JSON:
   python DomainUnifiedSegmentation/prepare_controlnet_json.py \
     --images_dir /path/to/source/images \
     --out_json   /path/to/train_controlnet.json \
     --prompt "Ultrasound"

2) Optionally precompute learned edge maps and set --hints_dir when preparing JSON.

3) Train:
   python DomainUnifiedSegmentation/train_controlnet_unifier.py \
     --stage1_root "/path/to/ADAptation-main/Stage1&2.Diffusion Model" \
     --train_json  /path/to/train_controlnet.json \
     --sd_ckpt     /path/to/v1-5-pruned.ckpt \
     --controlnet_ckpt /path/to/control_sd15_canny.pth \
     --out_dir     /path/to/exp_controlnet \
     --max_epochs  50

The output checkpoint can be used with `infer_seg_unify.py` via --finetuned_ckpt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets.controlnet_dataset import ControlNetDataset
from .utils.io import ensure_dir



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_root', type=str, required=True)
    p.add_argument('--train_json', type=str, required=True)

    p.add_argument('--cldm_yaml', type=str, default='./models/cldm_v15.yaml')
    p.add_argument('--sd_ckpt', type=str, required=True)
    p.add_argument('--controlnet_ckpt', type=str, required=True)

    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_epochs', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--acc_grad', type=int, default=1)
    p.add_argument('--logger_freq', type=int, default=50)

    p.add_argument('--resolution', type=int, default=512)

    p.add_argument('--out_dir', type=str, required=True)

    p.add_argument('--sd_locked', action='store_true', help='lock stable diffusion weights')
    p.add_argument('--only_mid_control', action='store_true')

    return p.parse_args()


def main():
    args = parse_args()
    stage_root = Path(args.stage1_root).resolve()
    if not stage_root.exists():
        raise FileNotFoundError(f'Stage1 root not found: {stage_root}')
    sys.path.insert(0, str(stage_root))

    from Stage1and2.cldm.model import create_model, load_state_dict
    from Stage1and2.cldm.logger import ImageLogger

    # model
    model = create_model(str(stage_root / args.cldm_yaml)).cpu()

    # load SD base
    sd_states = load_state_dict(str(stage_root / args.sd_ckpt), location='cpu')
    model.load_state_dict(sd_states, strict=False)

    # load ControlNet backbone
    cn_states = load_state_dict(str(stage_root / args.controlnet_ckpt), location='cpu')
    model.load_state_dict(cn_states, strict=False)

    model.learning_rate = args.lr
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    # dataset
    ds = ControlNetDataset(json_path=args.train_json, resolution=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # logs / ckpts
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    logger = ImageLogger(batch_frequency=args.logger_freq)

    trainer = pl.Trainer(
        precision=32,
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[logger],
        accumulate_grad_batches=args.acc_grad,
        default_root_dir=str(out_dir),
    )

    trainer.fit(model, dl)

import torch


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    main()
