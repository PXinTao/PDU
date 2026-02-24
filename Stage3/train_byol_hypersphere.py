# Stage3/train_byol_hypersphere.py
# BYOL-on-hypersphere for domain unified scoring (raw <-> generated pairs)
# - supports multi-gpu with accelerate
# - progress bar + logs only on main process
# - stable checkpointing
# - pairs raw image from controlnet json with generated image from unified_cache meta.csv by index

from __future__ import annotations

import os
import csv
import json
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


# -------------------------
# utils
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_rgb(path: str, image_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if image_size is not None and image_size > 0:
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    return img

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def sphere_loss(x: torch.Tensor, y: torch.Tensor, m: float = 4.0) -> torch.Tensor:
    """
    x, y: (B, D) already L2-normalized or not.
    hypersphere loss: minimize angular distance (scaled)
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    cos = (x * y).sum(dim=-1).clamp(-1.0, 1.0)  # (B,)
    theta = torch.acos(cos)                      # (B,)
    loss = (m * theta) ** 2
    return loss.mean()


# -------------------------
# dataset
# -------------------------

def load_controlnet_sources(json_path: Path) -> List[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sources = []
    for it in data:
        if "source" not in it:
            raise ValueError(f"Missing 'source' in {json_path}")
        sources.append(it["source"])
    return sources

def load_unified_gen_paths(meta_csv: Path) -> List[str]:
    """
    meta.csv must contain column gen_path.
    We do NOT rely on filename; we align by row order with controlnet json order.
    """
    rows = []
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if len(rows) == 0:
        raise FileNotFoundError(f"meta.csv empty: {meta_csv}")
    if "gen_path" not in rows[0]:
        raise ValueError(f"meta.csv missing 'gen_path' column: {meta_csv}")
    return [r["gen_path"] for r in rows]

class PairRawGenDataset(Dataset):
    """
    Each item:
      raw_path from controlnet_{split}.json (source)
      gen_path from unified_cache/{split}/meta.csv  (gen_path)
    Aligned by index.
    """
    def __init__(
        self,
        controlnet_json: str,
        meta_csv: str,
        image_size: int,
        pair_gen: bool,
        max_items: int = -1,
    ):
        self.controlnet_json = Path(controlnet_json)
        self.meta_csv = Path(meta_csv)
        self.image_size = image_size
        self.pair_gen = pair_gen

        self.raw_paths = load_controlnet_sources(self.controlnet_json)

        if self.pair_gen:
            self.gen_paths = load_unified_gen_paths(self.meta_csv)
            if len(self.gen_paths) != len(self.raw_paths):
                raise ValueError(
                    f"Length mismatch!\n"
                    f"  json: {len(self.raw_paths)} items\n"
                    f"  meta: {len(self.gen_paths)} items\n"
                    f"json={self.controlnet_json}\nmeta={self.meta_csv}\n"
                    f"Fix: ensure inference meta.csv built from the SAME json order."
                )
        else:
            self.gen_paths = None

        if max_items > 0:
            self.raw_paths = self.raw_paths[:max_items]
            if self.gen_paths is not None:
                self.gen_paths = self.gen_paths[:max_items]

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx: int):
        raw_path = self.raw_paths[idx]
        raw_img = read_rgb(raw_path, self.image_size)

        if not self.pair_gen:
            return raw_img, raw_img, idx  # (img1,img2) placeholder

        gen_path = self.gen_paths[idx]
        gen_img = read_rgb(gen_path, self.image_size)
        return raw_img, gen_img, idx


# -------------------------
# model: BYOL hypersphere
# -------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class BYOLHypersphere(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int = 2048,
        proj_dim: int = 256,
        proj_hidden: int = 4096,
        pred_hidden: int = 4096,
        ema: float = 0.99,
        sphere_m: float = 4.0,
    ):
        super().__init__()
        self.backbone_online = backbone
        self.projector_online = MLP(feat_dim, proj_hidden, proj_dim)
        self.predictor_online = MLP(proj_dim, pred_hidden, proj_dim)

        # target
        self.backbone_target = self._copy_no_grad(self.backbone_online)
        self.projector_target = self._copy_no_grad(self.projector_online)

        self.ema = ema
        self.sphere_m = sphere_m

    @staticmethod
    def _copy_no_grad(net: nn.Module) -> nn.Module:
        import copy
        tgt = copy.deepcopy(net)
        for p in tgt.parameters():
            p.requires_grad_(False)
        return tgt

    @torch.no_grad()
    def update_target(self):
        for (p_o, p_t) in zip(self.backbone_online.parameters(), self.backbone_target.parameters()):
            p_t.data.mul_(self.ema).add_(p_o.data, alpha=(1 - self.ema))
        for (p_o, p_t) in zip(self.projector_online.parameters(), self.projector_target.parameters()):
            p_t.data.mul_(self.ema).add_(p_o.data, alpha=(1 - self.ema))

    def encode_online(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone_online(x)             # (B,feat_dim)
        proj = self.projector_online(feat)         # (B,proj_dim)
        pred = self.predictor_online(proj)         # (B,proj_dim)
        return pred

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone_target(x)
        proj = self.projector_target(feat)
        return proj

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        BYOL symmetric loss on hypersphere:
          online(x1) match target(x2)
          online(x2) match target(x1)
        """
        p1 = self.encode_online(x1)
        p2 = self.encode_online(x2)

        with torch.no_grad():
            z1 = self.encode_target(x1)
            z2 = self.encode_target(x2)

        loss1 = sphere_loss(p1, z2.detach(), m=self.sphere_m)
        loss2 = sphere_loss(p2, z1.detach(), m=self.sphere_m)
        return 0.5 * (loss1 + loss2)


# -------------------------
# augment
# -------------------------

def build_augment(image_size: int) -> nn.Module:
    # Ultrasound: 保守增强（不要太重的颜色抖动）
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        # 简单归一化到 ImageNet 也行（resnet50 pretrained）
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# -------------------------
# backbone helper
# -------------------------

def build_backbone_resnet50_pretrained(accelerator: Accelerator) -> nn.Module:
    """
    Avoid multi-rank downloading:
      - rank0 triggers weights download
      - barrier
      - others load from cache
    """
    if accelerator.is_main_process:
        _ = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    accelerator.wait_for_everyone()

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # remove fc
    model.fc = nn.Identity()
    return model


# -------------------------
# training
# -------------------------

@dataclass
class SplitConfig:
    name: str
    controlnet_json: Path
    meta_csv: Path

def parse_args():
    p = argparse.ArgumentParser()

    # roots (defaults are your real dirs)
    p.add_argument("--unified_cache_root", type=str,
                   default="/home/data/pxt/USSeg/EchoNet_Merged/unified_cache_r256")
    p.add_argument("--controlnet_json_root", type=str,
                   default="/home/data/pxt/USSeg/EchoNet_Merged")

    p.add_argument("--splits", type=str, default="train",
                   help="comma splits: train,val,test. Usually train only.")
    p.add_argument("--pair_gen", action="store_true",
                   help="use (raw, gen) as positive pair. If off: BYOL on raw itself.")

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)

    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--sphere_m", type=float, default=4.0)

    p.add_argument("--save_dir", type=str,
                   default="/home/data/pxt/USSeg/stage3_byol_hypersphere")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)

    p.add_argument("--max_items", type=int, default=-1,
                   help="debug: limit dataset size per split")

    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])

    return p.parse_args()


def main():
    args = parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else "no",
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        print(f"[{now_str()}] Stage3 BYOL-Hypersphere start")
        print("Args:", vars(args))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unified_root = Path(args.unified_cache_root)
    json_root = Path(args.controlnet_json_root)

    split_cfgs: List[SplitConfig] = []
    for s in splits:
        controlnet_json = json_root / s / f"controlnet_{s}.json"
        meta_csv = unified_root / s / "meta.csv"
        if not controlnet_json.exists():
            raise FileNotFoundError(f"Missing controlnet json: {controlnet_json}")
        if args.pair_gen and (not meta_csv.exists()):
            raise FileNotFoundError(f"Missing meta.csv: {meta_csv}")
        split_cfgs.append(SplitConfig(s, controlnet_json, meta_csv))

    # dataset concat (simple: only train is typical)
    datasets = []
    for cfg in split_cfgs:
        ds = PairRawGenDataset(
            controlnet_json=str(cfg.controlnet_json),
            meta_csv=str(cfg.meta_csv),
            image_size=args.image_size,
            pair_gen=args.pair_gen,
            max_items=args.max_items,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        train_ds = datasets[0]
    else:
        # concat multiple splits if you really want
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset(datasets)

    aug = build_augment(args.image_size)

    def collate_fn(batch):
        # batch: list of (PIL1,PIL2,idx)
        x1 = torch.stack([aug(b[0]) for b in batch], dim=0)
        x2 = torch.stack([aug(b[1]) for b in batch], dim=0)
        idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return x1, x2, idx

    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # model
    backbone = build_backbone_resnet50_pretrained(accelerator)
    model = BYOLHypersphere(
        backbone=backbone,
        feat_dim=2048,
        proj_dim=args.proj_dim,
        ema=args.ema,
        sphere_m=args.sphere_m,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # prepare
    model, opt, dl = accelerator.prepare(model, opt, dl)

    # save dir + logs
    save_dir = Path(args.save_dir)
    if accelerator.is_main_process:
        ensure_dir(save_dir)
        (save_dir / "ckpt").mkdir(exist_ok=True, parents=True)
        (save_dir / "logs").mkdir(exist_ok=True, parents=True)
        log_path = save_dir / "logs" / "train_log.csv"
        if not log_path.exists():
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["time", "epoch", "step", "loss"])

    accelerator.wait_for_everyone()
    log_path = save_dir / "logs" / "train_log.csv"

    # tqdm only on main
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        it = dl
        if accelerator.is_main_process and tqdm is not None:
            it = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for (x1, x2, _) in it:
            with accelerator.autocast():
                loss = model(x1, x2)

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)

            # EMA update (sync-safe)
            model.module.update_target() if hasattr(model, "module") else model.update_target()

            running += loss.detach().float().item()
            n += 1
            global_step += 1

            if accelerator.is_main_process and (global_step % args.log_every == 0):
                avg = running / max(1, n)
                msg = f"[{now_str()}] epoch={epoch} step={global_step} loss={avg:.6f}"
                print(msg)
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([now_str(), epoch, global_step, avg])

        # epoch end
        if accelerator.is_main_process:
            avg = running / max(1, n)
            print(f"[{now_str()}] Epoch {epoch} DONE | loss={avg:.6f}")

        # checkpoint
        if (epoch % args.save_every == 0) and accelerator.is_main_process:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "state_dict": accelerator.get_state_dict(model),
                "opt": opt.state_dict(),
                "args": vars(args),
            }
            ckpt_path = save_dir / "ckpt" / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[{now_str()}] Saved: {ckpt_path}")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"[{now_str()}] Training finished. Save dir: {save_dir}")


if __name__ == "__main__":
    main()
