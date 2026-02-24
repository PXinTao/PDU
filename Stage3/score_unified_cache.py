# DomainUnifiedSegmentation/stage3_hypersphere/score_unified_cache.py
from __future__ import annotations

import argparse
import csv
import os
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

import torchvision.models as models


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def imread_rgb01(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def load_encoder_from_byol_ckpt(ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()

    class Wrap(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.backbone = bb
        def forward(self, x): return self.backbone(x)

    wrapper = Wrap(backbone)
    wrapper.load_state_dict(ckpt["online_encoder"], strict=False)
    return backbone


@torch.no_grad()
def score_folder(
    encoder: torch.nn.Module,
    stats: dict,
    gen_dir: Path,
    out_csv: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
):
    mu = stats["mu"].to(device)                 # (D,)
    inv_cov = stats["inv_cov"].to(device)       # (D,D)
    tau = float(stats["tau"])
    beta = float(stats["beta"])

    paths = list_images(str(gen_dir))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images in {gen_dir}")

    encoder.eval().to(device)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"score {gen_dir.name}"):
        batch_paths = paths[i:i+batch_size]
        xs = torch.stack([imread_rgb01(p, image_size) for p in batch_paths], dim=0).to(device)  # (B,3,H,W)
        z = encoder(xs)                         # (B,D)
        z = F.normalize(z, dim=-1)              # hypersphere

        xc = z - mu[None, :]
        d2 = torch.einsum("bd,dd,bd->b", xc, inv_cov, xc).clamp_min(0)
        d = torch.sqrt(d2)

        # style score: sigmoid around tau (smaller distance => higher score)
        # score = sigmoid(-beta*(d - tau)) = 1/(1+exp(beta*(d-tau)))
        score = 1.0 / (1.0 + torch.exp(beta * (d - tau)))

        for p, di, si in zip(batch_paths, d.tolist(), score.tolist()):
            rows.append({
                "name": Path(p).stem,
                "path": str(Path(p).resolve()),
                "mahalanobis_d": float(di),
                "s_style": float(si),
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "path", "mahalanobis_d", "s_style"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] wrote -> {out_csv} ({len(rows)} rows)")


def main():
    DEFAULT_CACHE = "/home/data/pxt/USSeg/EchoNet_Merged/unified_cache_r256"
    DEFAULT_STATS = "/home/data/pxt/USSeg/stage3_byol_hypersphere/mahalanobis_stats.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--byol_ckpt", required=True, type=str)
    ap.add_argument("--stats", default=DEFAULT_STATS, type=str)
    ap.add_argument("--cache_root", default=DEFAULT_CACHE, type=str)
    ap.add_argument("--splits", default="train,val,test", type=str)
    ap.add_argument("--image_size", default=256, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--device", default="cuda", type=str)
    args = ap.parse_args()

    device = torch.device(args.device)
    stats = torch.load(args.stats, map_location="cpu")
    encoder = load_encoder_from_byol_ckpt(args.byol_ckpt)

    cache_root = Path(args.cache_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for s in splits:
        gen_dir = cache_root / s / "gen"
        out_csv = cache_root / s / f"style_scores_{s}.csv"
        score_folder(
            encoder=encoder,
            stats=stats,
            gen_dir=gen_dir,
            out_csv=out_csv,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
