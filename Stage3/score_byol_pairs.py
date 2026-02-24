"""
Stage3: Score BYOL embedding similarity between RAW images and UNIFIED (generated) images.

Your ckpt keys are like:
  backbone_online.conv1.weight
  backbone_online.layer1.0.conv1.weight
so we load them into a vanilla torchvision resnet50 (conv1/layer1...).

Pairing logic (per split):
- Read ControlNet JSON: <EchoNet_Merged>/<split>/controlnet_<split>.json
  each item contains {"source": ".../imgs/xxx.png", "prompt": "...", "hint": "..."}
- Find generated image under:
    <unified_cache_root>/<split>/gen/
  We try in order:
    A) same stem: "<stem>.png"
    B) legacy index name: f"{split}_{idx:06d}_{sample_id:02d}.png"  (default sample_id=0)
If none found -> skip.

Outputs:
- <out_dir>/scores_<split>.csv  (cosine, angle, l2, optional mahal-diag)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("PIL is required: pip install pillow") from e

try:
    from torchvision.models import resnet50, ResNet50_Weights
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError("torchvision is required") from e

from tqdm import tqdm


# ------------------------- defaults (your paths) -------------------------

DEFAULT_UNIFIED_CACHE_ROOT = "/home/data/pxt/USSeg/EchoNet_Merged/unified_cache_r256"
DEFAULT_CONTROLNET_JSON_ROOT = "/home/data/pxt/USSeg/EchoNet_Merged"  # contains train/val/test/controlnet_*.json


# ------------------------- utils -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def stem_from_path(p: str) -> str:
    return Path(p).stem

def normalize_key_for_resnet(k: str) -> str:
    """
    Map ckpt keys -> resnet keys.
    Your ckpt uses backbone_online.* ; we strip that prefix.
    """
    nk = k
    for prefix in ["module.", "model.", "net."]:
        if nk.startswith(prefix):
            nk = nk[len(prefix):]
    # your BYOL ckpt prefix
    if nk.startswith("backbone_online."):
        nk = nk[len("backbone_online."):]
    if nk.startswith("backbone_target."):
        # if you ever want target weights, change here; we use online by default
        nk = nk[len("backbone_target."):]
    return nk


# ------------------------- encoder -------------------------

class ResNet50Encoder(nn.Module):
    """ResNet50 -> 2048-d feature. Keep original conv1/layer1/... parameter names."""
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = resnet50(weights=None)
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B,2048)


def load_stage3_ckpt_into_resnet(enc: ResNet50Encoder, ckpt_path: str, device: torch.device) -> None:
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    enc_sd = enc.model.state_dict()
    mapped: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        nk = normalize_key_for_resnet(k)
        if nk in enc_sd and enc_sd[nk].shape == v.shape:
            mapped[nk] = v

    if len(mapped) == 0:
        ks = list(sd.keys())[:50]
        print("[Load-Debug] ckpt head keys:", ks)
        print("[Load-Debug] enc head keys:", list(enc_sd.keys())[:50])
        raise RuntimeError(f"Could not match any weights into ResNet. ckpt={ckpt_path}")

    enc_sd.update(mapped)
    enc.model.load_state_dict(enc_sd, strict=False)
    print(f"[Load] matched {len(mapped)} tensors into ResNet50 from: {ckpt_path}")


# ------------------------- pairing dataset -------------------------

@dataclass
class PairItem:
    idx: int
    stem: str
    raw_path: str
    gen_path: str
    prompt: str


class RawGenPairDataset(Dataset):
    def __init__(
        self,
        split: str,
        unified_cache_root: str,
        controlnet_json_root: str,
        sample_id: int = 0,
        image_size: int = 256,
        max_items: int = -1,
        strict: bool = False,
    ):
        super().__init__()
        self.split = split
        self.image_size = int(image_size)

        json_path = Path(controlnet_json_root) / split / f"controlnet_{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing json: {json_path}")

        gen_dir = Path(unified_cache_root) / split / "gen"
        if not gen_dir.exists():
            raise FileNotFoundError(f"Missing gen dir: {gen_dir}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items: List[PairItem] = []
        for i, it in enumerate(data):
            raw = it["source"]
            prompt = it.get("prompt", "")

            st = stem_from_path(raw)

            # A) same stem
            cand_a = gen_dir / f"{st}.png"
            # B) legacy index naming: split_000123_00.png
            cand_b = gen_dir / f"{split}_{i:06d}_{sample_id:02d}.png"

            gen = None
            if cand_a.exists():
                gen = str(cand_a)
            elif cand_b.exists():
                gen = str(cand_b)

            if gen is None:
                if strict:
                    raise FileNotFoundError(f"[strict] gen not found for idx={i}, stem={st}. tried: {cand_a}, {cand_b}")
                continue

            items.append(PairItem(idx=i, stem=st, raw_path=raw, gen_path=gen, prompt=prompt))

            if max_items > 0 and len(items) >= max_items:
                break

        if len(items) == 0:
            raise RuntimeError(f"No pairs found for split={split}. Check naming / paths.")

        self.items = items

        # transforms
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        raw = read_rgb(it.raw_path)
        gen = read_rgb(it.gen_path)
        raw_t = self.tf(raw)
        gen_t = self.tf(gen)
        return {
            "raw": raw_t,
            "gen": gen_t,
            "idx": it.idx,
            "stem": it.stem,
            "raw_path": it.raw_path,
            "gen_path": it.gen_path,
            "prompt": it.prompt,
        }


# ------------------------- scoring -------------------------

@torch.no_grad()
def compute_embeddings(enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
    z = enc(x)  # (B,2048)
    z = z.float()
    z = F.normalize(z, dim=-1, p=2)  # hypersphere
    return z

@torch.no_grad()
def score_batch(z_raw: torch.Tensor, z_gen: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Returns:
      cos: (B,)
      angle: (B,) in radians
      l2: (B,)
    """
    cos = (z_raw * z_gen).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos)
    l2 = torch.norm(z_raw - z_gen, dim=-1)
    return {"cos": cos, "angle": angle, "l2": l2}

def fit_diag_cov(z: torch.Tensor) -> torch.Tensor:
    """
    z: (N,D) normalized embeddings
    return: diag variance (D,)
    """
    var = z.var(dim=0, unbiased=False) + 1e-6
    return var

@torch.no_grad()
def mahal_diag(d: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    d: (B,D) difference
    var: (D,)
    returns: (B,) sqrt( sum(d^2 / var) )
    """
    return torch.sqrt((d * d / var[None, :]).sum(dim=-1).clamp_min(1e-12))


# ------------------------- main -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--unified_cache_root", type=str, default=DEFAULT_UNIFIED_CACHE_ROOT)
    p.add_argument("--controlnet_json_root", type=str, default=DEFAULT_CONTROLNET_JSON_ROOT)

    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--sample_id", type=int, default=0, help="if gen uses split_%06d_%02d.png, choose which sample id (00/01/...)")

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_items", type=int, default=-1)
    p.add_argument("--strict", action="store_true", help="fail if any pair missing")

    p.add_argument("--out_dir", type=str, default="", help="default: <save next to ckpt>/scores")
    p.add_argument("--mahal_diag", action="store_true", help="also output diagonal-Mahalanobis (fit on TRAIN raw embeddings)")

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # build encoder + load ckpt
    enc = ResNet50Encoder(pretrained=False).to(device).eval()
    load_stage3_ckpt_into_resnet(enc, args.ckpt, device=device)

    # out dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.ckpt).resolve().parent.parent / "scores"  # .../ckpt/epoch_050.pt -> .../scores
    ensure_dir(out_dir)

    # If need Mahalanobis(diag), fit on TRAIN raw embeddings (paired items)
    diag_var = None
    if args.mahal_diag:
        if "train" not in splits:
            print("[Mahalanobis] You didn't include train in --splits; still fitting var on train.")
        train_ds = RawGenPairDataset(
            split="train",
            unified_cache_root=args.unified_cache_root,
            controlnet_json_root=args.controlnet_json_root,
            sample_id=args.sample_id,
            image_size=args.image_size,
            max_items=args.max_items,
            strict=False,
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
        zs = []
        for batch in tqdm(train_dl, desc="[FitMahalanobis] embed train raw"):
            raw = batch["raw"].to(device, non_blocking=True)
            z_raw = compute_embeddings(enc, raw)
            zs.append(z_raw.cpu())
        z_all = torch.cat(zs, dim=0)  # (N,2048)
        diag_var = fit_diag_cov(z_all).to(device)
        print("[Mahalanobis] fitted diag variance on train raw embeddings.")

    # score each split
    for split in splits:
        ds = RawGenPairDataset(
            split=split,
            unified_cache_root=args.unified_cache_root,
            controlnet_json_root=args.controlnet_json_root,
            sample_id=args.sample_id,
            image_size=args.image_size,
            max_items=args.max_items,
            strict=args.strict,
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

        csv_path = out_dir / f"scores_{split}.csv"
        print(f"[{split}] pairs={len(ds)} -> {csv_path}")

        fieldnames = ["idx", "stem", "raw_path", "gen_path", "prompt", "cos", "angle_rad", "l2"]
        if args.mahal_diag:
            fieldnames.append("mahal_diag")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for batch in tqdm(dl, desc=f"[Score] {split}"):
                raw = batch["raw"].to(device, non_blocking=True)
                gen = batch["gen"].to(device, non_blocking=True)

                z_raw = compute_embeddings(enc, raw)
                z_gen = compute_embeddings(enc, gen)

                sc = score_batch(z_raw, z_gen)
                cos = sc["cos"].detach().cpu().numpy()
                ang = sc["angle"].detach().cpu().numpy()
                l2 = sc["l2"].detach().cpu().numpy()

                if args.mahal_diag:
                    if diag_var is None:
                        raise RuntimeError("diag_var is None but --mahal_diag is set")
                    md = mahal_diag(z_raw - z_gen, diag_var).detach().cpu().numpy()
                else:
                    md = None

                B = raw.shape[0]
                for i in range(B):
                    row = {
                        "idx": int(batch["idx"][i]),
                        "stem": str(batch["stem"][i]),
                        "raw_path": str(batch["raw_path"][i]),
                        "gen_path": str(batch["gen_path"][i]),
                        "prompt": str(batch["prompt"][i]),
                        "cos": float(cos[i]),
                        "angle_rad": float(ang[i]),
                        "l2": float(l2[i]),
                    }
                    if args.mahal_diag:
                        row["mahal_diag"] = float(md[i])
                    w.writerow(row)

        print(f"[{split}] done.")

    print(f"All done. CSVs in: {out_dir}")


if __name__ == "__main__":
    main()
