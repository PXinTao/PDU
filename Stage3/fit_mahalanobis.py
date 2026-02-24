# DomainUnifiedSegmentation/stage3_hypersphere/fit_mahalanobis.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision.models as models

from .dataset_pairs import RawGenPairDataset


def load_encoder_from_byol_ckpt(ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # rebuild same backbone
    backbone = models.resnet50(weights=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()

    # load online encoder weights
    # online_encoder is wrapper(backbone) so state dict keys match wrapper
    online_sd = ckpt["online_encoder"]
    # online_encoder has fields: backbone.*
    # We load to wrapper then return backbone
    class Wrap(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.backbone = bb
        def forward(self, x): return self.backbone(x)

    wrapper = Wrap(backbone)
    wrapper.load_state_dict(online_sd, strict=False)
    return backbone


@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    feats = []
    encoder.eval().to(device)
    for batch in tqdm(dl):
        x = batch["raw"].to(device)  # (B,3,H,W) in [0,1]
        z = encoder(x)               # (B,feat_dim)
        z = F.normalize(z, dim=-1)   # hypersphere
        feats.append(z.cpu())
    return torch.cat(feats, dim=0)   # (N,D)


def main():
    RAW_TRAIN = "/home/data/pxt/USSeg/EchoNet_Merged/train/imgs"

    ap = argparse.ArgumentParser()
    ap.add_argument("--byol_ckpt", required=True, type=str)
    ap.add_argument("--raw_dir", default=RAW_TRAIN, type=str)
    ap.add_argument("--image_size", default=256, type=int)
    ap.add_argument("--batch_size", default=128, type=int)
    ap.add_argument("--num_workers", default=8, type=int)
    ap.add_argument("--device", default="cuda", type=str)

    ap.add_argument("--shrink", default=0.1, type=float, help="cov shrinkage lambda (0~1)")
    ap.add_argument("--out_stats", default="/home/data/pxt/USSeg/stage3_byol_hypersphere/mahalanobis_stats.pt", type=str)

    # score calibration
    ap.add_argument("--p_tau", default=0.95, type=float, help="quantile for tau on source distances")
    ap.add_argument("--beta", default=20.0, type=float, help="sigmoid sharpness for score")
    args = ap.parse_args()

    device = torch.device(args.device)

    # dataset: raw only
    ds = RawGenPairDataset(raw_dir=args.raw_dir, image_size=args.image_size, gen_dir=None, pair_gen=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    encoder = load_encoder_from_byol_ckpt(args.byol_ckpt)
    feats = extract_features(encoder, dl, device=device)  # (N,D) on cpu
    N, D = feats.shape
    print(f"[feat] N={N} D={D}")

    # mu
    mu = feats.mean(dim=0, keepdim=True)  # (1,D)

    # covariance
    xc = feats - mu
    cov = (xc.t() @ xc) / max(1, (N - 1))  # (D,D)
    # shrinkage
    lam = float(args.shrink)
    cov = (1 - lam) * cov + lam * torch.eye(D)

    inv_cov = torch.linalg.inv(cov)

    # compute source distances for tau
    d2 = torch.einsum("nd,dd,nd->n", xc, inv_cov, xc).clamp_min(0)
    d = torch.sqrt(d2)
    tau = torch.quantile(d, torch.tensor(args.p_tau)).item()

    stats = {
        "mu": mu.squeeze(0),           # (D,)
        "inv_cov": inv_cov,            # (D,D)
        "shrink": lam,
        "tau": tau,
        "beta": float(args.beta),
        "feat_dim": D,
        "p_tau": float(args.p_tau),
        "byol_ckpt": args.byol_ckpt,
        "raw_dir": args.raw_dir,
    }

    out = Path(args.out_stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, str(out))
    print(f"[DONE] saved stats -> {out}")
    print(f"tau(p={args.p_tau}) = {tau:.4f}")


if __name__ == "__main__":
    main()
