import os
import argparse
import math
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as tvm

# Make sure PYTHONPATH includes repo root so this import works.
from models_byol import BYOLHypersphere, BYOLConfig


# =========================================================
# IO + utils
# =========================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def try_load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def load_rgb(path: str, image_size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)

def make_pair_panel(raw_path: str, gen_path: str, title: str, out_path: Path, target_h=256, pad=8):
    raw = Image.open(raw_path).convert("RGB")
    gen = Image.open(gen_path).convert("RGB")

    def resize_keep(img: Image.Image, target_h: int):
        w, h = img.size
        if h == target_h:
            return img
        new_w = int(round(w * (target_h / float(h))))
        return img.resize((new_w, target_h), resample=Image.BILINEAR)

    raw = resize_keep(raw, target_h)
    gen = resize_keep(gen, target_h)

    font = try_load_font(16)

    w = raw.size[0] + gen.size[0] + pad * 3
    h = target_h + pad * 3 + 22
    canvas = Image.new("RGB", (w, h), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, pad), title, fill=(230, 230, 230), font=font)

    y0 = pad * 2 + 22
    canvas.paste(raw, (pad, y0))
    canvas.paste(gen, (pad * 2 + raw.size[0], y0))

    draw.text((pad, y0 + target_h + 2), "RAW", fill=(200, 200, 200), font=font)
    draw.text((pad * 2 + raw.size[0], y0 + target_h + 2), "GEN", fill=(200, 200, 200), font=font)

    canvas.save(str(out_path))

def make_mosaic(image_paths, out_path: Path, cols=6, pad=6, bg=10):
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    if len(imgs) == 0:
        return
    w0, h0 = imgs[0].size
    rows = int(math.ceil(len(imgs) / cols))
    W = cols * w0 + (cols + 1) * pad
    H = rows * h0 + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (bg, bg, bg))
    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        x = pad + c * (w0 + pad)
        y = pad + r * (h0 + pad)
        canvas.paste(im, (x, y))
    canvas.save(str(out_path))


def read_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        has_named = any(h in header for h in ["cos", "angle", "l2", "raw_path", "gen_path"])
        if has_named:
            idx_map = {h: i for i, h in enumerate(header)}
            for r in reader:
                rows.append({
                    "idx": r[idx_map.get("idx", 0)],
                    "name": r[idx_map.get("name", 1)] if "name" in idx_map else r[1],
                    "raw_path": r[idx_map.get("raw_path", 2)],
                    "gen_path": r[idx_map.get("gen_path", 3)],
                    "prompt": r[idx_map.get("prompt", 4)] if "prompt" in idx_map else "",
                    "cos_raw_gen": float(r[idx_map.get("cos", 5)]),
                    "angle_raw_gen": float(r[idx_map.get("angle", 6)]),
                    "l2_raw_gen": float(r[idx_map.get("l2", 7)]),
                })
        else:
            for r in reader:
                rows.append({
                    "idx": r[0],
                    "name": r[1],
                    "raw_path": r[2],
                    "gen_path": r[3],
                    "prompt": r[4] if len(r) > 4 else "",
                    "cos_raw_gen": float(r[5]),
                    "angle_raw_gen": float(r[6]),
                    "l2_raw_gen": float(r[7]),
                })
    return rows, header


# =========================================================
# Embedding extraction using your BYOLHypersphere
# =========================================================

class RawPathDataset(Dataset):
    def __init__(self, paths: List[str], image_size: int, imagenet_norm: bool):
        self.paths = paths
        self.image_size = image_size
        self.imagenet_norm = imagenet_norm
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = load_rgb(p, self.image_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)

        if self.imagenet_norm:
            x = (x - self.mean) / self.std

        return x, p


def build_resnet50_backbone(pretrained: bool = True) -> Tuple[nn.Module, int]:
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


from collections import Counter

def load_byol_model(ckpt_path: str, device: str, *, pretrained_backbone: bool = True) -> BYOLHypersphere:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # 1) strip "module."
    sd = {}
    for k, v in state.items():
        sd[k[len("module."):] if k.startswith("module.") else k] = v

    # 2) remap keys to match BYOLHypersphere naming
    remapped = {}
    for k, v in sd.items():
        nk = k

        # backbone
        if nk.startswith("backbone_online."):
            nk = "online_encoder.backbone." + nk[len("backbone_online."):]
        elif nk.startswith("backbone_target."):
            nk = "target_encoder.backbone." + nk[len("backbone_target."):]
        elif nk.startswith("backbone_momentum."):
            nk = "target_encoder.backbone." + nk[len("backbone_momentum."):]

        # projector
        elif nk.startswith("projector_online."):
            nk = "projector." + nk[len("projector_online."):]
        elif nk.startswith("projector_target."):
            nk = "target_projector." + nk[len("projector_target."):]
        elif nk.startswith("projector_momentum."):
            nk = "target_projector." + nk[len("projector_momentum."):]

        # predictor
        elif nk.startswith("predictor_online."):
            nk = "predictor." + nk[len("predictor_online."):]
        # predictor.* stays predictor.*

        remapped[nk] = v

    print(f"[ckpt] keys: raw={len(state)} stripped={len(sd)} remapped={len(remapped)}")

    # 3) infer MLP dims from checkpoint (IMPORTANT!)
    # projector: Linear(feat_dim -> proj_hidden) then Linear(proj_hidden -> proj_dim)
    if "projector.net.0.weight" not in remapped or "projector.net.3.weight" not in remapped:
        raise KeyError("Cannot find projector weights in ckpt after remap. Check key names.")

    proj_hidden = int(remapped["projector.net.0.weight"].shape[0])
    proj_dim = int(remapped["projector.net.3.weight"].shape[0])

    # predictor: Linear(proj_dim -> pred_hidden) then Linear(pred_hidden -> proj_dim)
    if "predictor.net.0.weight" not in remapped or "predictor.net.3.weight" not in remapped:
        raise KeyError("Cannot find predictor weights in ckpt after remap. Check key names.")

    pred_hidden = int(remapped["predictor.net.0.weight"].shape[0])

    # sanity check
    print(f"[ckpt] inferred dims: proj_hidden={proj_hidden}, proj_dim={proj_dim}, pred_hidden={pred_hidden}")

    # 4) build model with matching dims
    backbone, feat_dim = build_resnet50_backbone(pretrained=pretrained_backbone)

    cfg = BYOLConfig(
        proj_dim=proj_dim,
        proj_hidden=proj_hidden,
        pred_hidden=pred_hidden,
        ema=0.99,
        sphere_m=4.0
    )

    model = BYOLHypersphere(encoder=backbone, feat_dim=feat_dim, cfg=cfg)

    # 5) now load
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    if len(unexpected) > 0:
        print("[warn] unexpected keys (head):", unexpected[:30])
        pref = Counter([u.split(".")[0] for u in unexpected])
        print("[warn] unexpected prefix stats:", pref.most_common(10))

    if len(missing) > 0:
        print("[warn] missing keys (head):", missing[:30])
        pref = Counter([m.split(".")[0] for m in missing])
        print("[warn] missing prefix stats:", pref.most_common(10))

    model.to(device)
    model.eval()
    return model


# def load_gray01_resize(path: str, target_hw):
#     H, W = target_hw
#     img = Image.open(path).convert("L").resize((W, H), resample=Image.BILINEAR)
#     return np.asarray(img, dtype=np.float32) / 255.0

@torch.no_grad()
def extract_z_online_projector(model: BYOLHypersphere, x: torch.Tensor) -> torch.Tensor:
    o = model.online_encoder(x)
    z = model.projector(o)
    z = F.normalize(z, dim=-1, p=2)
    return z


@torch.no_grad()
def get_embeddings(
    model: BYOLHypersphere,
    paths: List[str],
    device: str,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    imagenet_norm: bool
) -> Dict[str, np.ndarray]:
    ds = RawPathDataset(paths, image_size=image_size, imagenet_norm=imagenet_norm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    out: Dict[str, np.ndarray] = {}
    for xb, pb in dl:
        xb = xb.to(device, non_blocking=True)
        z = extract_z_online_projector(model, xb)
        z = z.detach().cpu().numpy()
        for p, zi in zip(pb, z):
            out[str(p)] = zi.astype(np.float32)
    return out


# =========================================================
# Prototype + Mahalanobis (diagonal covariance, stable)
# =========================================================

def estimate_mu_var_diag(z_list: List[np.ndarray], eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.stack(z_list, axis=0).astype(np.float32)
    mu = Z.mean(axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-12)

    X = Z - mu[None, :]
    var = (X * X).mean(axis=0) + eps
    return mu.astype(np.float32), var.astype(np.float32)

def mahalanobis_d_diag(z: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    diff = z - mu
    d2 = float(np.sum((diff * diff) / var))
    return math.sqrt(max(d2, 0.0))

def cos_to_mu(z: np.ndarray, mu: np.ndarray) -> float:
    z = z / (np.linalg.norm(z) + 1e-12)
    mu = mu / (np.linalg.norm(mu) + 1e-12)
    return float(np.clip(np.sum(z * mu), -1.0, 1.0))


# =========================================================
# Reliability score (calibrated by SOURCE distance distribution)
# =========================================================

def compute_S_from_scale(cos_term: float, dM: float, scale: float, alpha: float, beta: float) -> float:
    """
    S = alpha * (cos_term+1)/2 + (1-alpha) * exp(-beta * (dM/scale))
    cos_term should be in [-1,1].
    scale calibrated from SOURCE dM distribution (e.g., p95).
    """
    cos_term = float(np.clip(cos_term, -1.0, 1.0))
    s_cos = 0.5 * (cos_term + 1.0)

    scale = float(max(scale, 1e-6))
    d_eff = float(dM) / scale
    s_dist = math.exp(-beta * d_eff)

    S = alpha * s_cos + (1.0 - alpha) * s_dist
    return float(np.clip(S, 0.0, 1.0))


# =========================================================
# Haar wavelet fusion (numpy, single level)
# =========================================================

def load_gray01(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return (np.asarray(img, dtype=np.float32) / 255.0)
def load_gray01_resize(path: str, target_hw):
    H, W = target_hw
    img = Image.open(path).convert("L").resize((W, H), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def dwt2_haar(x: np.ndarray):
    H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    LL = (a + b + c + d) * 0.5
    LH = (a - b + c - d) * 0.5
    HL = (a + b - c - d) * 0.5
    HH = (a - b - c + d) * 0.5
    return LL, LH, HL, HH

def idwt2_haar(LL, LH, HL, HH):
    h, w = LL.shape
    out = np.zeros((h * 2, w * 2), dtype=np.float32)
    a = (LL + LH + HL + HH) * 0.5
    b = (LL - LH + HL - HH) * 0.5
    c = (LL + LH - HL - HH) * 0.5
    d = (LL - LH - HL + HH) * 0.5
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    return out

def wavelet_fuse(raw: np.ndarray, uni: np.ndarray, S: float, lambda_max: float, eta: float, delta_max: float) -> np.ndarray:
    H, W = raw.shape
    if H % 2 or W % 2:
        H2 = H - (H % 2)
        W2 = W - (W % 2)
        raw = raw[:H2, :W2]
        uni = uni[:H2, :W2]

    LLr, LHr, HLr, HHr = dwt2_haar(raw)
    LLu, LHu, HLu, HHu = dwt2_haar(uni)

    S = float(np.clip(S, 0.0, 1.0))
    gamma = float(np.clip(lambda_max * (S ** eta), 0.0, 1.0))
    delta = float(np.clip(delta_max * S, 0.0, 1.0))

    LLf = (1 - delta) * LLr + delta * LLu
    LHf = (1 - gamma) * LHr + gamma * LHu
    HLf = (1 - gamma) * HLr + gamma * HLu
    HHf = (1 - gamma) * HHr + gamma * HHu

    fused = idwt2_haar(LLf, LHf, HLf, HHf)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--source_csv", required=True, type=str, help="scores_train.csv (source)")
    ap.add_argument("--target_csv", required=True, type=str, help="scores_val.csv or scores_test.csv (target split)")

    ap.add_argument("--ckpt", required=True, type=str, help="epoch_050.pt")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--device", default="cuda", type=str)

    # embedding extraction
    ap.add_argument("--image_size", default=256, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--num_workers", default=8, type=int)
    ap.add_argument("--imagenet_norm", action="store_true", help="apply imagenet mean/std (recommended for resnet)")
    ap.add_argument("--no_pretrained_backbone", action="store_true", help="build resnet50 without imagenet weights")

    # score
    ap.add_argument("--alpha", default=0.6, type=float)
    ap.add_argument("--beta", default=2.0, type=float)
    ap.add_argument("--scale_q", default=0.95, type=float, help="quantile for source dM calibration (e.g., 0.9/0.95/0.99)")

    # fusion
    ap.add_argument("--do_fuse", action="store_true")
    ap.add_argument("--lambda_max", default=1.0, type=float)
    ap.add_argument("--eta", default=2.0, type=float)
    ap.add_argument("--delta_max", default=0.0, type=float)

    # visualization
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--thumb_h", type=int, default=256)
    ap.add_argument("--mosaic_cols", type=int, default=6)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows_src, _ = read_rows(args.source_csv)
    rows_tgt, _ = read_rows(args.target_csv)
    if len(rows_src) == 0 or len(rows_tgt) == 0:
        raise RuntimeError("Empty CSV.")

    src_paths = [r["raw_path"] for r in rows_src]
    tgt_paths = [r["raw_path"] for r in rows_tgt]
    all_paths = sorted(set(src_paths + tgt_paths))

    print("[1] Load BYOLHypersphere ckpt...")
    model = load_byol_model(
        args.ckpt,
        args.device,
        pretrained_backbone=(not args.no_pretrained_backbone),
    )

    print("[2] Extract embeddings z = normalize(projector(online_encoder(x))) ...")
    emb = get_embeddings(
        model,
        all_paths,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        imagenet_norm=args.imagenet_norm
    )
    D = len(next(iter(emb.values())))
    print(f"  embedding dim D={D}")

    print("[3] Estimate prototype (mu, diag var) from SOURCE...")
    z_src = [emb[p] for p in src_paths if p in emb]
    if len(z_src) < 10:
        raise RuntimeError("Too few source embeddings. Check paths.")
    mu, var = estimate_mu_var_diag(z_src, eps=1e-5)

    # ---- distance scale calibration from source ----
    dM_src = []
    for p in src_paths:
        if p in emb:
            dM_src.append(mahalanobis_d_diag(emb[p], mu, var))
    dM_src = np.asarray(dM_src, dtype=np.float32)
    scale = float(np.quantile(dM_src, args.scale_q))
    scale = max(scale, 1e-6)
    print(f"[calib] source dM scale (q={args.scale_q}) = {scale:.4f} (N={len(dM_src)})")

    # ---- process target ----
    out_rows = []
    dM_list, cosmu_list, S_list = [], [], []

    fused_dir = out_dir / "fused" if args.do_fuse else None
    if fused_dir is not None:
        ensure_dir(fused_dir)

    print("[4] Compute dM + S for TARGET (S uses cos_raw_gen + exp(-beta*dM/scale)) ...")
    for r in rows_tgt:
        p = r["raw_path"]
        if p not in emb:
            continue
        z = emb[p]
        dM = mahalanobis_d_diag(z, mu, var)
        c_mu = cos_to_mu(z, mu)

        # similarity term: generation reliability from Stage3 inference
        cos_term = r["cos_raw_gen"]

        S = compute_S_from_scale(cos_term, dM, scale, args.alpha, args.beta)

        r2 = dict(r)
        r2["cos_mu"] = float(c_mu)   # log only
        r2["dM"] = float(dM)
        r2["S"] = float(S)

        if args.do_fuse:
            raw = load_gray01(r["raw_path"])
            uni = load_gray01_resize(r["gen_path"], raw.shape)  # 对齐尺寸
            fused = wavelet_fuse(raw, uni, S, args.lambda_max, args.eta, args.delta_max)


            fused_path = fused_dir / f'{r["name"]}.png'
            Image.fromarray((fused * 255.0).astype(np.uint8)).save(str(fused_path))
            r2["fused_path"] = str(fused_path)
        else:
            r2["fused_path"] = ""

        out_rows.append(r2)
        dM_list.append(dM)
        cosmu_list.append(c_mu)
        S_list.append(S)

    print("==== Target stats ====")
    print("N:", len(out_rows))
    if len(out_rows) > 0:
        print("cos_mu mean/std:", float(np.mean(cosmu_list)), float(np.std(cosmu_list)))
        print("dM     mean/std:", float(np.mean(dM_list)), float(np.std(dM_list)))
        print("S      mean/std:", float(np.mean(S_list)), float(np.std(S_list)))

    out_csv = out_dir / "scores_with_S.csv"
    fieldnames = [
        "idx", "name", "raw_path", "gen_path", "prompt",
        "cos_raw_gen", "angle_raw_gen", "l2_raw_gen",
        "cos_mu", "dM", "S", "fused_path"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rr in out_rows:
            w.writerow({k: rr.get(k, "") for k in fieldnames})
    print("[saved]", out_csv)

    # ---- visualize best/worst by S ----
    worst_dir = out_dir / "worst_S"
    best_dir = out_dir / "best_S"
    ensure_dir(worst_dir)
    ensure_dir(best_dir)

    out_rows_sorted = sorted(out_rows, key=lambda x: x["S"])
    worst = out_rows_sorted[: args.topk]
    best = out_rows_sorted[-args.topk:][::-1]

    worst_panels, best_panels = [], []
    for i, r in enumerate(worst):
        title = f'WORST S={r["S"]:.3f} dM={r["dM"]:.2f} cos_rg={r["cos_raw_gen"]:.3f} {r["name"]}'
        out_p = worst_dir / f'{i:04d}_{r["name"]}.png'
        try:
            make_pair_panel(r["raw_path"], r["gen_path"], title, out_p, target_h=args.thumb_h)
            worst_panels.append(str(out_p))
        except Exception as e:
            print("[skip worst]", r["name"], e)

    for i, r in enumerate(best):
        title = f'BEST  S={r["S"]:.3f} dM={r["dM"]:.2f} cos_rg={r["cos_raw_gen"]:.3f} {r["name"]}'
        out_p = best_dir / f'{i:04d}_{r["name"]}.png'
        try:
            make_pair_panel(r["raw_path"], r["gen_path"], title, out_p, target_h=args.thumb_h)
            best_panels.append(str(out_p))
        except Exception as e:
            print("[skip best]", r["name"], e)

    make_mosaic(worst_panels, out_dir / "mosaic_worst_S.png", cols=args.mosaic_cols)
    make_mosaic(best_panels, out_dir / "mosaic_best_S.png", cols=args.mosaic_cols)

    print("==== Done ====")
    if args.do_fuse:
        print("fused:", fused_dir)
    print("best/worst:", best_dir, worst_dir)


if __name__ == "__main__":
    main()
