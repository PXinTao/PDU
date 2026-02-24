import os
import argparse
from glob import glob

import numpy as np
import torch
import cv2
from tqdm import tqdm

# 你按自己的工程路径改这一行：
from models.edge.hed_net import HEDNet


def load_ckpt(net, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ["module.", "net.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    net.load_state_dict(cleaned, strict=False)


def list_images(images_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for e in exts:
        paths += glob(os.path.join(images_dir, e))
    return sorted(paths)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str, help="split imgs dir, e.g. .../train/imgs")
    ap.add_argument("--ckpt", required=True, type=str, help="HED checkpoint")
    ap.add_argument("--out_dir", required=True, type=str, help="where to save hints (grayscale 0-255)")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")

    # output type
    ap.add_argument("--save_prob", action="store_true", help="save prob grayscale (0-255). Recommended for ControlNet.")
    ap.add_argument("--save_bin", action="store_true", help="also save bin (thresholded) for debugging/metrics.")
    ap.add_argument("--threshold", type=float, default=0.35)

    # make edges more/less salient without retraining
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="prob <- prob^gamma. gamma<1 makes edges brighter/fatter; gamma>1 makes them thinner.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_prob_dir = os.path.join(args.out_dir, "prob")
    out_bin_dir = os.path.join(args.out_dir, "bin")

    if args.save_prob:
        os.makedirs(out_prob_dir, exist_ok=True)
    if args.save_bin:
        os.makedirs(out_bin_dir, exist_ok=True)

    device = torch.device(args.device)
    net = HEDNet().to(device).eval()
    load_ckpt(net, args.ckpt, device)

    paths = list_images(args.images_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No images found in: {args.images_dir}")

    print(f"[Export] N={len(paths)}  resize={args.resize}  gamma={args.gamma}")
    print(f"[Export] out_dir={args.out_dir}")
    if args.save_bin:
        print(f"[Export] bin threshold={args.threshold}")

    for p in tqdm(paths, desc="Export HED hints", dynamic_ncols=True):
        bn = os.path.basename(p)

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(img).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)

        out = net(x)         # your net returns probabilities already
        fused_prob = out[-1] # (1,1,H,W) in (0,1)
        prob = fused_prob.squeeze().detach().cpu().numpy().astype(np.float32)

        # optional saliency control
        if args.gamma != 1.0:
            prob = np.power(np.clip(prob, 0.0, 1.0), args.gamma)

        if args.save_prob:
            prob_u8 = (prob * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_prob_dir, bn), prob_u8)

        if args.save_bin:
            bin_u8 = (prob >= args.threshold).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(out_bin_dir, bn), bin_u8)

    print("Done.")


if __name__ == "__main__":
    main()



