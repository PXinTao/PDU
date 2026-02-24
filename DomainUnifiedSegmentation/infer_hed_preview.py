import os
import argparse
from glob import glob

import numpy as np
import torch
import cv2

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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--max_images", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_prob = os.path.join(args.out_dir, "prob")
    out_bin = os.path.join(args.out_dir, "bin")
    os.makedirs(out_prob, exist_ok=True)
    os.makedirs(out_bin, exist_ok=True)

    device = torch.device(args.device)
    net = HEDNet().to(device).eval()
    load_ckpt(net, args.ckpt, device)

    paths = sorted(glob(os.path.join(args.images_dir, "*.png")) + glob(os.path.join(args.images_dir, "*.jpg")))
    paths = paths[: args.max_images]
    print(f"Infer {len(paths)} images, resize={args.resize}, thr={args.threshold}")
    print(f"Save to: {args.out_dir}")

    for p in paths:
        bn = os.path.basename(p)

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(img).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)

        out = net(x)              # list of 6 probs: [p1..p5,fused], already sigmoid
        fused_prob = out[-1]      # (1,1,H,W) in (0,1)
        prob = fused_prob.squeeze().detach().cpu().numpy()

        prob_u8 = (prob * 255.0).clip(0, 255).astype(np.uint8)
        bin_u8 = (prob >= args.threshold).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(out_prob, bn), prob_u8)
        cv2.imwrite(os.path.join(out_bin, bn), bin_u8)

    print("Done.")


if __name__ == "__main__":
    main()
