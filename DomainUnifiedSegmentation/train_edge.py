"""
Train HED-like edge detector (probability-output version).

This trainer assumes:
- HEDNet.forward returns probabilities (after sigmoid) for side outputs + fused:
  [p1_prob, p2_prob, p3_prob, p4_prob, p5_prob, fused_prob]
  all shapes: (B,1,H,W), values in (0,1)

Therefore:
- DO NOT use BCEWithLogitsLoss / hed_deep_supervision_loss here.
- Use probability-based edge loss (your legacy CE/Dice) from edge_losses.py.

Example (precomputed edge GT in {0,255}):

python DomainUnifiedSegmentation/train_edge.py \
  --images_dir /home/data/pxt/USSeg/EchoNet_Merged/train/imgs \
  --edge_dir   /home/data/pxt/USSeg/EchoNet_Merged/train/edge \
  --out_ckpt   /home/data/pxt/USSeg/ckpts/hed_lv_prob.pth \
  --init_from_torchvision vgg16_bn \
  --epochs 30 --batch_size 8 --lr 1e-4 --resize 256 --num_workers 2 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.image_mask_dataset import ImageMaskFolderDataset
from models.edge.hed_net import HEDNet
from models.edge.edge_losses import EdgeCrossEntropyDice
from utils.augment import compose_default
from utils.edge_ops import boundary_from_mask
from utils.io import ensure_dir

    # export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0      选卡
# # nohup train_edge.py \
#   --images_dir /home/data/pxt/USSeg/EchoNet_Merged/train/imgs \
#   --edge_dir   /home/data/pxt/USSeg/EchoNet_Merged/train/edge \
#   --out_ckpt   /home/data/pxt/USSeg/ckpts/hed_lv_prob.pth \
#   --init_from_torchvision vgg16_bn \
#   --epochs 50 --batch_size 8 --lr 1e-4 --resize 256 \
#   --num_workers 2 --device cuda
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--images_dir", type=str, required=True)

    # Either provide edge_dir OR masks_dir (fallback)
    p.add_argument("--edge_dir", type=str, default="",
                   help="Precomputed edge GT dir (0/255). If set, masks_dir is ignored.")
    p.add_argument("--masks_dir", type=str, default="",
                   help="Mask dir used only if edge_dir is not set (derive edges from masks).")

    p.add_argument("--out_ckpt", type=str, required=True)

    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")

    # loss hyperparams (your style)
    p.add_argument("--dice_weight", type=float, default=0.0,
                   help="Dice term weight multiplier inside EdgeCrossEntropyDice.")
    p.add_argument("--side_weight", type=float, default=1.0,
                   help="Side-output CE contribution scaling inside EdgeCrossEntropyDice.")

    # Used only when deriving from masks
    p.add_argument("--mask_edge_dilation", type=int, default=1)

    # init
    p.add_argument("--init_weights", type=str, default="",
                   help="Optional local pretrained ckpt. If set, overrides torchvision init.")
    p.add_argument("--init_from_torchvision", type=str, default="vgg16_bn",
                   choices=["none", "vgg16", "vgg16_bn"],
                   help="If init_weights not set, init backbone from torchvision pretrained weights.")

    return p.parse_args()


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ["module.", "net.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def load_init_weights(net: torch.nn.Module, init_path: str, device: torch.device) -> None:
    state = torch.load(init_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at: {init_path}")
    state = _clean_state_dict_keys(state)
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[Init] Loaded init weights from: {init_path}")
    if missing:
        print(f"[Init] Missing keys (head): {missing[:10]} (total {len(missing)})")
    if unexpected:
        print(f"[Init] Unexpected keys (head): {unexpected[:10]} (total {len(unexpected)})")


@torch.no_grad()
def init_from_torchvision_vgg(net: HEDNet, variant: str, device: torch.device) -> None:
    """
    Map torchvision VGG16 conv weights into HEDNet conv layers by shape/order.
    NOTE: This assumes your HEDNet conv layout is VGG-like (2/2/3/3/3 convs).
    """
    try:
        from torchvision import models
    except Exception as e:
        raise RuntimeError("torchvision is required for init_from_torchvision") from e

    if variant == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).to(device)
    elif variant == "vgg16_bn":
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1
        vgg = models.vgg16_bn(weights=weights).to(device)
    else:
        return

    vgg_convs = [m for m in vgg.features.modules()
                 if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)]

    hed_convs = [m for m in net.modules()
                 if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)]

    loaded = 0
    vi = 0
    for hm in hed_convs:
        while vi < len(vgg_convs):
            vm = vgg_convs[vi]
            if vm.weight.shape == hm.weight.shape:
                hm.weight.copy_(vm.weight)
                if hm.bias is not None and vm.bias is not None:
                    hm.bias.copy_(vm.bias)
                loaded += 1
                vi += 1
                break
            vi += 1

    print(f"[Init] Torchvision {variant} pretrained loaded into {loaded} Conv2d layers (shape-matched).")


def main():
    args = parse_args()
    device = torch.device(args.device)

    use_precomputed_edge = bool(args.edge_dir)
    if (not use_precomputed_edge) and (not args.masks_dir):
        raise ValueError("Provide --edge_dir or --masks_dir")

    label_dir = args.edge_dir if use_precomputed_edge else args.masks_dir

    ds = ImageMaskFolderDataset(
        images_dir=args.images_dir,
        masks_dir=label_dir,
        resize=(args.resize, args.resize),
        to_grayscale=False,  # keep RGB
        transform=lambda img, mask: compose_default(img, mask),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    net = HEDNet().to(device)

    # init
    if args.init_weights:
        load_init_weights(net, args.init_weights, device)
    else:
        if args.init_from_torchvision != "none":
            init_from_torchvision_vgg(net, args.init_from_torchvision, device)
        else:
            print("[Init] Training from random init.")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Your probability-based loss (CE + optional Dice + side supervision)
    criterion = EdgeCrossEntropyDice(dice_weight=args.dice_weight, side_weight=args.side_weight)

    best_loss = float("inf")

    print("========== HED Training (prob outputs) ==========")
    print(f"images_dir : {args.images_dir}")
    print(f"label_dir  : {label_dir}")
    print(f"mode       : {'precomputed_edge' if use_precomputed_edge else 'derive_from_mask'}")
    print(f"resize     : {args.resize}")
    print(f"dice_weight: {args.dice_weight}  side_weight: {args.side_weight}")
    print("===============================================")

    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for img_t, mask_t, _ in pbar:
            img_t = img_t.to(device, non_blocking=True)

            if use_precomputed_edge:
                # mask_t is actually edge GT in {0,255} -> {0,1}, shape: (B,H,W)
                edge_t = (mask_t > 0).float().unsqueeze(1).to(device, non_blocking=True)  # (B,1,H,W)
            else:
                # derive edge GT from segmentation masks
                mask_np = mask_t.numpy()
                edges = []
                for i in range(mask_np.shape[0]):
                    e = boundary_from_mask(mask_np[i].astype(np.uint8), dilation=args.mask_edge_dilation)
                    edges.append(e)
                edge_np = np.stack(edges, axis=0)[:, None, :, :].astype(np.float32)  # (B,1,H,W)
                edge_t = torch.from_numpy(edge_np).to(device, non_blocking=True)

            out = net(img_t)
            if not isinstance(out, (list, tuple)) or len(out) != 6:
                raise RuntimeError("HEDNet.forward must return 6 tensors: 5 sides + 1 fused (all probabilities).")

            side_prob = out[:-1]   # list of 5 tensors (B,1,H,W) prob
            fused_prob = out[-1]   # (B,1,H,W) prob

            loss = criterion(fused_prob, edge_t, side_output=side_prob)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1
            pbar.set_postfix(loss=float(loss.item()))

        avg = running / max(1, n)
        print(f"Epoch {epoch}/{args.epochs} - loss={avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            out_path = Path(args.out_ckpt)
            ensure_dir(out_path.parent)
            torch.save({"state_dict": net.state_dict(), "epoch": epoch, "loss": best_loss}, str(out_path))
            print(f"Saved best checkpoint to: {out_path} (loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
