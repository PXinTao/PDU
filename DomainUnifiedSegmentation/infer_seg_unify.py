"""Inference-time domain unified segmentation.

Pipeline (per target image):
1) Extract edges (Canny/HED or multi-hint)
2) ControlNet diffusion generates a source-style image conditioned on edges (and optionally img2img)
3) Compute scores:
     S_struct = edge F1(raw, gen)
     S_style  = p_source(gen) from domain discriminator (optional)
     w        = fusion weight derived from the above
4) Fuse raw and gen to keep structure + enforce domain unification
5) Run segmentation model on fused (optionally ensemble with raw)

The key idea matches the multi-stage design in your notes: structure-guided generation
+ scoring + dynamic fusion.

Usage example:
python DomainUnifiedSegmentation/infer_seg_unify.py \
  --input_dir /path/to/target/images \
  --out_dir   /path/to/out \
  --seg_ckpt  /path/to/seg_unet.pth \
  --stage1_root "/path/to/ADAptation-main/Stage1&2.Diffusion Model" \
  --sd_ckpt /path/to/v1-5-pruned.ckpt \
  --controlnet_ckpt /path/to/control_sd15_canny.pth \
  --edge_method hed_controlnet \
  --prompt "Ultrasound of heart"
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from DomainUnifiedSegmentation.models.edge.extractor import EdgeExtractor, EdgeExtractorConfig
from DomainUnifiedSegmentation.models.domain.simple_cnn import DomainDiscriminator
from DomainUnifiedSegmentation.models.seg.unet import UNet
from DomainUnifiedSegmentation.unifier.controlnet_unifier import ControlNetUnifier, UnifierConfig
from DomainUnifiedSegmentation.utils.edge_ops import as_controlnet_hint
from DomainUnifiedSegmentation.utils.fusion import FusionConfig, fuse
from DomainUnifiedSegmentation.utils.io import ensure_dir, list_images, read_image, write_image
from DomainUnifiedSegmentation.utils.scoring import ScoreConfig, compute_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)

    # segmentation
    p.add_argument('--seg_ckpt', type=str, required=True)
    p.add_argument('--in_channels', type=int, default=1)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--base_channels', type=int, default=32)
    p.add_argument('--seg_ensemble', action='store_true', help='ensemble logits from raw and fused using score')

    # diffusion unifier
    p.add_argument('--stage1_root', type=str, required=True)
    p.add_argument('--cldm_yaml', type=str, default='./models/cldm_v15.yaml')
    p.add_argument('--sd_ckpt', type=str, required=True)
    p.add_argument('--controlnet_ckpt', type=str, required=True)
    p.add_argument('--finetuned_ckpt', type=str, default='')

    p.add_argument('--prompt', type=str, default='Ultrasound')
    p.add_argument('--ddim_steps', type=int, default=50)
    p.add_argument('--guidance_scale', type=float, default=9.0)
    p.add_argument('--control_strength', type=float, default=1.0)
    p.add_argument('--guess_mode', action='store_true')
    p.add_argument('--use_img2img', action='store_true')
    p.add_argument('--img2img_strength', type=float, default=0.6)

    # edge extractor
    p.add_argument('--edge_method', type=str, default='hed_controlnet', choices=['canny','hed_controlnet','hed_finetuned','multi'])
    p.add_argument('--hed_weights', type=str, default='')

    # domain discriminator (optional)
    p.add_argument('--domain_disc_ckpt', type=str, default='')

    # scoring / fusion
    p.add_argument('--style_weight', type=float, default=0.6)
    p.add_argument('--struct_weight', type=float, default=0.4)
    p.add_argument('--gamma_style', type=float, default=1.0)
    p.add_argument('--gamma_struct', type=float, default=1.0)
    p.add_argument('--edge_thr', type=float, default=0.2)
    p.add_argument('--edge_dilation', type=int, default=1)
    p.add_argument('--tau_low', type=float, default=0.35)
    p.add_argument('--tau_high', type=float, default=0.75)

    p.add_argument('--fusion_method', type=str, default='frequency', choices=['alpha','frequency','edge_overlay'])
    p.add_argument('--fusion_sigma', type=float, default=3.0)
    p.add_argument('--fusion_edge_dilation', type=int, default=2)
    p.add_argument('--fusion_edge_thr', type=float, default=0.2)

    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--resize', type=int, default=512)
    p.add_argument('--seed', type=int, default=-1)

    return p.parse_args()


def load_ckpt_state_dict(path: str, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # strip common prefixes
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ['module.', 'net.', 'model.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def preprocess_for_seg(img_rgb: np.ndarray, in_channels: int, resize: int) -> torch.Tensor:
    import cv2
    img = cv2.resize(img_rgb, (resize, resize), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    if in_channels == 1:
        x = x.mean(axis=2, keepdims=True)
    x_t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float()
    return x_t


def main():
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'gen')
    ensure_dir(out_dir / 'fused')
    ensure_dir(out_dir / 'pred')

    # --- Load segmentation model ---
    seg = UNet(in_channels=args.in_channels, num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    seg_sd = load_ckpt_state_dict(args.seg_ckpt, device)
    seg.load_state_dict(seg_sd, strict=False)
    seg.eval()

    # --- Load domain discriminator (optional) ---
    domain_disc = None
    if args.domain_disc_ckpt:
        domain_disc = DomainDiscriminator(in_channels=args.in_channels).to(device)
        dd_sd = load_ckpt_state_dict(args.domain_disc_ckpt, device)
        domain_disc.load_state_dict(dd_sd, strict=False)
        domain_disc.eval()

    # --- Edge extractor ---
    edge_cfg = EdgeExtractorConfig(
        method=args.edge_method,
        hed_weights=args.hed_weights if args.hed_weights else None,
        stage1_root=args.stage1_root,
        device=args.device,
    )
    edge_extractor = EdgeExtractor(edge_cfg)

    # --- Diffusion unifier ---
    uni_cfg = UnifierConfig(
        stage1_root=args.stage1_root,
        cldm_yaml=args.cldm_yaml,
        sd_ckpt=args.sd_ckpt,
        controlnet_ckpt=args.controlnet_ckpt,
        finetuned_ckpt=args.finetuned_ckpt if args.finetuned_ckpt else None,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        strength=args.control_strength,
        guess_mode=args.guess_mode,
        use_img2img=args.use_img2img,
        img2img_strength=args.img2img_strength,
        prompt=args.prompt,
        resolution=args.resize,
    )
    unifier = ControlNetUnifier(uni_cfg, device=args.device)

    # --- scoring and fusion config ---
    score_cfg = ScoreConfig(
        edge_thr=args.edge_thr,
        edge_dilation=args.edge_dilation,
        style_weight=args.style_weight,
        struct_weight=args.struct_weight,
        gamma_style=args.gamma_style,
        gamma_struct=args.gamma_struct,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
    )

    fusion_cfg = FusionConfig(
        method=args.fusion_method,
        sigma=args.fusion_sigma,
        edge_dilation=args.fusion_edge_dilation,
        edge_thr=args.fusion_edge_thr,
    )

    # --- Iterate images ---
    img_paths = list_images(args.input_dir)
    if len(img_paths) == 0:
        raise FileNotFoundError(f'No images under: {args.input_dir}')

    csv_path = out_dir / 'scores.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 's_style', 's_struct', 's_total', 'w', 'route'])

        for p in img_paths:
            img = read_image(p, to_rgb=True, grayscale=False)

            # 1) edge hint
            raw_hint = edge_extractor(img)
            if raw_hint.ndim == 2:
                hint_rgb = as_controlnet_hint(raw_hint)
                raw_edge = raw_hint
            else:
                hint_rgb = raw_hint  # already 3-ch
                raw_edge = raw_hint[..., -1]

            # 2) generate source-like
            gen_img = unifier.reconstruct(img, hint_rgb, prompt=args.prompt, num_samples=1, seed=args.seed)[0]

            # 3) compute scores (need edges on generated)
            gen_hint = edge_extractor(gen_img)
            if gen_hint.ndim == 2:
                gen_edge = gen_hint
            else:
                gen_edge = gen_hint[..., -1]

            scores = compute_scores(
                raw_img=img,
                gen_img=gen_img,
                raw_edge=raw_edge,
                gen_edge=gen_edge,
                cfg=score_cfg,
                domain_discriminator=domain_disc,
                ref_hist=None,
                device=args.device,
            )

            s_total = scores['s_total']
            w = scores['w']

            # 4) routing + fusion
            if s_total >= score_cfg.tau_high:
                route = 'gen_or_fused'
            elif s_total <= score_cfg.tau_low:
                route = 'raw_fallback'
            else:
                route = 'fused'

            if route == 'raw_fallback':
                fused_img = img
            else:
                fused_img = fuse(img, gen_img, w=w, cfg=fusion_cfg, edge_map=raw_edge)

            # 5) segmentation
            x_raw = preprocess_for_seg(img, args.in_channels, args.resize).to(device)
            x_fused = preprocess_for_seg(fused_img, args.in_channels, args.resize).to(device)

            with torch.no_grad():
                logit_fused = seg(x_fused)
                if args.seg_ensemble:
                    logit_raw = seg(x_raw)
                    alpha = float(np.clip(s_total, 0.0, 1.0))
                    logit = alpha * logit_fused + (1.0 - alpha) * logit_raw
                else:
                    logit = logit_fused

                pred = torch.argmax(logit, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # save outputs
            stem = Path(p).stem
            write_image(str(out_dir / 'gen' / f'{stem}.png'), gen_img)
            write_image(str(out_dir / 'fused' / f'{stem}.png'), fused_img)
            # prediction mask as grayscale (class index)
            write_image(str(out_dir / 'pred' / f'{stem}.png'), pred, from_rgb=False)

            writer.writerow([p, scores['s_style'], scores['s_struct'], scores['s_total'], scores['w'], route])
            print(f"[{stem}] s_total={s_total:.3f} w={w:.3f} route={route}")

    print(f"Done. Scores saved to: {csv_path}")


if __name__ == '__main__':
    main()
