"""
Infer (generate) unified/source-like images ONLY, using a trained ControlNet unifier.

This script does NOT run segmentation. It is intended for:
- caching unified images for a whole dataset (EchoNet/CAMUS/private)
- source-domain calibration: raw vs unified distance distribution

Example:
python DomainUnifiedSegmentation/infer_unify_only.py \
  --input_dir  /path/to/EchoNet_Merged/imgs \
  --out_dir    /path/to/unify_echo \
  --stage1_root /path/to/UniDoSeg/Stage1and2 \
  --sd_ckpt     /path/to/UniDoSeg/Stage1and2/models/v1-5-pruned.ckpt \
  --controlnet_ckpt /path/to/UniDoSeg/Stage1and2/models/control_sd15_canny.pth \
  --finetuned_ckpt  /path/to/your/finetuned_controlnet.ckpt \
  --edge_method hed_finetuned \
  --hed_weights /path/to/hed_lv.pth \
  --prompt "echocardiography, left ventricle" \
  --ddim_steps 50 --guidance_scale 9.0 --control_strength 1.0 \
  --use_img2img --img2img_strength 0.6 \
  --resize 512 --device cuda \
  --save_hint --keep_structure --skip_existing
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from DomainUnifiedSegmentation.models.edge.extractor import EdgeExtractor, EdgeExtractorConfig
from DomainUnifiedSegmentation.unifier.controlnet_unifier import ControlNetUnifier, UnifierConfig
from DomainUnifiedSegmentation.utils.edge_ops import as_controlnet_hint
from DomainUnifiedSegmentation.utils.io import ensure_dir, list_images, read_image, write_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # diffusion unifier
    p.add_argument("--stage1_root", type=str, required=True)
    p.add_argument("--cldm_yaml", type=str, default="./models/cldm_v15.yaml")
    p.add_argument("--sd_ckpt", type=str, required=True)
    p.add_argument("--controlnet_ckpt", type=str, required=True)
    p.add_argument("--finetuned_ckpt", type=str, default="")

    p.add_argument("--prompt", type=str, default="Ultrasound")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=9.0)
    p.add_argument("--control_strength", type=float, default=1.0)
    p.add_argument("--guess_mode", action="store_true")
    p.add_argument("--use_img2img", action="store_true")
    p.add_argument("--img2img_strength", type=float, default=0.6)

    # edge extractor
    p.add_argument(
        "--edge_method",
        type=str,
        default="hed_controlnet",
        choices=["canny", "hed_controlnet", "hed_finetuned", "multi"],
    )
    p.add_argument("--hed_weights", type=str, default="")

    # io / runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--num_samples", type=int, default=1)

    p.add_argument("--save_hint", action="store_true")
    p.add_argument("--keep_structure", action="store_true", help="keep subfolders under out_dir")
    p.add_argument("--skip_existing", action="store_true", help="skip if gen image already exists")

    return p.parse_args()


def _hint_to_vis_uint8(hint: np.ndarray) -> np.ndarray:
    """Convert edge/hint to uint8 png for visualization."""
    if hint.ndim == 2:
        x = (np.clip(hint, 0.0, 1.0) * 255.0).astype(np.uint8)
        return x
    # HxWx3 float in [0,1]
    x = (np.clip(hint, 0.0, 1.0) * 255.0).astype(np.uint8)
    return x


def main():
    args = parse_args()
    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    gen_dir = out_root / "gen"
    hint_dir = out_root / "hint"
    ensure_dir(gen_dir)
    if args.save_hint:
        ensure_dir(hint_dir)

    # edge extractor
    edge_cfg = EdgeExtractorConfig(
        method=args.edge_method,
        hed_weights=args.hed_weights if args.hed_weights else None,
        stage1_root=args.stage1_root,
        device=args.device,
    )
    edge_extractor = EdgeExtractor(edge_cfg)

    # unifier
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

    img_paths = list_images(str(in_root))
    if not img_paths:
        raise FileNotFoundError(f"No images under: {in_root}")

    # log meta
    csv_path = out_root / "unify_index.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_path", "gen_path", "hint_path", "seed"])

        for src in img_paths:
            src_p = Path(src)
            rel = src_p.relative_to(in_root) if args.keep_structure else Path(src_p.name)

            gen_p = gen_dir / rel.with_suffix(".png")
            hint_p = hint_dir / rel.with_suffix(".png")

            ensure_dir(gen_p.parent)
            if args.save_hint:
                ensure_dir(hint_p.parent)

            if args.skip_existing and gen_p.exists():
                w.writerow([str(src_p), str(gen_p), str(hint_p) if args.save_hint else "", "SKIPPED"])
                continue

            img = read_image(str(src_p), to_rgb=True, grayscale=False)

            # 1) edge/hint
            raw_hint = edge_extractor(img)
            if raw_hint.ndim == 2:
                hint_rgb = as_controlnet_hint(raw_hint)  # HxWx3 float
                hint_vis = _hint_to_vis_uint8(raw_hint)
            else:
                hint_rgb = raw_hint.astype(np.float32)
                hint_vis = _hint_to_vis_uint8(hint_rgb)

            # 2) generate
            gen_list = unifier.reconstruct(
                img,
                hint_rgb,
                prompt=args.prompt,
                num_samples=args.num_samples,
                seed=args.seed,
            )
            gen_img = gen_list[0]

            # save
            write_image(str(gen_p), gen_img)
            if args.save_hint:
                write_image(str(hint_p), hint_vis, from_rgb=(hint_vis.ndim == 3))

            w.writerow([str(src_p), str(gen_p), str(hint_p) if args.save_hint else "", args.seed])
            print(f"[OK] {rel} -> {gen_p}")

    print(f"Done. Index saved to: {csv_path}")


if __name__ == "__main__":
    main()
