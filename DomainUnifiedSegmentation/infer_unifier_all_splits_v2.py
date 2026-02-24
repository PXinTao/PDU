# DomainUnifiedSegmentation/infer_unifier_all_splits_v2.py

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from DomainUnifiedSegmentation.datasets.controlnet_dataset import ControlNetDataset
from DomainUnifiedSegmentation.utils.io import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_root", type=str, required=True)
    p.add_argument("--sd_ckpt", type=str, required=True)
    p.add_argument("--controlnet_ckpt", type=str, required=True)
    p.add_argument("--finetuned_ckpt", type=str, required=True)

    p.add_argument("--echonet_root", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--ddim_steps", type=int, default=31)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--max_images", type=int, default=-1)

    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _tensor_to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) in [-1,1] or [0,1] depending on what we convert from.
    We assume this is the 'samples' output from Stage1and2 log_images,
    usually in [-1,1]. We'll clamp and map to uint8.
    """
    x = x.detach().float().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) * 0.5
    x = torch.clamp(x, 0.0, 1.0)
    x = (x * 255.0).round().byte()
    return x.permute(1, 2, 0).numpy()  # HWC


@torch.no_grad()
def run_one_split(
    split: str,
    json_path: str,
    args,
    model,
):
    split_out = Path(args.out_root) / split
    gen_dir = split_out / "gen"
    ensure_dir(gen_dir)
    ensure_dir(split_out)

    ds = ControlNetDataset(
        json_path=json_path,
        resolution=args.resolution,
        # Inference: we want strict alignment
        return_paths=True,
        strict_hints=True,
        drop_prompt_prob=0.0,
        drop_hint_prob=0.0,
    )

    if args.max_images and args.max_images > 0:
        ds.items = ds.items[: args.max_images]

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    meta_path = split_out / "meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["name", "source_path", "gen_path", "prompt"])
        writer.writeheader()

        for bi, batch in enumerate(dl):
            # Stage1and2 expects numpy arrays in batch; but your ds already returns numpy for jpg/hint, txt str.
            # Depending on Stage1and2 code, batch may come as:
            # - jpg: numpy arrays or torch tensors
            # We'll pass through the batch to model.log_images as is.

            images: Dict[str, Any] = model.log_images(
                batch,
                split="val",
                ddim_steps=args.ddim_steps,
                unconditional_guidance_scale=args.cfg_scale,
            )

            # Prefer samples if present
            if "samples" not in images:
                # Some forks use "samples_cfg_scale_?.??" keys; fall back to any key startswith "samples"
                sample_keys = [k for k in images.keys() if str(k).startswith("samples")]
                if len(sample_keys) == 0:
                    raise KeyError(f"No samples found in log_images outputs. Keys={list(images.keys())[:10]}")
                samples = images[sample_keys[0]]
            else:
                samples = images["samples"]

            # samples: (B,3,H,W)
            if not torch.is_tensor(samples):
                samples = torch.from_numpy(samples)

            B = samples.shape[0]

            # Extract per-sample name/path/prompt
            names = batch.get("name", None)
            paths = batch.get("path", None)
            prompts = batch.get("txt", None)

            names_l = _as_list(names) if names is not None else [None] * B
            paths_l = _as_list(paths) if paths is not None else [None] * B
            prompts_l = _as_list(prompts) if prompts is not None else [""] * B

            for i in range(B):
                # Determine filename stem
                name = None
                if names_l[i] is not None:
                    name = str(names_l[i])
                elif paths_l[i] is not None:
                    name = Path(str(paths_l[i])).stem
                else:
                    name = f"{split}_{bi:06d}_{i:02d}"

                out_fn = f"{name}.png"
                out_path = gen_dir / out_fn

                img_u8 = _tensor_to_uint8_img(samples[i])
                Image.fromarray(img_u8).save(out_path)

                writer.writerow(
                    dict(
                        name=name,
                        source_path=str(paths_l[i]) if paths_l[i] is not None else "",
                        gen_path=str(out_path),
                        prompt=str(prompts_l[i]) if prompts_l[i] is not None else "",
                    )
                )

    print(f"[{split}] finished -> {split_out}")
    return str(split_out)


def main():
    args = parse_args()
    device = torch.device(args.device)

    stage_root = Path(args.stage1_root).resolve()
    if not stage_root.exists():
        raise FileNotFoundError(f"stage1_root not found: {stage_root}")

    # Import Stage1and2 modules
    import sys
    sys.path.insert(0, str(stage_root))

    from cldm.model import create_model, load_state_dict

    # Create model
    model = create_model(str(stage_root / "models" / "cldm_v15.yaml")).cpu()

    # Load SD base
    sd_states = load_state_dict(str(args.sd_ckpt), location="cpu")
    model.load_state_dict(sd_states, strict=False)

    # Load ControlNet backbone
    cn_states = load_state_dict(str(args.controlnet_ckpt), location="cpu")
    model.load_state_dict(cn_states, strict=False)

    # Load finetuned ControlNet (Lightning checkpoint)
    ckpt = torch.load(args.finetuned_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Lightning checkpoints often prefix with "model." or similar; strip common prefixes
    cleaned = {}
    for k, v in ckpt.items():
        nk = k
        for pref in ["model.", "net.", "module."]:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)

    model.to(device)
    model.eval()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        json_path = Path(args.echonet_root) / split / f"controlnet_{split}.json"
        # If you are using your own naming (e.g., controlnet_train.json), fall back:
        if not json_path.exists():
            # common fallback
            alt = Path(args.echonet_root) / split / f"controlnet_{split}.json"
            if alt.exists():
                json_path = alt
            else:
                # allow user to have controlnet_train.json inside split folder
                alt2 = Path(args.echonet_root) / split / f"controlnet_{split}.json"
                json_path = alt2

        # Most people have: EchoNet_Merged/train/controlnet_train.json, etc.
        # So we also try:
        if not json_path.exists():
            json_path2 = Path(args.echonet_root) / split / f"controlnet_{split}.json"
            if json_path2.exists():
                json_path = json_path2
        if not json_path.exists():
            # final: EchoNet_Merged/{split}/controlnet_{split}.json not found -> try controlnet_{split}.json or controlnet_{split}.json
            json_path3 = Path(args.echonet_root) / split / f"controlnet_{split}.json"
            json_path = json_path3

        # Practical fallback you likely use:
        # EchoNet_Merged/train/controlnet_train.json, EchoNet_Merged/val/controlnet_val.json, EchoNet_Merged/test/controlnet_test.json
        if not json_path.exists():
            json_path = Path(args.echonet_root) / split / f"controlnet_{split}.json"
        if not json_path.exists():
            json_path = Path(args.echonet_root) / split / f"controlnet_{split}.json"

        # Better: use exactly these conventional names:
        if not json_path.exists():
            json_path = Path(args.echonet_root) / split / f"controlnet_{split}.json"

        # If still not exist, try common: controlnet_train.json etc.
        if not json_path.exists():
            json_path = Path(args.echonet_root) / split / f"controlnet_{split}.json"

        # Finally, your known path pattern:
        # /home/data/pxt/USSeg/EchoNet_Merged/train/controlnet_train.json etc.
        if not json_path.exists():
            json_path = Path(args.echonet_root) / split / f"controlnet_{split}.json"

        # If you actually have these names:
        if not json_path.exists():
            json_path = Path(args.echonet_root) / split / f"controlnet_{split}.json"

        # OK, now be strict (you should set these correctly in your script if names differ)
        if not json_path.exists():
            # last attempt: controlnet_{split}.json doesn't exist; try controlnet_{split}.json? (keep strict)
            raise FileNotFoundError(f"Cannot find json for split={split}. Expected under: {Path(args.echonet_root)/split}")

        run_one_split(split=split, json_path=str(json_path), args=args, model=model)


if __name__ == "__main__":
    main()
