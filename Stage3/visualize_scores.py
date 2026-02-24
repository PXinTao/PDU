import os
import argparse
import math
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # try to detect columns
        # expected: idx,name,raw_path,gen_path,prompt,cos,angle,l2  (order may vary but usually fixed)
        # We'll map by header name if possible, else fallback by fixed indices.
        if any(h in header for h in ["cos", "angle", "l2", "raw_path", "gen_path"]):
            idx_map = {h: i for i, h in enumerate(header)}
            for r in reader:
                rows.append({
                    "idx": r[idx_map.get("idx", 0)],
                    "name": r[idx_map.get("name", 1)] if "name" in idx_map else r[1],
                    "raw_path": r[idx_map.get("raw_path", 2)],
                    "gen_path": r[idx_map.get("gen_path", 3)],
                    "prompt": r[idx_map.get("prompt", 4)] if "prompt" in idx_map else "",
                    "cos": float(r[idx_map.get("cos", 5)]),
                    "angle": float(r[idx_map.get("angle", 6)]),
                    "l2": float(r[idx_map.get("l2", 7)]),
                })
        else:
            # fallback fixed
            for r in reader:
                rows.append({
                    "idx": r[0],
                    "name": r[1],
                    "raw_path": r[2],
                    "gen_path": r[3],
                    "prompt": r[4] if len(r) > 4 else "",
                    "cos": float(r[5]),
                    "angle": float(r[6]),
                    "l2": float(r[7]),
                })
    return rows, header


def quantiles(x, qs=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)):
    x = np.asarray(x, dtype=np.float32)
    return {q: float(np.quantile(x, q)) for q in qs}


def try_load_font(size=16):
    # headless server often doesn't have fonts; we fallback to default
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def load_rgb(path: str):
    img = Image.open(path).convert("RGB")
    return img


def resize_keep(img: Image.Image, target_h: int):
    w, h = img.size
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / float(h))))
    return img.resize((new_w, target_h), resample=Image.BILINEAR)


def make_pair_panel(raw_path: str, gen_path: str, title: str, out_path: Path, target_h=256, pad=8):
    raw = load_rgb(raw_path)
    gen = load_rgb(gen_path)
    raw = resize_keep(raw, target_h)
    gen = resize_keep(gen, target_h)

    font = try_load_font(16)

    # compute canvas
    w = raw.size[0] + gen.size[0] + pad * 3
    h = target_h + pad * 3 + 22  # title bar
    canvas = Image.new("RGB", (w, h), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)

    # title
    draw.text((pad, pad), title, fill=(230, 230, 230), font=font)

    y0 = pad * 2 + 22
    canvas.paste(raw, (pad, y0))
    canvas.paste(gen, (pad * 2 + raw.size[0], y0))

    # labels
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str, help="score_byol_pairs output csv")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--topk", type=int, default=60, help="how many best/worst to visualize")
    ap.add_argument("--thumb_h", type=int, default=256, help="panel image height")
    ap.add_argument("--mosaic_cols", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    worst_dir = out_dir / "worst_cos"
    best_dir = out_dir / "best_cos"
    ensure_dir(worst_dir)
    ensure_dir(best_dir)

    rows, header = read_rows(args.csv)
    if len(rows) == 0:
        raise RuntimeError(f"Empty csv: {args.csv}")

    cos_list = [r["cos"] for r in rows]
    l2_list = [r["l2"] for r in rows]
    angle_list = [r["angle"] for r in rows]

    print("==== Stats ====")
    print(f"CSV: {args.csv}")
    print(f"N  : {len(rows)}")
    print("cos   mean/std:", float(np.mean(cos_list)), float(np.std(cos_list)))
    print("l2    mean/std:", float(np.mean(l2_list)), float(np.std(l2_list)))
    print("angle mean/std:", float(np.mean(angle_list)), float(np.std(angle_list)))
    print("cos quantiles:", quantiles(cos_list))
    print("l2  quantiles:", quantiles(l2_list))
    print("angle quantiles:", quantiles(angle_list))

    # sort
    rows_sorted = sorted(rows, key=lambda r: r["cos"])
    worst = rows_sorted[: args.topk]
    best = rows_sorted[-args.topk:][::-1]

    # write lists
    with open(out_dir / "worst_list.txt", "w", encoding="utf-8") as f:
        for r in worst:
            f.write(f'{r["idx"]},{r["name"]},{r["cos"]:.6f},{r["angle"]:.6f},{r["l2"]:.6f},{r["raw_path"]},{r["gen_path"]}\n')
    with open(out_dir / "best_list.txt", "w", encoding="utf-8") as f:
        for r in best:
            f.write(f'{r["idx"]},{r["name"]},{r["cos"]:.6f},{r["angle"]:.6f},{r["l2"]:.6f},{r["raw_path"]},{r["gen_path"]}\n')

    # render panels
    worst_panels = []
    for i, r in enumerate(worst):
        title = f'WORST cos={r["cos"]:.3f}  angle={r["angle"]:.3f}  l2={r["l2"]:.3f}  {r["name"]}'
        out_p = worst_dir / f'{i:04d}_{r["name"]}.png'
        try:
            make_pair_panel(r["raw_path"], r["gen_path"], title, out_p, target_h=args.thumb_h)
            worst_panels.append(str(out_p))
        except Exception as e:
            print("[skip worst]", r["name"], e)

    best_panels = []
    for i, r in enumerate(best):
        title = f'BEST  cos={r["cos"]:.3f}  angle={r["angle"]:.3f}  l2={r["l2"]:.3f}  {r["name"]}'
        out_p = best_dir / f'{i:04d}_{r["name"]}.png'
        try:
            make_pair_panel(r["raw_path"], r["gen_path"], title, out_p, target_h=args.thumb_h)
            best_panels.append(str(out_p))
        except Exception as e:
            print("[skip best]", r["name"], e)

    # mosaics
    make_mosaic(worst_panels, out_dir / "mosaic_worst_cos.png", cols=args.mosaic_cols)
    make_mosaic(best_panels, out_dir / "mosaic_best_cos.png", cols=args.mosaic_cols)

    print("==== Saved ====")
    print("Worst panels:", worst_dir)
    print("Best panels :", best_dir)
    print("Mosaic worst:", out_dir / "mosaic_worst_cos.png")
    print("Mosaic best :", out_dir / "mosaic_best_cos.png")
    print("Lists       :", out_dir / "worst_list.txt", out_dir / "best_list.txt")


if __name__ == "__main__":
    main()
