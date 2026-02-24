import argparse
import json
import os
from pathlib import Path

from utils.io import ensure_dir, list_images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--out_json', type=str, required=True)
    p.add_argument('--hints_dir', type=str, default='')
    p.add_argument('--prompt', type=str, default='Ultrasound')
    p.add_argument('--strict_hints', action='store_true',
                   help='If set, require every image to have a matching hint; otherwise raise error.')
    return p.parse_args()


def main():
    args = parse_args()
    img_paths = list_images(args.images_dir)
    if len(img_paths) == 0:
        raise FileNotFoundError(f'No images under: {args.images_dir}')

    items = []
    missing = []

    for p in img_paths:
        item = {'source': os.path.abspath(p), 'prompt': args.prompt}

        if args.hints_dir:
            rel = os.path.relpath(p, args.images_dir)
            hint_path = os.path.join(args.hints_dir, rel)
            if os.path.exists(hint_path):
                item['hint'] = os.path.abspath(hint_path)
            else:
                missing.append((p, hint_path))

        items.append(item)

    if args.hints_dir and args.strict_hints and len(missing) > 0:
        msg = "\n".join([f"IMG: {a}\nHINT_MISSING: {b}" for a, b in missing[:20]])
        raise FileNotFoundError(
            f"[strict_hints] Missing {len(missing)} hint files. Showing first 20:\n{msg}\n"
            f"Check that hints_dir mirrors images_dir filenames/structure."
        )

    out_path = Path(args.out_json)
    ensure_dir(out_path.parent)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f'Wrote {len(items)} items to: {out_path}')
    if args.hints_dir:
        print(f'Hints_dir: {args.hints_dir}')
        print(f'With hint: {sum(1 for x in items if "hint" in x)} / {len(items)}')


if __name__ == '__main__':
    main()



