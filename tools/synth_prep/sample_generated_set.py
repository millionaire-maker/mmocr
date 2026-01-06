#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import html
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Item:
    rel_path: str
    label: str


def parse_line(line: str) -> Optional[Item]:
    s = line.strip("\n\r")
    if not s:
        return None
    if "\t" not in s:
        return None
    p, lab = s.split("\t", 1)
    p = p.strip()
    lab = lab.strip()
    if not p or not lab:
        return None
    return Item(rel_path=p, label=lab)


def reservoir_sample(items_iter, k: int, seed: int) -> Tuple[List[Item], int]:
    rng = random.Random(int(seed))
    res: List[Item] = []
    n = 0
    for it in items_iter:
        n += 1
        if len(res) < k:
            res.append(it)
            continue
        j = rng.randrange(n)
        if j < k:
            res[j] = it
    rng.shuffle(res)
    return res, n


def write_html(out_dir: Path, out_tag: str, items: List[Item]) -> Path:
    out_html = out_dir / "index.html"
    cards = []
    for it in items:
        img_name = Path(it.rel_path).name
        img_rel = f"{out_tag}/{img_name}"
        cards.append(
            "<div class='card'>"
            f"<img loading='lazy' src='../{html.escape(img_rel)}' />"
            f"<div class='lab'>{html.escape(it.label)}</div>"
            f"<div class='path'>{html.escape(img_name)}</div>"
            "</div>"
        )

    page = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(out_tag)} sample</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; overflow: hidden; background: #fff; }}
    img {{ width: 100%; height: 80px; object-fit: contain; background: #f7f7f7; display: block; }}
    .lab {{ padding: 8px 10px 0 10px; font-size: 14px; word-break: break-all; }}
    .path {{ padding: 2px 10px 10px 10px; color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h3>{html.escape(out_tag)}（随机样本 {len(items)}）</h3>
  <div class="grid">
    {"".join(cards)}
  </div>
</body>
</html>
"""
    out_html.write_text(page, encoding="utf-8")
    return out_html


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Randomly sample generated images and create a small viewable subset.")
    ap.add_argument("--label-txt", required=True, help="label txt (rel_path\\tlabel), e.g. data/synth_rec_ch/<tag>.txt")
    ap.add_argument("--img-root", required=True, help="root dir to resolve rel_path, e.g. data/synth_rec_ch")
    ap.add_argument("--out-dir", required=True, help="output dir to place sampled images (symlink/copy)")
    ap.add_argument("--num", type=int, default=1000, help="number of samples")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    ap.add_argument("--clean", action="store_true", help="remove out-dir and out label first")
    ap.add_argument("--no-html", action="store_true", help="do not write index.html")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    label_txt = Path(args.label_txt).expanduser().resolve()
    img_root = Path(args.img_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_tag = out_dir.name
    out_label = out_dir.parent / f"{out_tag}.txt"

    if not label_txt.is_file():
        raise SystemExit(f"--label-txt not found: {label_txt}")
    if not img_root.is_dir():
        raise SystemExit(f"--img-root not found: {img_root}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    if args.clean and out_label.exists():
        out_label.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    def iter_items():
        with open(label_txt, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                it = parse_line(line)
                if it is not None:
                    yield it

    k = int(args.num)
    if k <= 0:
        raise SystemExit("--num must be > 0")

    sampled, total = reservoir_sample(iter_items(), k=k, seed=int(args.seed))
    if not sampled:
        raise SystemExit(f"no valid items from label txt: {label_txt}")

    copied = 0
    missing = 0
    out_lines: List[str] = []
    for it in sampled:
        src = (img_root / it.rel_path).resolve()
        img_name = Path(it.rel_path).name
        dst = out_dir / img_name
        if not src.is_file():
            missing += 1
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        copied += 1

        out_rel = f"{out_tag}/{img_name}"
        out_lines.append(f"{out_rel}\t{it.label}")

    out_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    out_html = None
    if not args.no_html:
        out_html = write_html(out_dir, out_tag, sampled)

    print(f"[OK] total lines in label txt: {total}")
    print(f"[OK] sampled: {len(sampled)} (seed={int(args.seed)})")
    print(f"[OK] wrote images: {copied} ({'copy' if args.copy else 'symlink'})")
    if missing:
        print(f"[WARN] missing source images: {missing}")
    print(f"[OK] out dir: {out_dir}")
    print(f"[OK] out label: {out_label}")
    if out_html is not None:
        print(f"[OK] html: {out_html}")


if __name__ == "__main__":
    main()

