#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import lmdb


@dataclass
class Item:
    rel_path: str
    label: str


def parse_label_line(line: str) -> Optional[Item]:
    s = line.strip("\n\r")
    if not s:
        return None

    # Prefer tab-separated "<path>\t<label>"
    if "\t" in s:
        p, lab = s.split("\t", 1)
        p = p.strip()
        lab = lab.strip()
        if not p or not lab:
            return None
        return Item(rel_path=p, label=lab)

    # Fallback: "<id> <label>" (original synth_chinese_ocr labels.txt style)
    if " " in s:
        p, lab = s.split(" ", 1)
        p = p.strip()
        lab = lab.strip()
        if not p or not lab:
            return None
        # If p is an id like 00000001, assume jpg under current tag directory is handled by caller;
        # keep as-is and let caller resolve.
        return Item(rel_path=p, label=lab)

    return None


def load_items(label_txt: Path, sort_by_path: bool) -> List[Item]:
    items: List[Item] = []
    for line in label_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        it = parse_label_line(line)
        if it is None:
            continue
        items.append(it)
    if sort_by_path:
        items.sort(key=lambda x: x.rel_path)
    return items


def resolve_image_path(img_root: Path, item: Item, default_ext: str) -> Path:
    # If rel_path already looks like a path with extension, use it.
    p = Path(item.rel_path)
    if p.suffix:
        return (img_root / p).resolve()

    # If rel_path is numeric id, map to <id>.<ext>
    if re.fullmatch(r"\d+", item.rel_path):
        return (img_root / f"{item.rel_path}.{default_ext.lstrip('.')}").resolve()
    if re.fullmatch(r"\d{8}", item.rel_path):
        return (img_root / f"{item.rel_path}.{default_ext.lstrip('.')}").resolve()

    # Otherwise treat it as relative path.
    return (img_root / item.rel_path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert generated dataset (images + label txt) to LMDB format.")
    parser.add_argument("--label-txt", required=True, help="label txt path (tab: rel_path\\tlabel)")
    parser.add_argument(
        "--img-root",
        required=True,
        help="root directory to resolve rel_path, e.g. data/synth_rec_ch",
    )
    parser.add_argument("--lmdb-dir", required=True, help="output lmdb directory")
    parser.add_argument("--start-index", type=int, default=1, help="lmdb key index start (default 1)")
    parser.add_argument("--default-ext", default="jpg", help="used when label file uses numeric id only")
    parser.add_argument("--commit-interval", type=int, default=2000, help="commit every N samples")
    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=64.0,
        help="LMDB map size in GB (upper bound, not real disk usage).",
    )
    parser.add_argument("--no-sort", action="store_true", help="do not sort by rel_path; keep label file order")
    parser.add_argument("--overwrite", action="store_true", help="delete existing lmdb-dir if exists")
    parser.add_argument("--write-path-key", action="store_true", help="also write path-%09d key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_txt = Path(args.label_txt).expanduser().resolve()
    img_root = Path(args.img_root).expanduser().resolve()
    lmdb_dir = Path(args.lmdb_dir).expanduser().resolve()

    if not label_txt.is_file():
        raise SystemExit(f"--label-txt not found: {label_txt}")
    if not img_root.is_dir():
        raise SystemExit(f"--img-root not found: {img_root}")

    if lmdb_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"--lmdb-dir exists (use --overwrite): {lmdb_dir}")
        shutil.rmtree(lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(label_txt, sort_by_path=(not args.no_sort))
    if not items:
        raise SystemExit(f"no valid items loaded from: {label_txt}")

    map_size = int(float(args.map_size_gb) * 1024 * 1024 * 1024)
    env = lmdb.open(str(lmdb_dir), map_size=map_size)

    start = int(args.start_index)
    commit_interval = max(1, int(args.commit_interval))

    n_written = 0
    txn = env.begin(write=True)
    try:
        for i, item in enumerate(items):
            idx = start + i
            img_path = resolve_image_path(img_root, item, default_ext=str(args.default_ext))
            if not img_path.is_file():
                raise FileNotFoundError(f"image not found: {img_path} (from {item.rel_path})")

            img_bytes = img_path.read_bytes()
            key_img = f"image-{idx:09d}".encode("utf-8")
            key_lab = f"label-{idx:09d}".encode("utf-8")
            txn.put(key_img, img_bytes)
            txn.put(key_lab, item.label.encode("utf-8"))
            if args.write_path_key:
                key_p = f"path-{idx:09d}".encode("utf-8")
                txn.put(key_p, item.rel_path.encode("utf-8"))

            n_written += 1
            if n_written % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.put(b"num-samples", str(n_written).encode("utf-8"))
        txn.commit()
    finally:
        try:
            txn.abort()
        except Exception:
            pass

    env.close()
    print(f"[OK] wrote lmdb: {lmdb_dir}")
    print(f"[OK] num-samples: {n_written}")


if __name__ == "__main__":
    main()
