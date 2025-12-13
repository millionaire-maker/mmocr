import argparse
import json
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image


def build_art_label_to_path(art_ann: str, art_img_root: str):
    with open(art_ann, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for img_id, insts in data.items():
        for inst in insts:
            if inst.get("illegibility") or inst.get("ignore"):
                continue
            text = inst.get("transcription", "")
            if not text:
                continue
            img_path = Path(art_img_root) / f"{img_id}.jpg"
            if img_path.exists():
                mapping.setdefault(text, str(img_path))
    return mapping


def find_fudan_matches(label_to_path, lmdb_root: str, max_pairs: int = 10):
    pairs = []
    for split in ["scene_val", "scene_test"]:
        env = lmdb.open(os.path.join(lmdb_root, split), readonly=True, lock=False)
        txn = env.begin()
        total = int(txn.get(b"num-samples").decode("utf-8"))
        for idx in range(1, total + 1):
            label_bytes = txn.get(f"label-{idx:09d}".encode("utf-8"))
            if label_bytes is None:
                continue
            text = label_bytes.decode("utf-8")
            if text in label_to_path:
                img_bytes = txn.get(f"image-{idx:09d}".encode("utf-8"))
                if img_bytes is None:
                    continue
                pairs.append(
                    {
                        "label": text,
                        "split": split,
                        "idx": idx,
                        "fudan_bytes": img_bytes,
                        "art_path": label_to_path[text],
                    }
                )
                if len(pairs) >= max_pairs:
                    env.close()
                    return pairs
        env.close()
    return pairs


def save_pairs(pairs, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for i, p in enumerate(pairs, start=1):
        label = p["label"]
        safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
        fudan_name = f"{i:02d}_fudan_{p['split']}_{p['idx']:09d}_{safe_label}.png"
        art_name = f"{i:02d}_art_{Path(p['art_path']).name}"
        fudan_path = os.path.join(out_dir, fudan_name)
        art_path_out = os.path.join(out_dir, art_name)

        img = Image.open(BytesIO(p["fudan_bytes"]))
        img.save(fudan_path)

        # copy ArT image
        Image.open(p["art_path"]).save(art_path_out)

        meta.append(
            {
                "label": label,
                "fudan_split": p["split"],
                "fudan_idx": p["idx"],
                "fudan_image": fudan_name,
                "art_src": p["art_path"],
                "art_image": art_name,
            }
        )
    meta_path = os.path.join(out_dir, "pairs_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pairs)} pairs to {out_dir}, meta: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract paired images from Fudan scene LMDB and ArT task2 for manual comparison."
    )
    parser.add_argument(
        "--fudan-root",
        default="data/fudan/scene",
        help="Root directory of Fudan scene LMDBs (contains scene_val, scene_test).",
    )
    parser.add_argument(
        "--art-ann",
        default="data/ICDAR 2019 - ArT/annotations/train_task2_labels.json",
        help="ArT task2 annotation json.",
    )
    parser.add_argument(
        "--art-img-root",
        default="data/ICDAR 2019 - ArT/train_task2_images",
        help="ArT task2 cropped images root.",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp/fudan_art_pairs",
        help="Directory to save paired images.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help="Maximum number of pairs to extract.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    label_to_path = build_art_label_to_path(args.art_ann, args.art_img_root)
    print(f"Loaded {len(label_to_path)} ArT labels with existing images.")
    pairs = find_fudan_matches(label_to_path, args.fudan_root, args.max_pairs)
    print(f"Found {len(pairs)} common-label pairs between Fudan and ArT.")
    if not pairs:
        return
    save_pairs(pairs, args.out_dir)


if __name__ == "__main__":
    main()
