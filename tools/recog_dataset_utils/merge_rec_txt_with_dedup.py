import argparse
import json
import os
from collections import defaultdict
from hashlib import md5
from typing import Dict, Tuple

from PIL import Image


def normalize_md5(path: str, size=(128, 32)) -> str:
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return md5(img.tobytes()).hexdigest()


def load_md5_set(path: str, hash_type: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if hash_type in data:
            return set(data[hash_type])
        if "md5" in data:
            return set(data["md5"])
        return set(data.keys())
    if isinstance(data, list):
        return set(data)
    raise ValueError("Unsupported md5 set format.")


def compute_md5(path: str, hash_type: str) -> str:
    if hash_type == "norm":
        return normalize_md5(path)
    hash_md5 = md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def resolve_image_path(img_path: str, label_file: str, repo_root: str) -> Tuple[str, str]:
    if not os.path.isabs(img_path):
        abs_path = os.path.normpath(os.path.join(os.path.dirname(label_file), img_path))
    else:
        abs_path = img_path
    rel_path = os.path.relpath(abs_path, repo_root) if repo_root else img_path
    return abs_path, rel_path


def merge_files(inputs, fudan_set_path, out_txt, repo_root, stats_path):
    fudan_md5 = load_md5_set(fudan_set_path, hash_type=args.hash_type)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    kept = 0
    with open(out_txt, "w", encoding="utf-8") as writer:
        for label_file in inputs:
            dataset_name = os.path.splitext(os.path.basename(label_file))[0]
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) != 2:
                        continue
                    img_path_raw, label = parts
                    abs_path, rel_out = resolve_image_path(img_path_raw, label_file, repo_root)
                    stats[dataset_name]["total"] += 1
                    if not os.path.exists(abs_path):
                        stats[dataset_name]["missing"] += 1
                        continue
                    img_md5 = compute_md5(abs_path, args.hash_type)
                    if img_md5 in fudan_md5:
                        stats[dataset_name]["fudan_dup"] += 1
                        continue
                    writer.write(f"{rel_out}\t{label}\n")
                    stats[dataset_name]["kept"] += 1
                    kept += 1
    summary = {
        "per_dataset": stats,
        "total_kept": kept,
        "inputs": inputs,
        "fudan_set": fudan_set_path,
    }
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(inputs)} files. Kept={kept}. Stats saved to {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple recog label txt files with dedup against Fudan val/test MD5."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input label txt files (img_path<TAB>label).",
    )
    parser.add_argument(
        "--fudan-md5-set",
        required=True,
        help="Path to Fudan val/test md5 set json.",
    )
    parser.add_argument(
        "--hash-type",
        choices=["raw", "norm"],
        default="norm",
        help="Hash type to compare with Fudan set (raw image md5 or normalized md5).",
    )
    parser.add_argument(
        "--out-txt",
        required=True,
        help="Output merged txt path.",
    )
    parser.add_argument(
        "--repo-root",
        default="/home/yzy/mmocr",
        help="Root path to make image paths relative to.",
    )
    parser.add_argument(
        "--stats",
        default="/home/yzy/mmocr/data/recog_mega/merge_stats.json",
        help="Path to save merge statistics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    merge_files(
        inputs=args.inputs,
        fudan_set_path=args.fudan_md5_set,
        out_txt=args.out_txt,
        repo_root=args.repo_root,
        stats_path=args.stats,
    )


if __name__ == "__main__":
    main()
