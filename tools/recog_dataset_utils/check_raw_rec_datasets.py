import argparse
import json
import os
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANN_EXTS = {".json", ".txt", ".xml", ".csv", ".tsv"}


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def scan_dataset(root):
    stats = defaultdict(int)
    ann_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            full = os.path.join(dirpath, fn)
            if ext in IMG_EXTS:
                stats["num_images"] += 1
            if ext in ANN_EXTS:
                stats["num_annotations"] += 1
                if len(ann_paths) < 10:
                    ann_paths.append(full)
    stats["sample_annotations"] = ann_paths
    return stats


def main():
    parser = argparse.ArgumentParser(description="Check raw detection datasets.")
    parser.add_argument(
        "--output",
        default="data/recog_raw/recog_raw_stats.json",
        help="Output json for dataset stats.",
    )
    args = parser.parse_args()

    datasets = [
        ("art", ["data/ICDAR 2019 - ArT"]),
        ("rctw17", ["data/RCTW-17"]),
        ("rects", ["data/rects"]),
        ("lsvt", ["data/ICDAR 2019 - LSVT"]),
        ("ctw1500", ["data/CTW1500", "data/ctw1500", "data/CTW"]),
    ]

    results = []
    for name, candidates in datasets:
        resolved = first_existing(candidates)
        entry = {"name": name, "candidates": candidates, "exists": bool(resolved)}
        if not resolved:
            print(f"[WARN] {name} missing. Tried: {candidates}")
            results.append(entry)
            continue
        entry["root"] = resolved
        stats = scan_dataset(resolved)
        entry.update(stats)
        print(
            f"[OK] {name}: root={resolved}, images={stats['num_images']}, "
            f"annotations={stats['num_annotations']}"
        )
        if stats["sample_annotations"]:
            print(f"  sample annotations: {stats['sample_annotations'][:3]}")
        results.append(entry)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved stats to {args.output}")


if __name__ == "__main__":
    main()
