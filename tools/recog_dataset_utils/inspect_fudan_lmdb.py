import argparse
import json
import os
from io import BytesIO

import lmdb
from PIL import Image


def find_scene_lmdbs(root: str):
    """Recursively find lmdb directories that contain 'scene'."""
    for dirpath, dirnames, filenames in os.walk(root):
        if "data.mdb" in filenames:
            name = os.path.basename(dirpath)
            if "scene" in name.lower() or "scene" in dirpath.lower():
                yield dirpath
        # Do not descend into hidden dirs to keep traversal fast
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]


def inspect_lmdb(path: str, preview: int = 5):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False)
    info = {"name": os.path.basename(path.rstrip("/")), "path": path}
    with env.begin() as txn:
        num_samples = txn.get(b"num-samples")
        num = int(num_samples.decode("utf-8")) if num_samples else 0
        info["num_samples"] = num
        print(f"[LMDB] {info['name']} ({path}) num_samples={num}")
        upper = min(preview, num)
        for idx in range(1, upper + 1):
            label_key = f"label-{idx:09d}".encode("utf-8")
            image_key = f"image-{idx:09d}".encode("utf-8")
            label_bytes = txn.get(label_key)
            img_bytes = txn.get(image_key)
            if label_bytes is None or img_bytes is None:
                print(f"  #{idx}: missing keys")
                continue
            label = label_bytes.decode("utf-8", errors="replace")
            try:
                with Image.open(BytesIO(img_bytes)) as img:
                    width, height = img.size
            except Exception as e:  # pragma: no cover - preview only
                width, height = -1, -1
                print(f"  #{idx}: failed to decode image ({e})")
            key_str = f"image-{idx:09d}"
            print(f"  #{idx}: key={key_str}, size=({width},{height}), label={label}")
    env.close()
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Fudan scene LMDBs and dump stats."
    )
    parser.add_argument(
        "--root",
        default="/home/yzy/mmocr/data/fudan/scene",
        help="Root directory to search for scene LMDBs.",
    )
    parser.add_argument(
        "--output",
        default="/home/yzy/mmocr/data/fudan_benchmark/fudan_scene_lmdb_info.json",
        help="Path to save LMDB info JSON.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of samples to preview for each LMDB.",
    )
    args = parser.parse_args()

    lmdb_paths = sorted(set(find_scene_lmdbs(args.root)))
    if not lmdb_paths:
        print(f"No scene LMDBs found under {args.root}")
        return

    infos = []
    for path in lmdb_paths:
        try:
            infos.append(inspect_lmdb(path, preview=args.preview))
        except Exception as e:  # pragma: no cover - runtime safety
            print(f"Failed to inspect {path}: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)
    print(f"Saved LMDB info to {args.output}")


if __name__ == "__main__":
    main()
