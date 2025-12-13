import argparse
import json
import os
import unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np


def normalize_label(text: str) -> str:
    """Normalize labels to keep them consistent."""
    if text is None:
        return ""
    text = text.replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_polygon(image: np.ndarray, polygon: List[float]) -> Optional[np.ndarray]:
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = order_points(box)
    w, h = rect[1]
    w = int(max(round(w), 1))
    h = int(max(round(h), 1))
    if w <= 1 or h <= 1:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    try:
        warped = cv2.warpPerspective(image, M, (w, h))
    except cv2.error:
        warped = None
    return warped


def rects_iter(root: str, split: str) -> Generator[Dict, None, None]:
    ann_dir = os.path.join(root, "annotations")
    img_dir = os.path.join(root, "imgs")
    for ann_file in sorted(os.listdir(ann_dir)):
        if not ann_file.endswith(".json"):
            continue
        img_name = os.path.splitext(ann_file)[0] + ".jpg"
        img_path = os.path.join(img_dir, img_name)
        with open(os.path.join(ann_dir, ann_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        for line in data.get("lines", []):
            if line.get("ignore", 0):
                continue
            pts = line.get("points", [])
            if len(pts) < 8:
                continue
            text = line.get("transcription", "")
            yield {"img_path": img_path, "polygon": pts, "text": text}


def rctw_iter(root: str, split: str) -> Generator[Dict, None, None]:
    import csv

    ann_dir = os.path.join(root, "annotations")
    img_dir = os.path.join(root, "imgs")
    for ann_file in sorted(os.listdir(ann_dir)):
        if not ann_file.endswith(".txt"):
            continue
        img_name = os.path.splitext(ann_file)[0] + ".jpg"
        img_path = os.path.join(img_dir, img_name)
        with open(os.path.join(ann_dir, ann_file), "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 9:
                    continue
                try:
                    coords = list(map(float, row[:8]))
                except ValueError:
                    continue
                label = row[-1].strip('"')
                flag = row[8].strip() if len(row) > 8 else "0"
                if flag != "0" or label in {"###", "*"}:
                    continue
                yield {"img_path": img_path, "polygon": coords, "text": label}


def load_art_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def art_iter(root: str, split: str) -> Generator[Dict, None, None]:
    ann_path = os.path.join(root, "annotations", "train_labels.json")
    data = load_art_json(ann_path)
    img_dir = os.path.join(root, "train_images")
    for img_id, instances in data.items():
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        for inst in instances:
            if inst.get("illegibility", False) or inst.get("ignore", False):
                continue
            points = inst.get("points", [])
            if not points:
                continue
            # flatten list of [x, y]
            flat = [c for pt in points for c in pt]
            text = inst.get("transcription", "")
            yield {"img_path": img_path, "polygon": flat, "text": text}


def art_task2_iter(root: str, split: str) -> Generator[Dict, None, None]:
    """ArT task2: 已裁剪行图，直接读取图片与标签。"""
    ann_path = os.path.join(root, "annotations", "train_task2_labels.json")
    img_dir = os.path.join(root, "train_task2_images")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ArT task2 annotation not found: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for img_id, instances in data.items():
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        for inst in instances:
            if inst.get("illegibility") or inst.get("ignore"):
                continue
            text = inst.get("transcription", "")
            polygon = [0, 0, 1, 0, 1, 1, 0, 1]  # 占位，避免透视变换报错
            yield {"img_path": img_path, "polygon": polygon, "text": text}


def build_lsvt_lookup(root: str) -> Dict[str, str]:
    lookup = {}
    for sub in ["train_full_images_0", "train_full_images_1"]:
        img_dir = os.path.join(root, sub)
        if not os.path.exists(img_dir):
            continue
        for fn in os.listdir(img_dir):
            if fn.lower().endswith(".jpg"):
                lookup[os.path.splitext(fn)[0]] = os.path.join(img_dir, fn)
    return lookup


def lsvt_iter(root: str, split: str) -> Generator[Dict, None, None]:
    ann_path = os.path.join(root, "annotations", "train_full_labels.json")
    lookup = build_lsvt_lookup(root)
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for img_id, instances in data.items():
        img_path = lookup.get(img_id)
        if not img_path:
            continue
        for inst in instances:
            if inst.get("illegibility", False) or inst.get("ignore", False):
                continue
            points = inst.get("points", [])
            if not points:
                continue
            flat = [c for pt in points for c in pt]
            text = inst.get("transcription", "")
            yield {"img_path": img_path, "polygon": flat, "text": text}


def ctw_iter(root: str, split: str) -> Generator[Dict, None, None]:
    ann_dir = os.path.join(root, "原始数据", "ctw1500_train_labels")
    img_dir = os.path.join(root, "train", "text_image")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"CTW1500 annotation dir not found: {ann_dir}")
    for ann_file in sorted(os.listdir(ann_dir)):
        if not ann_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(ann_dir, ann_file))
        root_node = tree.getroot()
        img_name = ann_file.replace(".xml", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        for box in root_node.iter("box"):
            label_node = box.find("label")
            segs_node = box.find("segs")
            if label_node is None or segs_node is None:
                continue
            text = label_node.text or ""
            coords = []
            for v in segs_node.text.split(","):
                v = v.strip()
                if not v:
                    continue
                try:
                    coords.append(float(v))
                except ValueError:
                    continue
            if len(coords) < 8:
                continue
            yield {"img_path": img_path, "polygon": coords, "text": text}


DATASET_LOADERS = {
    "rects": rects_iter,
    "rctw17": rctw_iter,
    "art": art_iter,
    "art_task2": art_task2_iter,
    "lsvt": lsvt_iter,
    "ctw1500": ctw_iter,
}


def process_dataset(
    dataset: str, root: str, split: str, out_root: str, max_samples: int, dry_run: bool
):
    if dataset not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset {dataset}")
    loader = DATASET_LOADERS[dataset]
    os.makedirs(out_root, exist_ok=True)
    images_dir = os.path.join(out_root, "images")
    total = 0
    saved = 0
    label_path = os.path.join(out_root, f"labels_{split}.txt")
    writer = None
    if not dry_run:
        os.makedirs(images_dir, exist_ok=True)
        writer = open(label_path, "w", encoding="utf-8")
    for sample in loader(root, split):
        total += 1
        if max_samples and saved >= max_samples:
            break
        label = normalize_label(sample.get("text", ""))
        if not label or label in {"###", "*"}:
            continue
        img = cv2.imread(sample["img_path"])
        if img is None:
            continue
        if dataset == "art_task2":
            crop = img  # 已裁剪，直接使用整图
        else:
            crop = warp_polygon(img, sample["polygon"])
            if crop is None:
                continue
        saved += 1
        if writer:
            fname = f"{dataset}_{split}_{saved:08d}.jpg"
            out_img_path = os.path.join(images_dir, fname)
            cv2.imwrite(out_img_path, crop)
            writer.write(f"images/{fname}\t{label}\n")
    if writer:
        writer.close()
    msg = (
        f"{dataset} split={split}: total instances={total}, saved={saved}"
        + (f", labels file={label_path}" if not dry_run else " (dry-run, not saved)")
    )
    print(msg)
    return {"dataset": dataset, "split": split, "total": total, "saved": saved}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop detection polygons to recognition line images with rectification."
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--root", required=True, help="Root path of the dataset.")
    parser.add_argument("--out-root", required=True, help="Output root directory.")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of cropped instances for quick sanity check. 0 means all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not save images/labels, only count usable instances.",
    )
    parser.add_argument(
        "--stats-json",
        default="",
        help="Optional path to save stats json (total/saved). Empty to skip.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else 0
    stats = process_dataset(
        args.dataset, args.root, args.split, args.out_root, max_samples, args.dry_run
    )
    if args.stats_json:
        import json
        os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Saved stats to {args.stats_json}")


if __name__ == "__main__":
    main()
