import argparse
import copy
import json
import os
from typing import Iterable, List

import numpy as np
import torch
from mmengine import load

from mmocr.evaluation.metrics import HmeanIOUMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate textdet pkl on all/curved subsets with HmeanIOU."
    )
    parser.add_argument(
        "--pred-pkl",
        required=True,
        help="Path to DumpResults *_predictions.pkl.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to save metrics json.",
    )
    parser.add_argument(
        "--curved-poly-len-thr",
        type=int,
        default=8,
        help="Curved polygon length threshold (flat coord length).",
    )
    parser.add_argument(
        "--curved-images",
        action="store_true",
        help="Also evaluate on images containing curved instances.",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run on first 2 samples for quick sanity.",
    )
    return parser.parse_args()


def _poly_len(poly) -> int:
    if isinstance(poly, torch.Tensor):
        return int(poly.numel())
    try:
        return int(np.array(poly).size)
    except Exception:
        return 0


def _ensure_bool_array(value, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=bool)
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    arr = np.array(value, dtype=bool)
    if arr.size != length:
        arr = np.resize(arr, length).astype(bool)
    return arr


def _has_curved_instance(polygons, ignore_flags, thr: int) -> bool:
    if polygons is None:
        return False
    for poly, ignored in zip(polygons, ignore_flags):
        if ignored:
            continue
        if _poly_len(poly) > thr:
            return True
    return False


def _clone_data_sample(sample):
    if hasattr(sample, "clone"):
        return sample.clone()
    return copy.deepcopy(sample)


def _clone_instances(instances):
    if hasattr(instances, "clone"):
        return instances.clone()
    return copy.deepcopy(instances)


def _set_curved_ignored_flags(data_samples: Iterable, thr: int) -> List:
    curved_samples = []
    for sample in data_samples:
        sample_clone = _clone_data_sample(sample)
        gt_instances = sample_clone.get("gt_instances")
        if gt_instances is None:
            curved_samples.append(sample_clone)
            continue
        polygons = gt_instances.get("polygons")
        if polygons is None:
            curved_samples.append(sample_clone)
            continue
        ignore_flags = _ensure_bool_array(
            gt_instances.get("ignored"), len(polygons)
        )
        curved_flags = np.array(
            [_poly_len(poly) > thr for poly in polygons], dtype=bool
        )
        new_ignore = ignore_flags | (~curved_flags)
        gt_clone = _clone_instances(gt_instances)
        gt_clone.ignored = new_ignore
        if hasattr(sample_clone, "gt_instances"):
            sample_clone.gt_instances = gt_clone
        else:
            sample_clone["gt_instances"] = gt_clone
        curved_samples.append(sample_clone)
    return curved_samples


def _eval_hmean(data_samples: Iterable) -> dict:
    metric = HmeanIOUMetric()
    metric.process(None, data_samples)
    return metric.compute_metrics(metric.results)


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)


def main():
    args = parse_args()
    data_samples = load(args.pred_pkl)
    if not isinstance(data_samples, list):
        raise TypeError(f"Expected list from pkl, got {type(data_samples)}")
    if args.sanity:
        data_samples = data_samples[:2]

    metrics_all = _eval_hmean(data_samples)
    curved_samples = _set_curved_ignored_flags(data_samples, args.curved_poly_len_thr)
    metrics_curved_instances = _eval_hmean(curved_samples)

    os.makedirs(args.out_dir, exist_ok=True)
    _write_json(os.path.join(args.out_dir, "metrics_all.json"), metrics_all)
    _write_json(
        os.path.join(args.out_dir, "metrics_curved_instances.json"),
        metrics_curved_instances,
    )

    if args.curved_images:
        curved_image_samples = []
        for sample in data_samples:
            gt_instances = sample.get("gt_instances")
            if gt_instances is None:
                continue
            polygons = gt_instances.get("polygons")
            if polygons is None:
                continue
            ignore_flags = _ensure_bool_array(
                gt_instances.get("ignored"), len(polygons)
            )
            if _has_curved_instance(polygons, ignore_flags, args.curved_poly_len_thr):
                curved_image_samples.append(sample)
        metrics_curved_images = _eval_hmean(curved_image_samples)
        _write_json(
            os.path.join(args.out_dir, "metrics_curved_images.json"),
            metrics_curved_images,
        )


if __name__ == "__main__":
    main()
