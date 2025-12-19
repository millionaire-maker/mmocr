import argparse
import json
import os
from typing import List, Tuple

import lmdb
import mmcv
from mmengine.structures import LabelData

from mmocr.apis.inferencers import TextRecInferencer
from mmocr.evaluation.metrics import OneMinusNEDMetric, WordMetric
from mmocr.structures import TextRecogDataSample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate textrecog on all/curved subsets with LMDB + meta."
    )
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--eval-lmdb", required=True, help="LMDB path.")
    parser.add_argument("--meta-json", required=True, help="Meta json path.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to save metrics json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples for sanity check. 0 means all.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (e.g. cuda, cpu).",
    )
    return parser.parse_args()


def load_meta(meta_path: str) -> List[dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "samples" in meta:
        meta = meta["samples"]
    if not isinstance(meta, list):
        raise TypeError(f"Meta json must be list or dict with 'samples', got {type(meta)}")
    meta = sorted(meta, key=lambda x: x.get("index", 0))
    return meta


def open_lmdb(lmdb_path: str):
    return lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def load_lmdb_images(
    env, meta_list: List[dict]
) -> Tuple[List, List[dict]]:
    images = []
    used_meta = []
    with env.begin(write=False) as txn:
        for entry in meta_list:
            idx = int(entry.get("index", 0))
            if idx <= 0:
                continue
            img_key = f"image-{idx:09d}".encode("utf-8")
            img_bytes = txn.get(img_key)
            if img_bytes is None:
                continue
            img = mmcv.imfrombytes(img_bytes, flag="color")
            images.append(img)
            used_meta.append(entry)
    return images, used_meta


def build_eval_samples(pred_samples: List, meta_list: List[dict]) -> List:
    eval_samples = []
    for pred_sample, meta in zip(pred_samples, meta_list):
        gt_text = LabelData()
        gt_text.item = meta.get("text", "")
        if isinstance(pred_sample, TextRecogDataSample):
            pred_sample = pred_sample.clone()
            pred_sample.gt_text = gt_text
            eval_samples.append(pred_sample)
        else:
            data_sample = TextRecogDataSample()
            data_sample.pred_text = pred_sample.pred_text
            data_sample.gt_text = gt_text
            eval_samples.append(data_sample)
    return eval_samples


def eval_rec_metrics(data_samples: List) -> dict:
    word_metric = WordMetric(mode="exact")
    ned_metric = OneMinusNEDMetric()
    word_metric.process(None, data_samples)
    ned_metric.process(None, data_samples)
    metrics = {}
    metrics.update(word_metric.compute_metrics(word_metric.results))
    metrics.update(ned_metric.compute_metrics(ned_metric.results))
    return metrics


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)


def main():
    args = parse_args()
    meta_list = load_meta(args.meta_json)
    if args.max_samples and args.max_samples > 0:
        meta_list = meta_list[: args.max_samples]

    env = open_lmdb(args.eval_lmdb)
    images, used_meta = load_lmdb_images(env, meta_list)
    env.close()

    inferencer = TextRecInferencer(
        model=args.config, weights=args.checkpoint, device=args.device
    )
    results = inferencer(
        images,
        return_datasamples=True,
        batch_size=1,
        progress_bar=True,
    )
    pred_samples = results["predictions"]
    eval_samples = build_eval_samples(pred_samples, used_meta)

    metrics_all = eval_rec_metrics(eval_samples)
    curved_samples = [
        sample for sample, meta in zip(eval_samples, used_meta) if meta.get("curved") == 1
    ]
    metrics_curved = eval_rec_metrics(curved_samples)

    os.makedirs(args.out_dir, exist_ok=True)
    write_json(os.path.join(args.out_dir, "metrics_rec_all.json"), metrics_all)
    write_json(
        os.path.join(args.out_dir, "metrics_rec_curved.json"), metrics_curved
    )


if __name__ == "__main__":
    main()
