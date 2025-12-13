import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import mmengine
import numpy as np

from mmocr.utils import boundary_iou


def parse_args():
    parser = argparse.ArgumentParser(description='级联检测结果评估（整体 + 弯曲子集）')
    parser.add_argument('--ann-file', required=True, help='GT 标注 json')
    parser.add_argument('--pred-dir', required=True, help='预测结果目录或 preds.json')
    parser.add_argument(
        '--iou-thr', type=float, default=0.5, help='匹配 IoU 阈值，默认 0.5')
    return parser.parse_args()


def load_predictions(pred_dir: Path) -> Dict[str, Dict]:
    if pred_dir.is_file():
        data = mmengine.load(str(pred_dir))
    else:
        data = mmengine.load(str(pred_dir / 'preds.json'))
    pred_map = {}
    for item in data:
        pred_map[item['file_name']] = item
    return pred_map


def load_gts(ann_file: Path) -> Dict[str, Dict]:
    data = mmengine.load(str(ann_file))
    gt_map = {}
    for item in data['data_list']:
        name = Path(item['img_path']).name
        instances = []
        for inst in item.get('instances', []):
            if inst.get('ignore', False):
                continue
            poly = inst.get('polygon') or inst.get('segmentation', [])
            if not poly:
                continue
            instances.append(poly)
        gt_map[name] = dict(polygons=instances)
    return gt_map


def match(gt_polys: List[List[float]], pred_polys: List[List[float]],
          pred_scores: List[float], iou_thr: float):
    matched_gt = set()
    order = np.argsort(pred_scores)[::-1]
    tp = 0
    for idx in order:
        pred = pred_polys[idx]
        best_iou = 0
        best_gt = -1
        for gi, gt in enumerate(gt_polys):
            if gi in matched_gt:
                continue
            iou = boundary_iou(np.array(pred), np.array(gt), 1)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_thr and best_gt >= 0:
            matched_gt.add(best_gt)
            tp += 1
    fp = len(pred_polys) - tp
    fn = len(gt_polys) - tp
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    if precision + recall == 0:
        hmean = 0
    else:
        hmean = 2 * precision * recall / (precision + recall)
    return precision, recall, hmean


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    preds = load_predictions(pred_dir)
    gts = load_gts(Path(args.ann_file))
    total_tp = total_fp = total_fn = 0
    curved_tp = curved_fp = curved_fn = 0
    for name, gt_item in gts.items():
        pred_item = preds.get(name, dict(polygons=[], scores=[]))
        gt_polys = gt_item['polygons']
        pred_polys = pred_item.get('polygons', [])
        pred_scores = pred_item.get('scores', [1.0] * len(pred_polys))
        tp, fp, fn = match(gt_polys, pred_polys, pred_scores, args.iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        gt_curved = [p for p in gt_polys if len(p) > 8]
        pred_curved = [p for p in pred_polys if len(p) > 8]
        pred_curved_scores = [
            s for p, s in zip(pred_polys, pred_scores) if len(p) > 8
        ]
        tp_c, fp_c, fn_c = match(gt_curved, pred_curved, pred_curved_scores,
                                 args.iou_thr)
        curved_tp += tp_c
        curved_fp += fp_c
        curved_fn += fn_c

    p, r, f = compute_metrics(total_tp, total_fp, total_fn)
    pc, rc, fc = compute_metrics(curved_tp, curved_fp, curved_fn)
    summary = dict(
        overall=dict(precision=p, recall=r, hmean=f),
        curved=dict(precision=pc, recall=rc, hmean=fc))
    out_path = pred_dir / 'eval_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[OVERALL] P={p:.4f}, R={r:.4f}, F={f:.4f} | [CURVED] P={pc:.4f}, R={rc:.4f}, F={fc:.4f}"
    )
    print(f'[SAVE] 评估结果已写入 {out_path}')


if __name__ == '__main__':
    main()
