import argparse
import json
from pathlib import Path
from typing import List, Tuple

import mmcv
import numpy as np
from mmocr.apis import TextDetInferencer
from mmocr.models.textdet.postprocessors.base import BaseTextDetPostProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='DBNet++ -> FCENet 级联推理')
    parser.add_argument('--dbnetpp-config', required=True, help='DBNet++ 配置')
    parser.add_argument('--dbnetpp-ckpt', required=True, help='DBNet++ 权重')
    parser.add_argument('--fcenet-config', required=True, help='FCENet 配置')
    parser.add_argument('--fcenet-ckpt', required=True, help='FCENet 权重')
    parser.add_argument('--img-root', required=True, help='待推理图片目录')
    parser.add_argument('--out-dir', required=True, help='输出目录')
    parser.add_argument(
        '--db-score-thr', type=float, default=0.3, help='DBNet++ score 阈值')
    parser.add_argument(
        '--expand-ratio',
        type=float,
        default=0.1,
        help='裁剪 patch 时的扩张比例')
    parser.add_argument(
        '--nms-thr', type=float, default=0.3, help='polygon NMS 阈值')
    parser.add_argument(
        '--device', default=None, help='设备号，如 cuda:0，默认自动')
    parser.add_argument(
        '--save-vis',
        action='store_true',
        help='是否额外保存可视化图片（绿色多边形）')
    return parser.parse_args()


def poly_to_bbox(poly: np.ndarray, expand: float,
                 img_shape: Tuple[int, int, int]):
    xs, ys = poly[::2], poly[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * expand)
    y1 = max(0, y1 - h * expand)
    x2 = min(img_shape[1] - 1, x2 + w * expand)
    y2 = min(img_shape[0] - 1, y2 + h * expand)
    return int(x1), int(y1), int(x2), int(y2)


def run_cascade_on_image(db_inferencer: TextDetInferencer,
                         fce_inferencer: TextDetInferencer, img_path: Path,
                         score_thr: float, expand_ratio: float,
                         nms_thr: float):
    img = mmcv.imread(str(img_path))
    db_pred = db_inferencer(
        str(img_path),
        return_datasamples=True,
        progress_bar=False,
        batch_size=1)
    db_sample = db_pred['predictions'][0]
    db_polys = db_sample.pred_instances.polygons
    db_scores = db_sample.pred_instances.scores
    refined_polys = []
    refined_scores = []
    for poly, score in zip(db_polys, db_scores):
        if score < score_thr:
            continue
        bbox = poly_to_bbox(poly, expand_ratio, img.shape)
        x1, y1, x2, y2 = bbox
        patch = img[y1:y2, x1:x2, :]
        if patch.size == 0:
            continue
        fc_pred = fce_inferencer(
            patch, return_datasamples=True, progress_bar=False, batch_size=1)
        fc_sample = fc_pred['predictions'][0]
        fc_polys = fc_sample.pred_instances.polygons
        fc_scores = fc_sample.pred_instances.scores
        if len(fc_polys) == 0:
            refined_polys.append(poly.tolist())
            refined_scores.append(float(score))
            continue
        for fpoly, fs in zip(fc_polys, fc_scores):
            mapped = []
            for idx, val in enumerate(fpoly):
                if idx % 2 == 0:
                    mapped.append(float(val + x1))
                else:
                    mapped.append(float(val + y1))
            refined_polys.append(mapped)
            refined_scores.append(float(fs))
    if not refined_polys:
        refined_polys = [p.tolist() for p in db_polys]
        refined_scores = [float(s) for s in db_scores]
    postprocessor = BaseTextDetPostProcessor()
    keep_polys, keep_scores = postprocessor.poly_nms(refined_polys,
                                                     refined_scores, nms_thr)
    return keep_polys, keep_scores


def save_results(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'preds.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'[SAVE] 写入预测到 {json_path}')


def visualize(img_path: Path, polys: List[List[float]], out_dir: Path):
    img = mmcv.imread(str(img_path))
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        img = mmcv.polylines(
            img, [pts], color=(0, 255, 0), thickness=2, is_closed=True)
    vis_path = out_dir / f'{img_path.stem}_vis.jpg'
    mmcv.imwrite(img, str(vis_path))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_inferencer = TextDetInferencer(
        model=args.dbnetpp_config,
        weights=args.dbnetpp_ckpt,
        device=args.device)
    fce_inferencer = TextDetInferencer(
        model=args.fcenet_config,
        weights=args.fcenet_ckpt,
        device=args.device)
    img_root = Path(args.img_root)
    image_files = sorted([
        p for p in img_root.rglob('*')
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    ])
    all_results = []
    for img_path in image_files:
        polys, scores = run_cascade_on_image(db_inferencer, fce_inferencer,
                                             img_path, args.db_score_thr,
                                             args.expand_ratio, args.nms_thr)
        all_results.append(
            dict(
                file_name=img_path.name,
                polygons=polys,
                scores=scores,
            ))
        if args.save_vis:
            visualize(img_path, polys, out_dir)
    save_results(all_results, out_dir)


if __name__ == '__main__':
    main()
