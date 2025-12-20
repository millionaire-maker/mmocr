# Copyright (c) OpenMMLab. All rights reserved.
"""可视化文本检测结果（对比图：左预测，右GT）。

特点：
- 自动从验证/测试 dataloader 跑推理（无需额外改 config）。
- 以单张图的 hmean 排序，默认各挑一半“效果好/效果差”样本，总数约 50 张。
- 左图：绿色=TP（正确检测），红色=FP（错检）+ FN（漏检，用 GT 位置标红并写 FN）。
- 右图：绿色=命中 GT，红色=漏检 GT，灰色=ignored GT。

用法示例（直接跑推理）：
  /bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && \
    python tools/visualizations/vis_textdet_compare.py \
      configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py \
      work_dirs/dbnetpp_r50_finetune_art_rctw_rects/best_icdar_hmean_epoch_24.pth \
      --out-dir work_dirs/dbnetpp_r50_finetune_art_rctw_rects/vis_compare'

用法示例（复用 tools/test.py --save-preds 的 pkl，不重复推理）：
  /bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && \
    python tools/test.py <config.py> <ckpt.pth> --save-preds --work-dir <work_dir>'
  /bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && \
    python tools/visualizations/vis_textdet_compare.py \
      --pred-pkl <work_dir>/<ckpt_basename>_predictions.pkl \
      --out-dir <work_dir>/vis_compare'
"""

from __future__ import annotations

import argparse
import heapq
import json
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine.fileio as fileio
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import ProgressBar, mkdir_or_exist

from mmocr.evaluation.functional import compute_hmean
from mmocr.utils import poly_intersection, poly_iou, polys2shapely
from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize textdet predictions vs GT (side-by-side).')
    parser.add_argument('config', nargs='?', help='Config file path.')
    parser.add_argument('checkpoint', nargs='?', help='Checkpoint file path.')
    parser.add_argument(
        '--pred-pkl',
        default=None,
        help='A predictions pkl dumped by tools/test.py --save-preds. '
        'If set, this script will NOT run inference again.')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Output directory to save comparison images.')
    parser.add_argument(
        '--split',
        default='val',
        choices=['val', 'test'],
        help='Which dataloader to use. Defaults to val.')
    parser.add_argument(
        '--num',
        type=int,
        default=50,
        help='Total number of images to save. Defaults to 50.')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='Prediction score threshold for visualization/matching.')
    parser.add_argument(
        '--match-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to treat a prediction as matched.')
    parser.add_argument(
        '--ignore-precision-thr',
        type=float,
        default=0.5,
        help='Precision threshold to ignore preds overlapping ignored GT.')
    parser.add_argument(
        '--strategy',
        default='vanilla',
        choices=['vanilla', 'max_matching'],
        help='Matching strategy, keep consistent with metric if needed.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Limit evaluated samples for selection. -1 means all.')
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device for inference. Defaults to cuda:0 if available.')
    parser.add_argument(
        '--dataset-prefixes',
        nargs='*',
        default=None,
        help='Optional dataset name to path prefix mapping: '
        'e.g. art=data/art_mmocr rctw=data/rctw17_mmocr.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    return parser.parse_args()


def _parse_dataset_prefixes(
        items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    if not items:
        return {}
    parsed: Dict[str, List[str]] = {}
    for item in items:
        if '=' not in item:
            raise ValueError(
                f'Invalid --dataset-prefixes item "{item}", expected k=v.')
        name, value = item.split('=', 1)
        prefixes = [v for v in value.split(',') if v]
        parsed[name] = prefixes
    return parsed


def _infer_dataset_name(img_path: str, dataset_prefixes: Dict[str, List[str]],
                        unknown: str = 'unknown') -> str:
    if not dataset_prefixes:
        return unknown
    if not isinstance(img_path, str) or not img_path:
        return unknown

    img_path_abs = osp.abspath(img_path)
    img_path_parts = set(osp.normpath(img_path).split(osp.sep))

    best_score = -1
    best_name = unknown

    for name, prefixes in dataset_prefixes.items():
        for prefix in prefixes:
            if not prefix:
                continue
            prefix_norm = osp.normpath(prefix)
            prefix_abs = osp.abspath(prefix_norm)

            try:
                if osp.commonpath([img_path_abs, prefix_abs]) == prefix_abs:
                    score = len(prefix_abs)
                    if score > best_score:
                        best_score = score
                        best_name = name
                    continue
            except Exception:
                pass

            base = osp.basename(prefix_norm)
            if base and base in img_path_parts:
                score = len(base)
                if score > best_score:
                    best_score = score
                    best_name = name

    return best_name


def _valid_poly_indices(polys: Sequence) -> List[int]:
    valid = []
    for i, p in enumerate(polys):
        try:
            arr = np.asarray(p, dtype=np.float32).reshape(-1)
            if arr.size >= 6 and arr.size % 2 == 0:
                valid.append(i)
        except Exception:
            continue
    return valid


def _to_numpy_bool(x, length: int) -> np.ndarray:
    if x is None:
        return np.zeros((length, ), dtype=bool)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=bool)
    if x.size != length:
        raise ValueError(f'ignored flags length mismatch: {x.size} vs {length}')
    return x


def _as_float_polys(polys: Sequence, indices: Sequence[int]) -> List[np.ndarray]:
    out = []
    for i in indices:
        arr = np.asarray(polys[i], dtype=np.float32).reshape(-1)
        out.append(arr)
    return out


def _draw_poly(img: np.ndarray,
               poly: np.ndarray,
               color: Tuple[int, int, int],
               thickness: int = 2) -> None:
    pts = np.round(poly).astype(np.int32).reshape(-1, 2)
    pts = pts.reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)


def _put_text(img: np.ndarray,
              text: str,
              org: Tuple[int, int],
              font_scale: float = 0.6,
              color: Tuple[int, int, int] = (255, 255, 255),
              bg_color: Tuple[int, int, int] = (0, 0, 0),
              thickness: int = 1) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x = max(0, x)
    y = max(h + baseline + 2, y)
    cv2.rectangle(img, (x, y - h - baseline - 2), (x + w + 2, y + 2),
                  bg_color, thickness=-1)
    cv2.putText(img, text, (x + 1, y - baseline - 1), font, font_scale, color,
                thickness, lineType=cv2.LINE_AA)


def _match_polys(gt_polys: List[np.ndarray],
                 gt_ignore_flags: np.ndarray,
                 pred_polys: List[np.ndarray],
                 pred_scores: np.ndarray,
                 pred_score_thr: float,
                 match_iou_thr: float,
                 ignore_precision_thr: float,
                 strategy: str) -> Tuple[dict, List[dict], List[dict]]:
    """Compute per-image match and return stats + drawable items.

    Returns:
        stats (dict): tp/fp/fn/precision/recall/hmean/gt_num/pred_num
        pred_items (list[dict]): each has poly/score/status('tp'|'fp')
        gt_items (list[dict]): each has poly/status('matched'|'missed'|'ignored')
    """
    # Convert to shapely (keep all valid polys; handle invalid via poly_iou utils)
    gt_shapely = polys2shapely(gt_polys) if gt_polys else []
    pred_shapely = polys2shapely(pred_polys) if pred_polys else []

    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    pred_ignore_flags = pred_scores < float(pred_score_thr)

    # Ignore predictions overlapping ignored GT
    if gt_polys and pred_polys and gt_ignore_flags.any():
        ignored_gt_idx = np.where(gt_ignore_flags)[0]
        for pred_id in np.where(~pred_ignore_flags)[0]:
            pred_area = float(pred_shapely[pred_id].area) + 1e-5
            for gt_id in ignored_gt_idx:
                precision = poly_intersection(gt_shapely[gt_id],
                                              pred_shapely[pred_id]) / pred_area
                if precision > ignore_precision_thr:
                    pred_ignore_flags[pred_id] = True
                    break

    valid_gt_idx = np.where(~gt_ignore_flags)[0]
    valid_pred_idx = np.where(~pred_ignore_flags)[0]

    gt_num = int(valid_gt_idx.size)
    pred_num = int(valid_pred_idx.size)

    matched_gt_rows: set = set()
    matched_pred_cols: set = set()

    if gt_num > 0 and pred_num > 0:
        iou_mat = np.zeros((gt_num, pred_num), dtype=np.float32)
        for r, gt_id in enumerate(valid_gt_idx):
            for c, pred_id in enumerate(valid_pred_idx):
                iou_mat[r, c] = poly_iou(gt_shapely[gt_id], pred_shapely[pred_id])

        matched_metric = iou_mat > float(match_iou_thr)
        if strategy == 'max_matching':
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import maximum_bipartite_matching
            matched_preds = maximum_bipartite_matching(
                csr_matrix(matched_metric), perm_type='row')
            for r, c in enumerate(matched_preds.tolist()):
                if c != -1:
                    matched_gt_rows.add(r)
                    matched_pred_cols.add(int(c))
        else:
            for r, c in zip(*np.nonzero(matched_metric)):
                r = int(r)
                c = int(c)
                if r in matched_gt_rows or c in matched_pred_cols:
                    continue
                matched_gt_rows.add(r)
                matched_pred_cols.add(c)

    tp = int(len(matched_gt_rows))
    recall, precision, hmean = compute_hmean(tp, tp, gt_num, pred_num)
    fn = int(gt_num - tp)
    fp = int(pred_num - tp)

    # Build drawable items
    gt_items: List[dict] = []
    matched_gt_abs = {int(valid_gt_idx[r]) for r in matched_gt_rows}
    missed_gt_abs = {int(i) for i in valid_gt_idx.tolist()} - matched_gt_abs
    for i, poly in enumerate(gt_polys):
        if bool(gt_ignore_flags[i]):
            status = 'ignored'
        elif i in matched_gt_abs:
            status = 'matched'
        elif i in missed_gt_abs:
            status = 'missed'
        else:
            status = 'missed'
        gt_items.append(dict(poly=poly, status=status))

    pred_items: List[dict] = []
    matched_pred_abs = {int(valid_pred_idx[c]) for c in matched_pred_cols}
    for c, pred_id in enumerate(valid_pred_idx.tolist()):
        status = 'tp' if int(pred_id) in matched_pred_abs else 'fp'
        pred_items.append(
            dict(
                poly=pred_polys[int(pred_id)],
                score=float(pred_scores[int(pred_id)]),
                status=status))

    stats = dict(
        gt_num=gt_num,
        pred_num=pred_num,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        hmean=float(hmean),
    )
    return stats, pred_items, gt_items


def _safe_name(name: str) -> str:
    name = name.replace(' ', '_')
    for ch in ['/', '\\', ':', ';', '?', '&', '#', '%']:
        name = name.replace(ch, '_')
    return name


def main():
    args = parse_args()
    register_all_modules()

    outputs_iter: Optional[Sequence] = None
    dataloader = None
    model = None

    if args.pred_pkl:
        outputs_iter = fileio.load(args.pred_pkl)
        if not isinstance(outputs_iter, list):
            raise ValueError(f'Invalid pkl content: {type(outputs_iter)}')
    else:
        if not args.config or not args.checkpoint:
            raise ValueError(
                '需要提供 (config, checkpoint) 或者 --pred-pkl 其一。')

        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        dataloader_cfg = cfg.get('val_dataloader' if args.split == 'val' else
                                 'test_dataloader')
        if dataloader_cfg is None:
            raise ValueError(f'Config has no {args.split}_dataloader')

        # Build a non-shuffled dataloader for deterministic selection
        dataloader_cfg = dataloader_cfg.copy()
        if isinstance(dataloader_cfg.get('sampler', None), dict):
            dataloader_cfg['sampler'] = dataloader_cfg['sampler'].copy()
            dataloader_cfg['sampler']['shuffle'] = False
        dataloader = Runner.build_dataloader(dataloader_cfg, seed=0)

        device = torch.device(args.device)
        from mmocr.registry import MODELS
        model = MODELS.build(cfg.model)
        model.to(device)
        model.eval()
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)

    out_dir = osp.abspath(args.out_dir)
    mkdir_or_exist(out_dir)
    out_good = osp.join(out_dir, 'good')
    out_bad = osp.join(out_dir, 'bad')
    mkdir_or_exist(out_good)
    mkdir_or_exist(out_bad)

    good_num = max(0, args.num // 2)
    bad_num = max(0, args.num - good_num)
    best_heap: List[Tuple[float, int, dict]] = []
    worst_heap: List[Tuple[float, int, dict]] = []  # store (-hmean, idx, rec)

    dataset_prefixes = _parse_dataset_prefixes(args.dataset_prefixes)

    if outputs_iter is not None:
        total = len(outputs_iter)
    else:
        assert dataloader is not None
        total = len(dataloader.dataset)  # type: ignore[arg-type]
    prog = ProgressBar(total)
    global_idx = 0
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    def _iter_outputs():
        if outputs_iter is not None:
            for o in outputs_iter:
                yield o
        else:
            assert dataloader is not None and model is not None
            with torch.no_grad():
                for data_batch in dataloader:
                    for o in model.test_step(data_batch):
                        yield o

    for output in _iter_outputs():
        if isinstance(output, dict):
            metainfo = output.get('metainfo', {}) or {}
            img_path = metainfo.get('img_path', '')
            gt_instances = output.get('gt_instances', None)
            pred_instances = output.get('pred_instances', None)
        else:
            img_path = getattr(output, 'img_path', None) or output.metainfo.get(
                'img_path', '')
            gt_instances = getattr(output, 'gt_instances', None)
            pred_instances = getattr(output, 'pred_instances', None)

        if gt_instances is None or pred_instances is None:
            global_idx += 1
            prog.update()
            continue

        gt_polys_raw = gt_instances.get('polygons', []) or []
        gt_valid_idx = _valid_poly_indices(gt_polys_raw)
        gt_polys = _as_float_polys(gt_polys_raw, gt_valid_idx)
        gt_ignore_raw = gt_instances.get('ignored', None)
        gt_ignore_flags = _to_numpy_bool(gt_ignore_raw, len(gt_polys_raw))
        gt_ignore_flags = gt_ignore_flags[gt_valid_idx] if gt_valid_idx else np.zeros(
            (0, ), dtype=bool)

        pred_polys_raw = pred_instances.get('polygons', []) or []
        pred_scores_raw = pred_instances.get('scores', []) or []
        if isinstance(pred_scores_raw, torch.Tensor):
            pred_scores_raw = pred_scores_raw.detach().cpu().numpy()
        pred_scores_raw = np.asarray(pred_scores_raw, dtype=np.float32)

        pred_valid_idx = _valid_poly_indices(pred_polys_raw)
        pred_polys = _as_float_polys(pred_polys_raw, pred_valid_idx)
        pred_scores = pred_scores_raw[pred_valid_idx] if pred_valid_idx else np.zeros(
            (0, ), dtype=np.float32)

        try:
            stats, pred_items, gt_items = _match_polys(
                gt_polys=gt_polys,
                gt_ignore_flags=gt_ignore_flags,
                pred_polys=pred_polys,
                pred_scores=pred_scores,
                pred_score_thr=args.pred_score_thr,
                match_iou_thr=args.match_iou_thr,
                ignore_precision_thr=args.ignore_precision_thr,
                strategy=args.strategy,
            )
        except Exception:
            global_idx += 1
            prog.update()
            continue

        dataset_name = _infer_dataset_name(img_path, dataset_prefixes)
        record = dict(
            idx=global_idx,
            dataset=dataset_name,
            img_path=img_path,
            stats=stats,
            pred_items=pred_items,
            gt_items=gt_items,
        )

        h = float(stats['hmean'])
        if good_num > 0:
            item = (h, global_idx, record)
            if len(best_heap) < good_num:
                heapq.heappush(best_heap, item)
            elif item[0] > best_heap[0][0]:
                heapq.heapreplace(best_heap, item)
        if bad_num > 0:
            item = (-h, global_idx, record)
            if len(worst_heap) < bad_num:
                heapq.heappush(worst_heap, item)
            elif item[0] > worst_heap[0][0]:
                heapq.heapreplace(worst_heap, item)

        global_idx += 1
        prog.update()
        if max_samples is not None and global_idx >= max_samples:
            break

    best_records = [r for _, _, r in sorted(best_heap, key=lambda x: x[0], reverse=True)]
    worst_records = [r for _, _, r in sorted(worst_heap, key=lambda x: x[0])]

    def _read_bgr(path: str) -> Optional[np.ndarray]:
        try:
            img_bytes = fileio.get(path)
            return mmcv.imfrombytes(img_bytes, channel_order='bgr')
        except Exception:
            return None

    summary: Dict[str, List[dict]] = dict(good=[], bad=[])

    def _save_group(records: List[dict], group: str, out_subdir: str) -> None:
        for rank, rec in enumerate(records):
            img = _read_bgr(rec['img_path'])
            if img is None:
                continue
            pred_img = img.copy()
            gt_img = img.copy()

            stats = rec['stats']
            dataset_name = rec['dataset']
            base = osp.basename(rec['img_path'])
            base_safe = _safe_name(osp.splitext(base)[0])

            # Colors in BGR
            color_tp = (0, 255, 0)
            color_err = (0, 0, 255)
            color_ign = (180, 180, 180)

            # Pred panel: TP green, FP red
            for item in rec['pred_items']:
                poly = item['poly']
                status = item['status']
                score = item.get('score', None)
                color = color_tp if status == 'tp' else color_err
                _draw_poly(pred_img, poly, color=color, thickness=2)
                if score is not None:
                    pts = np.round(poly).astype(np.int32).reshape(-1, 2)
                    cx = int(np.clip(pts[:, 0].mean(), 0, pred_img.shape[1] - 1))
                    cy = int(np.clip(pts[:, 1].mean(), 0, pred_img.shape[0] - 1))
                    _put_text(pred_img, f'{status.upper()} {score:.2f}',
                              (cx, cy), font_scale=0.5)

            # Pred panel: mark missed GT as FN (red)
            for item in rec['gt_items']:
                if item['status'] != 'missed':
                    continue
                poly = item['poly']
                _draw_poly(pred_img, poly, color=color_err, thickness=3)
                pts = np.round(poly).astype(np.int32).reshape(-1, 2)
                cx = int(np.clip(pts[:, 0].mean(), 0, pred_img.shape[1] - 1))
                cy = int(np.clip(pts[:, 1].mean(), 0, pred_img.shape[0] - 1))
                _put_text(pred_img, 'FN', (cx, cy), font_scale=0.6)

            # GT panel: matched green, missed red, ignored gray
            for item in rec['gt_items']:
                poly = item['poly']
                status = item['status']
                if status == 'ignored':
                    color = color_ign
                    thickness = 2
                elif status == 'matched':
                    color = color_tp
                    thickness = 2
                else:
                    color = color_err
                    thickness = 3
                _draw_poly(gt_img, poly, color=color, thickness=thickness)

            # Titles & stats
            stat_line = (
                f'H={stats["hmean"]:.3f} P={stats["precision"]:.3f} '
                f'R={stats["recall"]:.3f} TP={stats["tp"]} FP={stats["fp"]} '
                f'FN={stats["fn"]}')
            _put_text(pred_img, f'PRED {dataset_name} | {base}', (5, 22))
            _put_text(pred_img, stat_line, (5, 46), font_scale=0.55)
            _put_text(gt_img, f'GT {dataset_name} | {base}', (5, 22))
            _put_text(gt_img, stat_line, (5, 46), font_scale=0.55)

            comp = np.concatenate([pred_img, gt_img], axis=1)

            out_name = (f'{rank:03d}_h{stats["hmean"]:.3f}_p{stats["precision"]:.3f}'
                        f'_r{stats["recall"]:.3f}_{base_safe}.jpg')
            out_path = osp.join(out_subdir, out_name)
            mmcv.imwrite(comp, out_path)

            summary[group].append(
                dict(
                    out_file=out_path,
                    img_path=rec['img_path'],
                    dataset=dataset_name,
                    **stats,
                ))

    _save_group(best_records, 'good', out_good)
    _save_group(worst_records, 'bad', out_bad)

    with open(osp.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'已生成对比图：{out_dir}')


if __name__ == '__main__':
    main()
