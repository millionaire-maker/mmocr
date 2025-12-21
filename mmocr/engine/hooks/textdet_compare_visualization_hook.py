# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import heapq
import json
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine.dist as dist
import mmengine.fileio as fileio
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmocr.evaluation.functional import compute_hmean
from mmocr.registry import HOOKS
from mmocr.utils import poly_intersection, poly_iou, polys2shapely


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
               poly: Union[np.ndarray, Sequence[float]],
               color: Tuple[int, int, int],
               thickness: int = 2) -> None:
    pts = np.round(np.asarray(poly, dtype=np.float32)).astype(np.int32)
    pts = pts.reshape(-1, 2).reshape(-1, 1, 2)
    cv2.polylines(img, [pts],
                  isClosed=True,
                  color=color,
                  thickness=thickness,
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


def _safe_name(name: str) -> str:
    name = name.replace(' ', '_')
    for ch in ['/', '\\', ':', ';', '?', '&', '#', '%']:
        name = name.replace(ch, '_')
    return name


def _poly_to_list(poly: Union[np.ndarray, Sequence[float]]) -> List[float]:
    arr = np.asarray(poly, dtype=np.float32).reshape(-1)
    return [float(x) for x in arr.tolist()]


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


def _match_polys(gt_polys: List[np.ndarray],
                 gt_ignore_flags: np.ndarray,
                 pred_polys: List[np.ndarray],
                 pred_scores: np.ndarray,
                 pred_score_thr: float,
                 match_iou_thr: float,
                 ignore_precision_thr: float,
                 strategy: str) -> Tuple[dict, List[dict], List[dict]]:
    gt_shapely = polys2shapely(gt_polys) if gt_polys else []
    pred_shapely = polys2shapely(pred_polys) if pred_polys else []

    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    pred_ignore_flags = pred_scores < float(pred_score_thr)

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
                    matched_gt_rows.add(int(r))
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
    for pred_id in valid_pred_idx.tolist():
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


@HOOKS.register_module()
class TextDetCompareVisualizationHook(Hook):
    """在每次验证/测试 epoch 结束后，自动输出对比可视化（左预测/右GT）与坐标文件。

    - 左图：只画预测框；绿色=TP，红色=FP
    - 右图：绿色=命中GT，红色=漏检GT，灰色=ignored GT
    - 输出：work_dir/<out_dir>/val_epoch_xxx 或 test_xxx 下的图片与
      compare_instances.json

    注意：如需生效，需要在 config 中启用该 hook（custom_hooks / default_hooks）。
    """

    def __init__(self,
                 enable: bool = True,
                 out_dir: str = 'vis_compare_auto',
                 num_images: int = 50,
                 good_ratio: float = 0.5,
                 pred_score_thr: float = 0.3,
                 match_iou_thr: float = 0.5,
                 ignore_precision_thr: float = 0.5,
                 strategy: str = 'vanilla',
                 dataset_prefixes: Optional[Dict[str, Union[str,
                                                           Sequence[str]]]] =
                 None,
                 only_datasets: Optional[Sequence[str]] = None,
                 max_samples: int = -1,
                 show_score: bool = True) -> None:
        self.enable = enable
        self.out_dir = out_dir
        self.num_images = int(num_images)
        self.good_ratio = float(good_ratio)
        self.pred_score_thr = float(pred_score_thr)
        self.match_iou_thr = float(match_iou_thr)
        self.ignore_precision_thr = float(ignore_precision_thr)
        self.strategy = str(strategy)
        self.max_samples = int(max_samples)
        self.show_score = bool(show_score)

        normalized: Dict[str, List[str]] = {}
        if dataset_prefixes:
            for name, prefixes in dataset_prefixes.items():
                if isinstance(prefixes, str):
                    normalized[name] = [prefixes]
                else:
                    normalized[name] = list(prefixes)
        self.dataset_prefixes = normalized
        self.only_datasets = set(only_datasets) if only_datasets else None

        self._reset_state()

    def _reset_state(self) -> None:
        self._counter = 0
        self._seen = 0
        self._best_heap: List[Tuple[float, int, dict]] = []
        self._worst_heap: List[Tuple[float, int, dict]] = []

    def before_val_epoch(self, runner: Runner) -> None:
        if not self.enable:
            return
        self._reset_state()

    def before_test_epoch(self, runner: Runner) -> None:
        if not self.enable:
            return
        self._reset_state()

    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: Sequence[dict],
                       outputs: Sequence) -> None:
        if not self.enable:
            return
        self._process_outputs(outputs)

    def after_test_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: Sequence[dict],
                        outputs: Sequence) -> None:
        if not self.enable:
            return
        self._process_outputs(outputs)

    def _process_outputs(self, outputs: Sequence) -> None:
        if self.max_samples > 0 and self._seen >= self.max_samples:
            return

        good_num = max(0, int(round(self.num_images * self.good_ratio)))
        bad_num = max(0, int(self.num_images - good_num))

        for output in outputs:
            if self.max_samples > 0 and self._seen >= self.max_samples:
                break

            img_path = getattr(output, 'img_path', None)
            if not img_path:
                img_path = getattr(output, 'metainfo', {}).get('img_path', '')
            if not img_path:
                continue

            dataset_name = _infer_dataset_name(img_path, self.dataset_prefixes)
            if self.only_datasets is not None and dataset_name not in self.only_datasets:
                self._seen += 1
                continue

            gt_instances = getattr(output, 'gt_instances', None)
            pred_instances = getattr(output, 'pred_instances', None)
            if gt_instances is None or pred_instances is None:
                self._seen += 1
                continue

            gt_polys_raw = gt_instances.get('polygons', None)
            if gt_polys_raw is None:
                self._seen += 1
                continue

            gt_valid_idx = _valid_poly_indices(gt_polys_raw)
            gt_polys = _as_float_polys(gt_polys_raw, gt_valid_idx)

            gt_ignore_raw = gt_instances.get('ignored', None)
            gt_ignore_flags = _to_numpy_bool(gt_ignore_raw, len(gt_polys_raw))
            gt_ignore_flags = gt_ignore_flags[gt_valid_idx] if gt_valid_idx else np.zeros(
                (0, ), dtype=bool)

            pred_polys_raw = pred_instances.get('polygons', None) or []
            pred_scores_raw = pred_instances.get('scores', None)
            if pred_scores_raw is None:
                pred_scores_raw = []
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
                    pred_score_thr=self.pred_score_thr,
                    match_iou_thr=self.match_iou_thr,
                    ignore_precision_thr=self.ignore_precision_thr,
                    strategy=self.strategy,
                )
            except Exception:
                self._seen += 1
                continue

            record = dict(
                key=str(img_path),
                dataset=dataset_name,
                img_path=str(img_path),
                stats=stats,
                pred_items=pred_items,
                gt_items=gt_items,
            )

            h = float(stats['hmean'])
            if good_num > 0:
                item = (h, self._counter, record)
                self._counter += 1
                if len(self._best_heap) < good_num:
                    heapq.heappush(self._best_heap, item)
                elif item[0] > self._best_heap[0][0]:
                    heapq.heapreplace(self._best_heap, item)
            if bad_num > 0:
                item = (-h, self._counter, record)
                self._counter += 1
                if len(self._worst_heap) < bad_num:
                    heapq.heappush(self._worst_heap, item)
                elif item[0] > self._worst_heap[0][0]:
                    heapq.heapreplace(self._worst_heap, item)

            self._seen += 1

    def after_val_epoch(self, runner: Runner, metrics: Optional[dict] = None) -> None:
        if not self.enable:
            return
        self._dump(runner, phase='val')

    def after_test_epoch(self, runner: Runner, metrics: Optional[dict] = None) -> None:
        if not self.enable:
            return
        self._dump(runner, phase='test')

    def _dump(self, runner: Runner, phase: str) -> None:
        good_num = max(0, int(round(self.num_images * self.good_ratio)))
        bad_num = max(0, int(self.num_images - good_num))

        local_best = [r for _, _, r in self._best_heap]
        local_worst = [r for _, _, r in self._worst_heap]
        gathered = dist.all_gather_object(dict(best=local_best, worst=local_worst))

        if not dist.is_main_process():
            return

        merged_best: List[dict] = []
        merged_worst: List[dict] = []
        for item in gathered:
            merged_best.extend(item.get('best', []))
            merged_worst.extend(item.get('worst', []))

        def _unique_sort(records: List[dict], reverse: bool) -> List[dict]:
            seen: set = set()
            out: List[dict] = []
            for rec in sorted(records,
                              key=lambda r: float(r['stats']['hmean']),
                              reverse=reverse):
                key = rec.get('key', rec.get('img_path', ''))
                if key in seen:
                    continue
                seen.add(key)
                out.append(rec)
            return out

        best_records = _unique_sort(merged_best, reverse=True)[:good_num]
        worst_records = _unique_sort(merged_worst, reverse=False)[:bad_num]

        if phase == 'val':
            epoch = getattr(runner, 'epoch', 0)
            tag = f'val_epoch_{int(epoch) + 1}'
        else:
            tag = f'test_{runner.timestamp}'

        out_root = osp.join(runner.work_dir, self.out_dir, tag)
        out_good = osp.join(out_root, 'good')
        out_bad = osp.join(out_root, 'bad')
        mkdir_or_exist(out_good)
        mkdir_or_exist(out_bad)

        def _read_bgr(path: str) -> Optional[np.ndarray]:
            try:
                img_bytes = fileio.get(path)
                return mmcv.imfrombytes(img_bytes, channel_order='bgr')
            except Exception:
                return None

        export_items: List[dict] = []

        def _save_group(records: List[dict], group: str, out_subdir: str) -> None:
            for rank, rec in enumerate(records):
                img = _read_bgr(rec['img_path'])
                if img is None:
                    continue
                pred_img = img.copy()
                gt_img = img.copy()

                stats = rec['stats']
                dataset_name = rec.get('dataset', 'unknown')
                base = osp.basename(rec['img_path'])
                base_safe = _safe_name(osp.splitext(base)[0])

                # Colors in BGR
                color_tp = (0, 255, 0)
                color_err = (0, 0, 255)
                color_ign = (180, 180, 180)

                # Left: predictions only
                for item in rec['pred_items']:
                    poly = item['poly']
                    status = item.get('status', '')
                    score = item.get('score', None)
                    color = color_tp if status == 'tp' else color_err
                    _draw_poly(pred_img, poly, color=color, thickness=2)
                    if self.show_score and score is not None:
                        pts = np.round(np.asarray(poly, dtype=np.float32)).astype(
                            np.int32).reshape(-1, 2)
                        cx = int(np.clip(pts[:, 0].mean(), 0, pred_img.shape[1] - 1))
                        cy = int(np.clip(pts[:, 1].mean(), 0, pred_img.shape[0] - 1))
                        _put_text(pred_img, f'{status.upper()} {float(score):.2f}',
                                  (cx, cy), font_scale=0.5)

                # Right: GT status
                for item in rec['gt_items']:
                    poly = item['poly']
                    status = item.get('status', '')
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

                stat_line = (
                    f'H={stats["hmean"]:.3f} P={stats["precision"]:.3f} '
                    f'R={stats["recall"]:.3f} TP={stats["tp"]} FP={stats["fp"]} '
                    f'FN={stats["fn"]}')
                _put_text(pred_img, f'PRED {dataset_name} | {base}', (5, 22))
                _put_text(pred_img, stat_line, (5, 46), font_scale=0.55)
                _put_text(gt_img, f'GT {dataset_name} | {base}', (5, 22))
                _put_text(gt_img, stat_line, (5, 46), font_scale=0.55)

                comp = np.concatenate([pred_img, gt_img], axis=1)

                out_name = (
                    f'{rank:03d}_h{stats["hmean"]:.3f}_p{stats["precision"]:.3f}'
                    f'_r{stats["recall"]:.3f}_{base_safe}.jpg')
                out_path = osp.join(out_subdir, out_name)
                mmcv.imwrite(comp, out_path)

                export_items.append(
                    dict(
                        group=group,
                        rank=rank,
                        out_file=out_path,
                        img_path=rec['img_path'],
                        dataset=dataset_name,
                        stats=stats,
                        pred_instances=[
                            dict(
                                poly=_poly_to_list(p['poly']),
                                score=float(p.get('score', 0.0)),
                                status=str(p.get('status', '')),
                            ) for p in rec['pred_items']
                        ],
                        gt_instances=[
                            dict(
                                poly=_poly_to_list(g['poly']),
                                status=str(g.get('status', '')),
                                ignored=(g.get('status', '') == 'ignored'),
                            ) for g in rec['gt_items']
                        ],
                    ))

        _save_group(best_records, 'good', out_good)
        _save_group(worst_records, 'bad', out_bad)

        export_path = osp.join(out_root, 'compare_instances.json')
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(
                dict(
                    meta=dict(
                        phase=phase,
                        tag=tag,
                        num_images=self.num_images,
                        good_ratio=self.good_ratio,
                        pred_score_thr=self.pred_score_thr,
                        match_iou_thr=self.match_iou_thr,
                        ignore_precision_thr=self.ignore_precision_thr,
                        strategy=self.strategy,
                        dataset_prefixes=self.dataset_prefixes,
                        only_datasets=sorted(self.only_datasets)
                        if self.only_datasets else None,
                        max_samples=self.max_samples,
                    ),
                    items=export_items,
                ),
                f,
                ensure_ascii=False,
                indent=2)

        runner.logger.info(f'TextDetCompareVisualizationHook saved: {out_root}')

