#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import copy
import json
import math
import os
import os.path as osp
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mmcv
import numpy as np
import torch
from mmengine.config import Config
from mmengine.fileio import dump, load
from mmengine.runner.checkpoint import load_checkpoint

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils import poly2shapely, poly_iou, register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='离线 coarse-to-fine gated refine（Stage-2: FCENet）')
    parser.add_argument('--stage1-pred-pkl', required=True, help='Stage-1 pkl')
    parser.add_argument('--out-pkl', required=True, help='refined pkl 输出')
    parser.add_argument('--stage2-config', required=True, help='FCENet config')
    parser.add_argument(
        '--stage2-ckpt',
        default='',
        help='FCENet ckpt（无权重时配合 --dry-run-refine）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dry-run-refine',
        action='store_true',
        help='不跑 FCENet，仅验证裁 patch + 映射 + NMS + fallback 流程')
    parser.add_argument('--expand-ratio', type=float, default=0.2)
    parser.add_argument('--max-patch-long-edge', type=int, default=1024)
    parser.add_argument('--pad-divisor', type=int, default=32)
    parser.add_argument(
        '--gating-mode',
        choices=['ratio', 'score_range'],
        default='ratio')
    parser.add_argument('--refine-ratio', type=float, default=0.3)
    parser.add_argument('--score-low', type=float, default=0.3)
    parser.add_argument('--score-high', type=float, default=0.7)
    parser.add_argument('--aspect-ratio-thr', type=float, default=5.0)
    parser.add_argument('--topk-per-image', type=int, default=10)
    parser.add_argument('--nms-iou-thr', type=float, default=0.2)
    parser.add_argument('--save-debug-vis', action='store_true')
    parser.add_argument('--max-images', type=int, default=0)
    return parser.parse_args()


def _get_img_path(data_sample) -> str:
    if data_sample is None:
        return ''
    if isinstance(data_sample, dict):
        img_path = data_sample.get('img_path', '')
        if img_path:
            return img_path
        metainfo = data_sample.get('metainfo', {})
        if isinstance(metainfo, dict):
            return metainfo.get('img_path', '')
        return ''
    metainfo = getattr(data_sample, 'metainfo', None)
    if isinstance(metainfo, dict) and metainfo.get('img_path', ''):
        return metainfo['img_path']
    if hasattr(data_sample, 'get'):
        img_path = data_sample.get('img_path', '')
        if img_path:
            return img_path
        metainfo = data_sample.get('metainfo', {})
        if isinstance(metainfo, dict):
            return metainfo.get('img_path', '')
    return ''


def _as_poly_array(poly) -> Optional[np.ndarray]:
    try:
        arr = np.array(poly, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 6 or (arr.size % 2 != 0):
        return None
    return arr


def _poly_points(poly: np.ndarray) -> np.ndarray:
    return poly.reshape(-1, 2).astype(np.float32)


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def _safe_poly_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    try:
        return float(poly_iou(poly2shapely(poly_a), poly2shapely(poly_b)))
    except Exception:
        return 0.0


def _poly_area(poly: np.ndarray) -> float:
    try:
        pts = _poly_points(poly)
        return float(abs(cv2.contourArea(pts)))
    except Exception:
        return 0.0


def _min_area_rect_stats(poly: np.ndarray) -> Tuple[float, float, float]:
    """Return (w, h, aspect_ratio=max(w,h)/min(w,h))."""
    pts = _poly_points(poly)
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    w = float(w)
    h = float(h)
    if w <= 1e-6 or h <= 1e-6:
        return w, h, float('inf')
    return w, h, max(w, h) / max(min(w, h), 1e-6)


def _crop_rotated_patch(
    image_bgr: np.ndarray,
    poly: np.ndarray,
    expand_ratio: float,
    max_long_edge: int,
    pad_divisor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Crop a rotated patch by minAreaRect + perspective warp.

    Returns:
        patch_bgr: padded patch (H_pad, W_pad, 3)
        mat_orig2patch: 3x3 perspective transform (orig -> patch_scaled)
        mat_patch2orig: inverse transform (patch_scaled -> orig)
        valid_hw: (h_scaled, w_scaled) before padding
    """
    pts = _poly_points(poly)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    w = max(float(w), 2.0)
    h = max(float(h), 2.0)

    # expand by margin on both sides -> (1 + 2r)
    w *= (1.0 + 2.0 * expand_ratio)
    h *= (1.0 + 2.0 * expand_ratio)
    rect_expanded = ((float(cx), float(cy)), (float(w), float(h)), float(angle))

    box = cv2.boxPoints(rect_expanded)  # (4,2)
    box = _order_points_clockwise(box)

    raw_w = max(int(round(np.linalg.norm(box[0] - box[1]))), 2)
    raw_h = max(int(round(np.linalg.norm(box[0] - box[3]))), 2)

    scale = float(max_long_edge) / float(max(raw_w, raw_h))
    scaled_w = max(int(round(raw_w * scale)), 2)
    scaled_h = max(int(round(raw_h * scale)), 2)

    dst = np.array(
        [[0, 0], [scaled_w - 1, 0], [scaled_w - 1, scaled_h - 1],
         [0, scaled_h - 1]],
        dtype=np.float32)

    mat_orig2patch = cv2.getPerspectiveTransform(box, dst)
    patch = cv2.warpPerspective(
        image_bgr,
        mat_orig2patch,
        (scaled_w, scaled_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    pad_w = int(math.ceil(scaled_w / pad_divisor) * pad_divisor)
    pad_h = int(math.ceil(scaled_h / pad_divisor) * pad_divisor)
    if pad_w != scaled_w or pad_h != scaled_h:
        padded = np.zeros((pad_h, pad_w, 3), dtype=patch.dtype)
        padded[:scaled_h, :scaled_w] = patch
        patch = padded

    mat_patch2orig = np.linalg.inv(mat_orig2patch)
    return patch, mat_orig2patch, mat_patch2orig, (scaled_h, scaled_w)


def _transform_polygons(
    polys: Sequence[np.ndarray],
    mat_3x3: np.ndarray,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for poly in polys:
        arr = _as_poly_array(poly)
        if arr is None:
            continue
        pts = arr.reshape(-1, 2).astype(np.float32)
        pts = pts.reshape(-1, 1, 2)
        try:
            mapped = cv2.perspectiveTransform(pts, mat_3x3)
        except Exception:
            continue
        out.append(mapped.reshape(-1, 2).reshape(-1).astype(np.float32))
    return out


def _clip_polygon(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    arr = poly.reshape(-1, 2)
    arr[:, 0] = np.clip(arr[:, 0], 0, max(w - 1, 0))
    arr[:, 1] = np.clip(arr[:, 1], 0, max(h - 1, 0))
    return arr.reshape(-1).astype(np.float32)


def _bbox_from_poly(poly: np.ndarray) -> Tuple[float, float, float, float]:
    pts = poly.reshape(-1, 2)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return x1, y1, x2, y2


def _bbox_overlap(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax2, bx2) - max(ax1, bx1) > 0) and (min(ay2, by2) -
                                                    max(ay1, by1) > 0)


def _greedy_poly_nms(polys: List[np.ndarray], scores: List[float],
                     iou_thr: float) -> Tuple[List[np.ndarray], List[float]]:
    if not polys:
        return [], []
    order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
    bboxes = [_bbox_from_poly(p) for p in polys]
    shapely_cache: Dict[int, object] = {}
    keep: List[int] = []
    for idx in order.tolist():
        suppressed = False
        for kept in keep:
            if not _bbox_overlap(bboxes[idx], bboxes[kept]):
                continue
            if idx not in shapely_cache:
                try:
                    shapely_cache[idx] = poly2shapely(polys[idx])
                except Exception:
                    suppressed = True
                    break
            if kept not in shapely_cache:
                try:
                    shapely_cache[kept] = poly2shapely(polys[kept])
                except Exception:
                    continue
            try:
                iou = float(poly_iou(shapely_cache[idx], shapely_cache[kept]))
            except Exception:
                iou = 0.0
            if iou > iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(idx)
    keep_polys = [polys[i] for i in keep]
    keep_scores = [float(scores[i]) for i in keep]
    return keep_polys, keep_scores


def _build_stage2_model(stage2_config: str, stage2_ckpt: str,
                        device: str) -> torch.nn.Module:
    cfg = Config.fromfile(stage2_config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, stage2_ckpt, map_location='cpu')
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _infer_stage2_patch(model: torch.nn.Module, patch_bgr: np.ndarray,
                        device: str) -> Tuple[List[np.ndarray], List[float]]:
    # patch_bgr: (H,W,3) uint8, BGR
    assert patch_bgr.ndim == 3 and patch_bgr.shape[2] == 3
    inputs = torch.from_numpy(patch_bgr).permute(2, 0, 1).contiguous()

    data_sample = TextDetDataSample()
    h, w = patch_bgr.shape[:2]
    data_sample.set_metainfo(
        dict(
            img_path='',
            ori_shape=(h, w),
            img_shape=(h, w),
            scale_factor=(1.0, 1.0),
        ))

    data = dict(inputs=[inputs], data_samples=[data_sample])
    outputs = model.test_step(data)
    if not outputs:
        return [], []
    out0 = outputs[0]
    pred_instances = out0.get('pred_instances')
    if pred_instances is None:
        return [], []
    polys = pred_instances.get('polygons', [])
    scores = pred_instances.get('scores', [])
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy().tolist()
    scores = [float(s) for s in list(scores)]

    poly_list: List[np.ndarray] = []
    score_list: List[float] = []
    for p, s in zip(polys, scores):
        arr = _as_poly_array(p)
        if arr is None:
            continue
        poly_list.append(arr.astype(np.float32))
        score_list.append(float(s))
    return poly_list, score_list


def _select_to_refine(
    polys: List[np.ndarray],
    scores: List[float],
    gating_mode: str,
    refine_ratio: float,
    score_low: float,
    score_high: float,
    aspect_ratio_thr: float,
    topk_per_image: int,
) -> Tuple[List[int], Dict[int, Dict]]:
    """Return indices of polys to refine (<= topk_per_image)."""
    stats: Dict[int, Dict] = {}
    candidates: List[int] = []
    for i, (p, s) in enumerate(zip(polys, scores)):
        w, h, ar = _min_area_rect_stats(p)
        area = _poly_area(p)
        stats[i] = dict(score=float(s), rect_w=w, rect_h=h, aspect_ratio=ar,
                        area=area)
        if (score_low <= float(s) <= score_high) or (ar > aspect_ratio_thr):
            candidates.append(i)

    if not candidates:
        return [], stats

    # difficulty: score closer to 0.5 -> harder
    candidates = sorted(
        candidates, key=lambda idx: abs(float(scores[idx]) - 0.5))

    if gating_mode == 'ratio':
        refine_ratio = float(np.clip(refine_ratio, 0.0, 1.0))
        if refine_ratio <= 0:
            selected: List[int] = []
        else:
            k = int(math.ceil(refine_ratio * len(candidates)))
            selected = candidates[:k]
    else:
        selected = candidates

    if topk_per_image > 0:
        selected = selected[:topk_per_image]
    return selected, stats


def _draw_polys(img_bgr: np.ndarray, polys: List[np.ndarray],
                color: Tuple[int, int, int]) -> np.ndarray:
    out = img_bgr.copy()
    for poly in polys:
        arr = _as_poly_array(poly)
        if arr is None:
            continue
        pts = arr.reshape(-1, 2).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
    return out


def main():
    args = parse_args()
    register_all_modules()

    out_pkl_path = Path(args.out_pkl)
    out_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_pkl_path.parent

    if (not args.dry_run_refine) and (not args.stage2_ckpt):
        raise ValueError(
            '未提供 --stage2-ckpt 且未开启 --dry-run-refine。'
            '请先在 configs/textdet/fcenet/README.md 中下载/准备 FCENet 权重，'
            '再通过 --stage2-ckpt 指定。')
    if (not args.dry_run_refine) and (not osp.isfile(args.stage2_ckpt)):
        raise FileNotFoundError(
            f'未找到 stage2 ckpt: {args.stage2_ckpt}。'
            '请检查路径，或参考 configs/textdet/fcenet/README.md 的官方下载链接。')

    preds = load(args.stage1_pred_pkl)
    if not isinstance(preds, list):
        raise TypeError('stage1 pkl 期望为 list[DataSample]')
    if args.max_images and args.max_images > 0:
        preds = preds[:args.max_images]

    stage2_model = None
    if not args.dry_run_refine:
        stage2_model = _build_stage2_model(args.stage2_config,
                                           args.stage2_ckpt, args.device)

    total_images = 0
    total_stage1 = 0
    total_candidates = 0
    total_selected = 0
    total_refine_success = 0
    total_refine_fallback = 0

    refined_per_image: List[int] = []
    stage2_time_s = 0.0
    stage2_patches = 0
    total_time_s = 0.0

    refined_outputs = []

    save_vis_max = 20
    vis_dir = out_dir / 'vis'
    patch_vis_dir = vis_dir / 'patches'
    if args.save_debug_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        patch_vis_dir.mkdir(parents=True, exist_ok=True)

    for img_idx, data_sample in enumerate(preds):
        t0 = time.perf_counter()
        img_path = _get_img_path(data_sample)
        if not img_path:
            raise ValueError('pkl 中缺少 metainfo.img_path')

        img = mmcv.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        pred_instances = data_sample.get('pred_instances')
        if pred_instances is None:
            raise ValueError('pkl 中缺少 pred_instances')

        polys_raw = pred_instances.get('polygons', [])
        scores_raw = pred_instances.get('scores', [])
        if isinstance(scores_raw, torch.Tensor):
            scores_raw = scores_raw.detach().cpu().numpy().tolist()

        pairs: List[Tuple[np.ndarray, float]] = []
        for p, s in zip(polys_raw, list(scores_raw)):
            arr = _as_poly_array(p)
            if arr is None:
                continue
            pairs.append((arr.astype(np.float32), float(s)))

        polys = [p for p, _ in pairs]
        scores = [s for _, s in pairs]
        total_images += 1
        total_stage1 += len(polys)

        selected, stats = _select_to_refine(
            polys=polys,
            scores=scores,
            gating_mode=args.gating_mode,
            refine_ratio=args.refine_ratio,
            score_low=args.score_low,
            score_high=args.score_high,
            aspect_ratio_thr=args.aspect_ratio_thr,
            topk_per_image=args.topk_per_image,
        )
        candidates_num = sum(
            1 for i in range(len(polys))
            if (args.score_low <= scores[i] <= args.score_high) or
            (stats[i]['aspect_ratio'] > args.aspect_ratio_thr))
        total_candidates += candidates_num
        total_selected += len(selected)

        refined_map: Dict[int, Tuple[np.ndarray, float]] = {}
        refine_success = 0
        refine_fallback = 0

        for i in selected:
            coarse_poly = polys[i]
            coarse_score = scores[i]

            try:
                patch, mat_o2p, mat_p2o, _ = _crop_rotated_patch(
                    img,
                    coarse_poly,
                    expand_ratio=args.expand_ratio,
                    max_long_edge=args.max_patch_long_edge,
                    pad_divisor=args.pad_divisor,
                )
            except Exception:
                refine_fallback += 1
                continue

            if args.dry_run_refine:
                patch_polys = _transform_polygons([coarse_poly], mat_o2p)
                patch_scores = [coarse_score]
            else:
                t_stage2 = time.perf_counter()
                patch_polys, patch_scores = _infer_stage2_patch(
                    stage2_model, patch, args.device)  # type: ignore[arg-type]
                stage2_time_s += (time.perf_counter() - t_stage2)
                stage2_patches += 1

            if not patch_polys:
                refine_fallback += 1
                continue

            if args.save_debug_vis and img_idx < save_vis_max:
                patch_base = patch.copy()
                patch_pred_vis = _draw_polys(
                    patch_base, patch_polys, color=(0, 255, 0))
                mmcv.imwrite(
                    patch, str(patch_vis_dir /
                               f'{img_idx:04d}_sel{i:03d}_patch.jpg'))
                mmcv.imwrite(
                    patch_pred_vis,
                    str(patch_vis_dir /
                        f'{img_idx:04d}_sel{i:03d}_patch_pred.jpg'))

            mapped_polys = _transform_polygons(patch_polys, mat_p2o)
            if not mapped_polys:
                refine_fallback += 1
                continue

            # choose the best polygon for this coarse instance: max IoU
            best_j = -1
            best_iou = -1.0
            for j, mp in enumerate(mapped_polys):
                iou = _safe_poly_iou(coarse_poly, mp)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j < 0 or best_iou < 0.1:
                refine_fallback += 1
                continue

            refined_poly = mapped_polys[best_j]
            refined_poly = _clip_polygon(refined_poly, w=img.shape[1],
                                         h=img.shape[0])

            area = _poly_area(refined_poly)
            # area sanity: too small or too large (relative to coarse rect area)
            rect_w = float(stats[i]['rect_w'])
            rect_h = float(stats[i]['rect_h'])
            rect_area = max(rect_w * rect_h, 1.0)
            if area < 50.0 or area > rect_area * 1.5:
                refine_fallback += 1
                continue

            # score: use stage2 score if available, otherwise inherit stage1
            refined_score = float(coarse_score)
            if best_j < len(patch_scores):
                refined_score = float(patch_scores[best_j])

            refined_map[i] = (refined_poly.astype(np.float32), refined_score)
            refine_success += 1

        total_refine_success += refine_success
        total_refine_fallback += refine_fallback
        refined_per_image.append(refine_success)

        merged_polys: List[np.ndarray] = []
        merged_scores: List[float] = []
        for i, (p, s) in enumerate(zip(polys, scores)):
            if i in refined_map:
                rp, rs = refined_map[i]
                merged_polys.append(rp)
                merged_scores.append(float(rs))
            else:
                merged_polys.append(p)
                merged_scores.append(float(s))

        keep_polys, keep_scores = _greedy_poly_nms(merged_polys, merged_scores,
                                                   args.nms_iou_thr)

        if isinstance(data_sample, dict):
            out_sample = copy.deepcopy(data_sample)
            out_sample['pred_instances'] = dict(
                polygons=[p.astype(np.float32) for p in keep_polys],
                scores=torch.tensor(keep_scores, dtype=torch.float32))
        else:
            out_sample = copy.deepcopy(data_sample)
            out_sample.pred_instances.polygons = [
                p.astype(np.float32) for p in keep_polys
            ]
            out_sample.pred_instances.scores = torch.tensor(
                keep_scores, dtype=torch.float32)
        refined_outputs.append(out_sample)

        if args.save_debug_vis and img_idx < save_vis_max:
            overlay = img.copy()
            overlay = _draw_polys(overlay, polys, color=(0, 0, 255))
            selected_polys = [polys[i] for i in selected] if selected else []
            overlay = _draw_polys(overlay, selected_polys, color=(0, 255, 255))
            overlay = _draw_polys(overlay, keep_polys, color=(0, 255, 0))
            mmcv.imwrite(overlay, str(vis_dir / f'{img_idx:04d}.jpg'))

        total_time_s += (time.perf_counter() - t0)

    dump(refined_outputs, str(out_pkl_path))

    refined_count_hist = Counter(refined_per_image)
    avg_ms_img = (total_time_s / max(total_images, 1)) * 1000.0
    avg_ms_patch = (stage2_time_s / max(stage2_patches, 1)) * 1000.0

    summary = dict(
        stage1_pred_pkl=osp.abspath(args.stage1_pred_pkl),
        out_pkl=osp.abspath(str(out_pkl_path)),
        stage2_config=osp.abspath(args.stage2_config),
        stage2_ckpt=osp.abspath(args.stage2_ckpt) if args.stage2_ckpt else '',
        dry_run_refine=bool(args.dry_run_refine),
        gating=dict(
            mode=args.gating_mode,
            refine_ratio=float(args.refine_ratio),
            score_low=float(args.score_low),
            score_high=float(args.score_high),
            aspect_ratio_thr=float(args.aspect_ratio_thr),
            topk_per_image=int(args.topk_per_image),
        ),
        patch=dict(
            expand_ratio=float(args.expand_ratio),
            max_patch_long_edge=int(args.max_patch_long_edge),
            pad_divisor=int(args.pad_divisor),
        ),
        nms=dict(iou_thr=float(args.nms_iou_thr)),
        stats=dict(
            total_images=int(total_images),
            total_stage1_instances=int(total_stage1),
            total_candidates=int(total_candidates),
            total_selected=int(total_selected),
            refine_success=int(total_refine_success),
            refine_fallback=int(total_refine_fallback),
            refined_per_image_hist=dict(sorted(refined_count_hist.items())),
        ),
        timing_ms=dict(
            total=float(total_time_s * 1000.0),
            avg_per_image=float(avg_ms_img),
            stage2_total=float(stage2_time_s * 1000.0),
            stage2_avg_per_patch=float(avg_ms_patch),
            stage2_patches=int(stage2_patches),
        ),
    )

    dump(summary, str(out_dir / 'refine_summary.json'))
    print(json.dumps(summary['stats'], ensure_ascii=False, indent=2))
    print(json.dumps(summary['timing_ms'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
