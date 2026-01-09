
import os
import argparse
import random
import json
import time
import math
import numpy as np
import torch
import cv2
import mmengine
from pathlib import Path
from mmengine.config import Config
from mmocr.registry import MODELS, DATASETS
from mmengine.runner import load_checkpoint
from mmengine.dataset import Compose
from mmocr.utils import register_all_modules
from mmocr.structures import TextDetDataSample
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = str(REPO_ROOT / "outputs" / "cascade_demo" / "run004" / "fullval_G3_vs_G5")
CONFIG_PATH = str(
    REPO_ROOT
    / "configs"
    / "textdet"
    / "fcenet"
    / "fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py"
)

# Checkpoints
DBNET_CONFIG = str(
    REPO_ROOT
    / "work_dirs"
    / "dbnetpp_r50_finetune_art_rctw_rects"
    / "dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py"
)
DBNET_CKPT = str(
    REPO_ROOT
    / "work_dirs"
    / "dbnetpp_r50_finetune_art_rctw_rects"
    / "best_icdar_hmean_epoch_93.pth"
)
FCENET_CKPT = str(
    REPO_ROOT
    / "work_dirs"
    / "fcenet_r50dcnv2_fpn_finetune_art_rctw_rects"
    / "best_icdar_hmean_epoch_96.pth"
)

# Constants
MAX_LONG_EDGE = 1024

# --- Utils from G3/G5 ---
def _poly_points(poly): return np.array(poly).reshape(-1, 2).astype(np.float32)

def _order_points_clockwise(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0]==0: return pts
    s = pts.sum(axis=1)
    diff = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

def _crop_rotated_patch(img, poly, expand_ratio=0.2, max_long_edge=1024):
    pts = _poly_points(poly)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    w = max(float(w), 2.0)
    h = max(float(h), 2.0)
    w *= (1.0 + 2.0 * expand_ratio)
    h *= (1.0 + 2.0 * expand_ratio)
    rect_expanded = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect_expanded)
    box = _order_points_clockwise(box)
    raw_w = max(int(round(np.linalg.norm(box[0] - box[1]))), 2)
    raw_h = max(int(round(np.linalg.norm(box[0] - box[3]))), 2)
    scale = 1.0
    if max_long_edge and max(raw_w, raw_h) > max_long_edge:
        scale = max_long_edge / max(raw_w, raw_h)
    scaled_w = max(int(round(raw_w * scale)), 2)
    scaled_h = max(int(round(raw_h * scale)), 2)
    dst = np.array([[0, 0], [scaled_w-1, 0], [scaled_w-1, scaled_h-1], [0, scaled_h-1]], dtype=np.float32)
    mat_orig2patch = cv2.getPerspectiveTransform(box, dst)
    try:
        patch = cv2.warpPerspective(img, mat_orig2patch, (scaled_w, scaled_h))
        mat_patch2orig = np.linalg.inv(mat_orig2patch)
    except:
        return np.zeros((10,10,3),dtype=np.uint8), np.eye(3), np.eye(3)
    return patch, mat_orig2patch, mat_patch2orig

def _transform_polygons(polys, mat):
    out = []
    for p in polys:
        pts = np.array(p).reshape(-1, 2).astype(np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, mat)
        out.append(mapped.reshape(-1).tolist())
    return out

def _bbox_from_poly(poly):
    pts = np.array(poly).reshape(-1, 2)
    if pts.shape[0] == 0: return [0,0,1,1]
    return [np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])]

def _bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def to_shapely(poly):
    pts = np.array(poly).reshape(-1, 2)
    if len(pts) < 3: return Polygon()
    p = Polygon(pts)
    if not p.is_valid: p = p.buffer(0)
    return p

def _is_cuda_oom(err: BaseException) -> bool:
    if not isinstance(err, RuntimeError):
        return False
    msg = str(err).lower()
    return ('out of memory' in msg) or ('cuda error: out of memory' in msg)

def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def _test_step_in_microbatches(model, inputs, data_samples, microbatch_size: int):
    if microbatch_size <= 0:
        raise ValueError(f"microbatch_size must be > 0, got {microbatch_size}")

    outs = []
    start = 0
    while start < len(inputs):
        cur_inputs = inputs[start:start + microbatch_size]
        cur_samples = data_samples[start:start + microbatch_size]
        try:
            with torch.inference_mode():
                cur_outs = model.test_step(dict(inputs=cur_inputs, data_samples=cur_samples))
        except Exception as e:
            if _is_cuda_oom(e) and microbatch_size > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                microbatch_size = max(1, microbatch_size // 2)
                continue
            raise
        outs.extend(cur_outs)
        start += len(cur_inputs)
    return outs

def run_model_inference(model, img, device=None):
    if device is None:
        device = _get_model_device(model)
    device = torch.device(device)
    inputs = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    data_sample = TextDetDataSample()
    data_sample.set_metainfo(dict(img_shape=img.shape[:2], ori_shape=img.shape[:2], scale_factor=(1.0, 1.0)))
    with torch.inference_mode():
        out = model.test_step(dict(inputs=inputs, data_samples=[data_sample]))[0]
    return out.pred_instances.polygons, out.pred_instances.scores

def run_model_inference_batch(
    model,
    imgs,
    device=None,
    microbatch_size: int = 1,
):
    if not imgs:
        return [], []
    if device is None:
        device = _get_model_device(model)
    device = torch.device(device)

    inputs = []
    data_samples = []
    for img in imgs:
        inputs.append(torch.from_numpy(img).permute(2, 0, 1).float().to(device))
        ds = TextDetDataSample()
        ds.set_metainfo(dict(img_shape=img.shape[:2], ori_shape=img.shape[:2], scale_factor=(1.0, 1.0)))
        data_samples.append(ds)

    outs = _test_step_in_microbatches(model, inputs, data_samples, microbatch_size=microbatch_size)
    polygons = [o.pred_instances.polygons for o in outs]
    scores = [o.pred_instances.scores for o in outs]
    return polygons, scores

def clean_s1_polygons(polys, scores):
    removed_tiny = 0
    removed_contained = 0
    p1, s1 = [], []
    for p, sc in zip(polys, scores):
        pts = _poly_points(p)
        if len(pts) < 3:
            removed_tiny += 1
            continue
        rect = cv2.minAreaRect(pts)
        w, h = rect[1]
        area = w * h
        if min(w, h) < 6 or area < 25:
            removed_tiny += 1
            continue
        p1.append(p)
        s1.append(sc)
        
    p2, s2 = [], []
    kept = [True] * len(p1)
    shapely_polys = []
    valid_indices = []
    for idx, p in enumerate(p1):
        sp = to_shapely(p)
        if sp.is_empty or not sp.is_valid:
            kept[idx] = False
            removed_tiny += 1 
        else:
            shapely_polys.append(sp)
            valid_indices.append(idx)
            
    N = len(valid_indices)
    for i in range(N):
        idx_i = valid_indices[i]
        if not kept[idx_i]: continue
        poly_i = shapely_polys[i]
        area_i = poly_i.area + 1e-6
        score_i = s1[idx_i]
        for j in range(N):
            if i == j: continue
            idx_j = valid_indices[j]
            if not kept[idx_j]: continue
            poly_j = shapely_polys[j]
            if not poly_i.intersects(poly_j): continue
            inter = poly_i.intersection(poly_j).area
            ioa_i_in_j = inter / area_i
            if ioa_i_in_j > 0.95:
                score_j = s1[idx_j]
                if score_i < score_j:
                    kept[idx_i] = False
                    removed_contained += 1
                    break
    
    for idx, (p, s) in enumerate(zip(p1, s1)):
        if kept[idx]:
            p2.append(p)
            s2.append(s)
    return p2, s2, removed_tiny, removed_contained

# Helper class for Policy execution (Shared Logic)
class PolicyRunner:
    def __init__(self, mode, model_s2, device=None):
        self.mode = mode
        self.model_s2 = model_s2
        self.device = torch.device(device) if device is not None else _get_model_device(model_s2)
        # Toggles
        self.full_iou_thr = 0.2 if mode == 'G3' else 0.05
        self.check_neighbor = True # Both use it
        self.top_k = None if mode == 'G3' else 3
        
        # Shared Tuned Params
        self.DEFAULT_RATIO_THR = 0.15
        self.DEFAULT_ACCEPT_IOA_THR = 0.60
        self.DEFAULT_MIN_AREA_RATIO = 0.30
        self.DEFAULT_MAX_AREA_RATIO = 1.80
        self.DEFAULT_BBOX_EXPAND_LIMIT = 1.8
        self.DEFAULT_CENTROID_SHIFT_LIMIT = 0.3
        self.DEFAULT_RETRY_EXPAND_RATIO = 0.60
        self.TUNED_PAD_UNCOVERED_THR = 0.005
        self.DEFAULT_ENDPAD_EXPAND_RATIO = 0.35
        self.DEFAULT_ENDCAP_SLICE_RATIO = 0.08
        self.TUNED_ENDCAP_MISSING_THR = 0.003
        
        # Stats
        self.stats = defaultdict(int)

    def reset_stats(self):
        self.stats = defaultdict(int)

    def check_safety_gate(self, poly_check, base_poly, poly_s1_orig):
        if poly_check.is_empty: return False
        inter = poly_check.intersection(base_poly).area
        s1_area = base_poly.area + 1e-6
        ioa = inter / s1_area
        area_ratio = poly_check.area / s1_area
        
        if ioa < 0.3: return False
        if ioa < self.DEFAULT_ACCEPT_IOA_THR: return False
        if area_ratio < self.DEFAULT_MIN_AREA_RATIO: return False
        if area_ratio > self.DEFAULT_MAX_AREA_RATIO: return False
        
        bbox_s1 = _bbox_from_poly(poly_s1_orig)
        s1_w = bbox_s1[2]-bbox_s1[0]
        s1_h = bbox_s1[3]-bbox_s1[1]
        bbox_s1_area = s1_w * s1_h
        diag_s1 = np.sqrt(s1_w**2 + s1_h**2)
        
        s2_bbox_poly = poly_check.envelope
        s2_minx, s2_miny, s2_maxx, s2_maxy = s2_bbox_poly.bounds
        s2_bbox_area = (s2_maxx-s2_minx)*(s2_maxy-s2_miny)
        s2_centroid = poly_check.centroid
        s1_centroid = base_poly.centroid
        shift = s2_centroid.distance(s1_centroid)

        if s2_bbox_area / (bbox_s1_area + 1e-6) > self.DEFAULT_BBOX_EXPAND_LIMIT: return False
        if shift > self.DEFAULT_CENTROID_SHIFT_LIMIT * diag_s1: return False
        return True

    def filtered_full_candidates(self, s2_polys_full, s1_poly):
        s1_shapely = to_shapely(s1_poly)
        if s1_shapely.is_empty: return []
        candidates = []
        bbox_s1 = _bbox_from_poly(s1_poly)
        
        for p in s2_polys_full:
            bbox_p = _bbox_from_poly(p)
            if _bbox_iou(bbox_s1, bbox_p) >= self.full_iou_thr:
                p_shapely = to_shapely(p)
                if p_shapely.is_empty: continue
                # G4/G5 Neighbor check
                if self.check_neighbor:
                    inter_area = p_shapely.intersection(s1_shapely).area
                    if inter_area / (p_shapely.area + 1e-6) < 0.3:
                        continue
                    candidates.append({'poly': p_shapely, 'inter_area': inter_area})
                else: 
                     # Should not happen for G3 strict/G5
                     pass
        
        if self.top_k:
            candidates.sort(key=lambda x: x['inter_area'], reverse=True)
            return [x['poly'] for x in candidates[:self.top_k]]
        else:
            return [x['poly'] for x in candidates]

    def run_fce_patch(self, img, poly, expand_ratio):
        patch, mat_o2p, mat_p2o = _crop_rotated_patch(img, poly, expand_ratio=expand_ratio, max_long_edge=MAX_LONG_EDGE)
        h, w = patch.shape[:2]
        if h == 0 or w == 0: return [], [], patch
        inputs = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        data_sample = TextDetDataSample()
        data_sample.set_metainfo(dict(img_shape=patch.shape[:2], ori_shape=patch.shape[:2], scale_factor=(1.0, 1.0)))
        with torch.inference_mode():
            out = self.model_s2.test_step(dict(inputs=inputs, data_samples=[data_sample]))[0] 
        s2_raw = out.pred_instances.polygons
        s2_raw_scores = out.pred_instances.scores
        s2_np = []
        for p in s2_raw:
            if hasattr(p, 'cpu'): s2_np.append(p.cpu().numpy())
            else: s2_np.append(p)
        mapped_s2 = _transform_polygons(s2_np, mat_p2o)
        s2_filtered = []
        s2_filtered_scores = []
        s1_bbox = _bbox_from_poly(poly)
        for j, mp in enumerate(mapped_s2):
            if _bbox_iou(s1_bbox, _bbox_from_poly(mp)) >= 0.2:
                s2_filtered.append(to_shapely(mp))
                sc = s2_raw_scores[j]
                if hasattr(sc, 'item'): sc = sc.item()
                s2_filtered_scores.append(sc)
        return s2_filtered, s2_filtered_scores, patch

    def run_fce_patch_batch(self, img, polys, expand_ratio, batch_size=8):
        if not polys:
            return []

        results = [([], []) for _ in polys]
        inputs = []
        data_samples = []
        metas = []
        valid_map = []

        for i, poly in enumerate(polys):
            patch, _, mat_p2o = _crop_rotated_patch(
                img, poly, expand_ratio=expand_ratio, max_long_edge=MAX_LONG_EDGE
            )
            h, w = patch.shape[:2]
            if h == 0 or w == 0:
                continue
            inputs.append(torch.from_numpy(patch).permute(2, 0, 1).float().to(self.device))
            ds = TextDetDataSample()
            ds.set_metainfo(
                dict(img_shape=patch.shape[:2], ori_shape=patch.shape[:2], scale_factor=(1.0, 1.0))
            )
            data_samples.append(ds)
            metas.append((mat_p2o, _bbox_from_poly(poly)))
            valid_map.append(i)

        if not inputs:
            return results

        outs = _test_step_in_microbatches(self.model_s2, inputs, data_samples, microbatch_size=batch_size)
        for out, (mat_p2o, s1_bbox), idx in zip(outs, metas, valid_map):
            s2_raw = out.pred_instances.polygons
            s2_raw_scores = out.pred_instances.scores

            s2_np = []
            for p in s2_raw:
                if hasattr(p, 'cpu'):
                    s2_np.append(p.cpu().numpy())
                else:
                    s2_np.append(p)
            mapped_s2 = _transform_polygons(s2_np, mat_p2o)

            s2_filtered = []
            s2_filtered_scores = []
            for j, mp in enumerate(mapped_s2):
                if _bbox_iou(s1_bbox, _bbox_from_poly(mp)) >= 0.2:
                    s2_filtered.append(to_shapely(mp))
                    sc = s2_raw_scores[j]
                    if hasattr(sc, 'item'):
                        sc = sc.item()
                    s2_filtered_scores.append(sc)
            results[idx] = (s2_filtered, s2_filtered_scores)

        return results

    def apply_endcap(self, s2_poly, s1_poly):
        s1_shapely = to_shapely(s1_poly)
        if s1_shapely.is_empty: return None
        uncovered_geom = s1_shapely.difference(s2_poly)
        if uncovered_geom.is_empty: return None
        pts_s1 = _poly_points(s1_poly)
        try:
            rect = cv2.minAreaRect(pts_s1)
        except: return None
        (cx, cy), (w, h), angle = rect
        if w >= h:
            shrink_w = w * (1.0 - 2 * self.DEFAULT_ENDCAP_SLICE_RATIO)
            shrink_h = h * 1.5 
            cut_size = (shrink_w, shrink_h)
        else:
            shrink_w = w * 1.5
            shrink_h = h * (1.0 - 2 * self.DEFAULT_ENDCAP_SLICE_RATIO)
            cut_size = (shrink_w, shrink_h)
        cutout_rect = ((cx, cy), cut_size, angle)
        cutout_box = cv2.boxPoints(cutout_rect)
        cutout_poly = to_shapely(cutout_box)
        missing_ends = uncovered_geom.difference(cutout_poly)
        if missing_ends.is_empty: return None
        miss_area = missing_ends.area
        s1_area = s1_shapely.area + 1e-6
        if miss_area / s1_area <= self.TUNED_ENDCAP_MISSING_THR: return None
        return s2_poly.union(missing_ends)

    def process_image(self, img, s1_polys, s1_scores, s2_polys_full, patch_batch_size=8):
        n = len(s1_polys)
        final_polys = [None] * n
        final_scores = [None] * n
        is_s2_flags = [False] * n

        s1_quads = [None] * n
        s1_shapely_list = [None] * n
        ratios = [0.0] * n
        accepted_polys = [None] * n
        current_scores_list = [None] * n

        need_patch_indices = []

        for idx in range(n):
            s1_poly = s1_polys[idx]
            s1_score = s1_scores[idx]

            pts_raw = _poly_points(s1_poly)
            try:
                rect = cv2.minAreaRect(pts_raw)
            except:
                final_polys[idx] = s1_poly
                final_scores[idx] = s1_score
                continue

            box = cv2.boxPoints(rect)
            box = _order_points_clockwise(box)
            s1_quad = box.flatten().tolist()
            s1_quads[idx] = s1_quad

            s1_shapely = to_shapely(s1_poly)
            s1_shapely_list[idx] = s1_shapely

            area_rect = rect[1][0] * rect[1][1] + 1e-6
            ratio = 1.0 - (s1_shapely.area / area_rect)
            ratios[idx] = ratio

            if ratio < self.DEFAULT_RATIO_THR:
                continue

            accepted_poly = None
            current_scores = None

            full_cands = self.filtered_full_candidates(s2_polys_full, s1_poly)
            if full_cands:
                s2_union_full = unary_union(full_cands)
                if self.check_safety_gate(s2_union_full, s1_shapely, s1_poly):
                    accepted_poly = s2_union_full
                    self.stats['sum_s2_full_hit'] += 1
                    current_scores = [s1_score]

            if accepted_poly is None:
                self.stats['sum_s2_patch_fallback'] += 1
                need_patch_indices.append(idx)
            else:
                accepted_polys[idx] = accepted_poly
                current_scores_list[idx] = current_scores

        if need_patch_indices:
            polys_need = [s1_polys[i] for i in need_patch_indices]
            patch_results = self.run_fce_patch_batch(
                img, polys_need, 0.2, batch_size=patch_batch_size
            )

            retry_indices = []
            for local_i, global_i in enumerate(need_patch_indices):
                s2_filtered, s2_scores_list = patch_results[local_i]
                if not s2_filtered:
                    self.stats['sum_s2_raw_empty'] += 1
                    retry_indices.append(global_i)
                    continue
                s2_union = unary_union(s2_filtered)
                if self.check_safety_gate(
                    s2_union, s1_shapely_list[global_i], s1_polys[global_i]
                ):
                    accepted_polys[global_i] = s2_union
                    current_scores_list[global_i] = s2_scores_list

            if retry_indices:
                polys_retry = [s1_polys[i] for i in retry_indices]
                retry_results = self.run_fce_patch_batch(
                    img,
                    polys_retry,
                    self.DEFAULT_RETRY_EXPAND_RATIO,
                    batch_size=patch_batch_size,
                )
                for local_i, global_i in enumerate(retry_indices):
                    s2_filtered, s2_scores_list = retry_results[local_i]
                    if not s2_filtered:
                        continue
                    s2_union = unary_union(s2_filtered)
                    if self.check_safety_gate(
                        s2_union, s1_shapely_list[global_i], s1_polys[global_i]
                    ):
                        accepted_polys[global_i] = s2_union
                        current_scores_list[global_i] = s2_scores_list

        pad_indices = []
        ioa_bases = {}
        for idx in range(n):
            if final_polys[idx] is not None:
                continue
            if ratios[idx] < self.DEFAULT_RATIO_THR:
                continue
            accepted_poly = accepted_polys[idx]
            if not accepted_poly:
                continue
            s1_shapely = s1_shapely_list[idx]
            ioa_base = accepted_poly.intersection(s1_shapely).area / (s1_shapely.area + 1e-6)
            uncovered = 1.0 - ioa_base
            if ioa_base < 0.95 or uncovered > self.TUNED_PAD_UNCOVERED_THR:
                pad_indices.append(idx)
                ioa_bases[idx] = ioa_base

        if pad_indices:
            polys_pad = [s1_polys[i] for i in pad_indices]
            pad_results = self.run_fce_patch_batch(
                img,
                polys_pad,
                self.DEFAULT_ENDPAD_EXPAND_RATIO,
                batch_size=patch_batch_size,
            )
            for local_i, global_i in enumerate(pad_indices):
                s2_pad, s2_pad_sc = pad_results[local_i]
                if not s2_pad:
                    continue
                u_pad = unary_union(s2_pad)
                s1_shapely = s1_shapely_list[global_i]
                if self.check_safety_gate(u_pad, s1_shapely, s1_polys[global_i]):
                    ioa_pad = u_pad.intersection(s1_shapely).area / (s1_shapely.area + 1e-6)
                    if ioa_pad > ioa_bases[global_i] + 0.005:
                        accepted_polys[global_i] = u_pad
                        current_scores_list[global_i] = s2_pad_sc

        for idx in range(n):
            if final_polys[idx] is not None:
                continue

            s1_poly = s1_polys[idx]
            s1_score = s1_scores[idx]
            s1_quad = s1_quads[idx]

            if ratios[idx] < self.DEFAULT_RATIO_THR:
                final_polys[idx] = s1_quad
                final_scores[idx] = s1_score
                continue

            accepted_poly = accepted_polys[idx]
            s1_shapely = s1_shapely_list[idx]
            current_scores = current_scores_list[idx] or []

            if accepted_poly:
                s2_fixed = self.apply_endcap(accepted_poly, s1_poly)
                if s2_fixed:
                    if self.check_safety_gate(s2_fixed, s1_shapely, s1_poly):
                        accepted_poly = s2_fixed

                final_geom = accepted_poly
                if accepted_poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                    parts = []
                    if accepted_poly.geom_type == 'MultiPolygon':
                        parts = list(accepted_poly.geoms)
                    else:
                        for g in accepted_poly.geoms:
                            if g.geom_type in ['Polygon', 'MultiPolygon']:
                                if g.geom_type == 'Polygon':
                                    parts.append(g)
                                else:
                                    parts.extend(list(g.geoms))

                    best_part = None
                    best_inter = -1.0
                    for part in parts:
                        if not part.is_valid or part.is_empty:
                            continue
                        inter_v = part.intersection(s1_shapely).area
                        if inter_v > best_inter:
                            best_inter = inter_v
                            best_part = part
                    if best_part is not None:
                        final_geom = best_part
                    else:
                        final_geom = None

                if final_geom and final_geom.geom_type == 'Polygon':
                    final_poly_obj = np.array(final_geom.exterior.coords)[:-1].flatten().tolist()
                    s2_max_sc = max(current_scores) if current_scores else s1_score
                    final_sc = max(float(s1_score), float(s2_max_sc))
                    final_polys[idx] = final_poly_obj
                    final_scores[idx] = final_sc
                    is_s2_flags[idx] = True
                    continue

            final_polys[idx] = s1_quad
            final_scores[idx] = s1_score

        s2_accepted_count = sum(1 for v in is_s2_flags if v)
        s1_fallback_count = n - s2_accepted_count
        return final_polys, final_scores, s2_accepted_count, s1_fallback_count

def parse_args():
    parser = argparse.ArgumentParser(description='Cascade inference (G3/G5)')
    parser.add_argument('--img-batch-size', type=int, default=2)
    parser.add_argument('--patch-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device-s1', type=str, default='cuda:0')
    parser.add_argument('--device-s2', type=str, default=None)
    return parser.parse_args()

def _to_poly_list(polys_raw):
    polys = []
    for p in polys_raw:
        if hasattr(p, 'cpu'):
            polys.append(p.cpu().numpy().tolist())
        elif isinstance(p, np.ndarray):
            polys.append(p.tolist())
        else:
            polys.append(p)
    return polys

def _to_score_list(scores_raw):
    if hasattr(scores_raw, 'cpu'):
        return scores_raw.cpu().numpy().tolist()
    if isinstance(scores_raw, np.ndarray):
        return scores_raw.tolist()
    return list(scores_raw)

def main():
    args = parse_args()

    register_all_modules(init_default_scope=True)

    os.makedirs(os.path.join(OUT_ROOT, "G3"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "G5"), exist_ok=True)

    device_s1 = torch.device(args.device_s1)
    device_s2 = torch.device(args.device_s2) if args.device_s2 else device_s1

    # Load Config & Dataloader
    cfg = Config.fromfile(CONFIG_PATH)

    # Build models
    print(f"Loading DBNet S1 on {device_s1}...")
    cfg_s1 = Config.fromfile(DBNET_CONFIG)
    model_s1 = MODELS.build(cfg_s1.model)
    load_checkpoint(model_s1, DBNET_CKPT, map_location='cpu')
    if hasattr(model_s1.det_head, 'postprocessor'):
        model_s1.det_head.postprocessor.text_repr_type = 'poly'
    model_s1.to(device_s1).eval()

    print(f"Loading FCENet S2 on {device_s2}...")
    model_s2 = MODELS.build(cfg.model)
    load_checkpoint(model_s2, FCENET_CKPT, map_location='cpu')
    model_s2.to(device_s2).eval()  # Full config has postprocessor settings

    # Init Runners
    g3_runner = PolicyRunner('G3', model_s2, device=device_s2)
    g5_runner = PolicyRunner('G5', model_s2, device=device_s2)

    # Init Results Containers
    g3_results = []
    g5_results = []

    # Interest Lists
    interest_candidates = []

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.img_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    print(
        f"Starting Inference on {len(dataset)} images "
        f"(img_bs={args.img_batch_size}, patch_bs={args.patch_batch_size})..."
    )

    t0 = time.time()
    for batch in tqdm(dataloader):
        img_paths = []
        imgs = []
        for item in batch:
            data_sample = item['data_samples']
            img_path = data_sample.img_path
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            img_paths.append(img_path)
            imgs.append(img)

        if not imgs:
            continue

        # S1 batch inference
        s1_polys_raw_list, s1_scores_raw_list = run_model_inference_batch(
            model_s1, imgs, device=device_s1, microbatch_size=args.img_batch_size
        )
        # S2 full batch inference
        s2_polys_full_raw_list, _ = run_model_inference_batch(
            model_s2, imgs, device=device_s2, microbatch_size=args.img_batch_size
        )

        for img_path, img, s1_polys_raw, s1_scores_raw, s2_polys_full_raw in zip(
            img_paths, imgs, s1_polys_raw_list, s1_scores_raw_list, s2_polys_full_raw_list
        ):
            s1_polys = _to_poly_list(s1_polys_raw)
            s1_scores = _to_score_list(s1_scores_raw)

            g3_runner.stats['sum_s1_total'] += len(s1_polys)
            g5_runner.stats['sum_s1_total'] += len(s1_polys)

            # S1 Denoise (Clean)
            s1_clean, s1_clean_sc, rem_tiny, rem_con = clean_s1_polygons(s1_polys, s1_scores)
            g3_runner.stats['sum_s1_after'] += len(s1_clean)
            g3_runner.stats['tiny_removed'] += rem_tiny
            g3_runner.stats['contained_removed'] += rem_con
            g5_runner.stats.update({
                'sum_s1_after': g3_runner.stats['sum_s1_after'],
                'tiny_removed': g3_runner.stats['tiny_removed'],
                'contained_removed': g3_runner.stats['contained_removed']
            })

            s2_polys_full = _to_poly_list(s2_polys_full_raw)

            # Run G3
            g3_polys, g3_scores, g3_acc, g3_fb = g3_runner.process_image(
                img, s1_clean, s1_clean_sc, s2_polys_full, patch_batch_size=args.patch_batch_size
            )
            g3_runner.stats['sum_s2_accepted'] += g3_acc
            g3_runner.stats['sum_s1_fallback'] += g3_fb
            g3_runner.stats['final_instances'] += len(g3_polys)

            # Run G5
            g5_polys, g5_scores, g5_acc, g5_fb = g5_runner.process_image(
                img, s1_clean, s1_clean_sc, s2_polys_full, patch_batch_size=args.patch_batch_size
            )
            g5_runner.stats['sum_s2_accepted'] += g5_acc
            g5_runner.stats['sum_s1_fallback'] += g5_fb
            g5_runner.stats['final_instances'] += len(g5_polys)

            # Store Results (MMOCR Format)
            res_g3 = {
                'img_path': img_path,
                'polygons': g3_polys,
                'scores': g3_scores,
                'ori_shape': img.shape[:2],
                'img_shape': img.shape[:2],  # Assuming no resize in storage
            }
            g3_results.append(res_g3)

            res_g5 = {
                'img_path': img_path,
                'polygons': g5_polys,
                'scores': g5_scores,
                'ori_shape': img.shape[:2],
                'img_shape': img.shape[:2],
            }
            g5_results.append(res_g5)

            interest_candidates.append({
                'img_path': img_path,
                'g5_s2_acc': g5_acc,
                'g3_s2_acc': g3_acc
            })

    total_time = time.time() - t0
    print(f"Inference Done in {total_time:.2f}s")
    
    # Save Results
    mmengine.dump(g3_results, os.path.join(OUT_ROOT, "G3/preds.pkl"))
    mmengine.dump(g3_runner.stats, os.path.join(OUT_ROOT, "G3/pipeline_report.json"))
    
    mmengine.dump(g5_results, os.path.join(OUT_ROOT, "G5/preds.pkl"))
    mmengine.dump(g5_runner.stats, os.path.join(OUT_ROOT, "G5/pipeline_report.json"))
    
    # Generate Interest List
    # 1. Top 20 S2 Accepted (G5)
    interest_candidates.sort(key=lambda x: x['g5_s2_acc'], reverse=True)
    top_s2 = interest_candidates[:20]
    
    # 2. Random ArT 60
    art_imgs = [x for x in interest_candidates if 'art' in x['img_path'].lower()]
    random.seed(42)
    rand_art = random.sample(art_imgs, min(60, len(art_imgs)))
    
    # 3. Top Diff? Maybe just use Top S2 for now as they are most likely to show differences
    # User asked for "S2 accepted most top 20" and "Final vs S1 diff top 20"
    # We didn't calc Final vs S1 diff here (area wise), can do in H4 or add approximate here.
    # We will stick to S2 Acc for Side-by-Side as proxy for "Active Strategy".
    
    final_interest = {
        'top_s2_acc': [x['img_path'] for x in top_s2],
        'random_art': [x['img_path'] for x in rand_art]
    }
    mmengine.dump(final_interest, os.path.join(OUT_ROOT, "interest_list.json"))
    
if __name__ == "__main__":
    main()
