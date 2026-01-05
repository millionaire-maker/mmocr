#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from mmengine.fileio import load


def _get_polygons(data_sample: Any) -> List[np.ndarray]:
    pred_instances = None
    if isinstance(data_sample, dict):
        pred_instances = data_sample.get('pred_instances') or data_sample.get(
            'instances')
    elif hasattr(data_sample, 'get'):
        pred_instances = data_sample.get('pred_instances') or data_sample.get(
            'instances')
    if pred_instances is None:
        return []

    polys = []
    if isinstance(pred_instances, dict):
        polys = pred_instances.get('polygons', []) or []
    elif hasattr(pred_instances, 'get'):
        polys = pred_instances.get('polygons', []) or []
    out: List[np.ndarray] = []
    for p in list(polys):
        try:
            arr = np.array(p, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        out.append(arr)
    return out


def _parse_item(s: str) -> Tuple[str, str]:
    if '=' in s:
        name, path = s.split('=', 1)
        name = name.strip()
        path = path.strip()
        if not name:
            name = osp.basename(path)
        return name, path
    return osp.basename(s), s


def _summarize_one(pkl_path: str) -> Dict[str, Any]:
    preds = load(pkl_path)
    if not isinstance(preds, list):
        raise TypeError(f'pred pkl 期望为 list[DataSample]: {pkl_path}')

    per_image_counts: List[int] = []
    vertex_counter: Counter = Counter()
    invalid_polys = 0
    empty_polys = 0
    total_polys = 0

    for ds in preds:
        polys = _get_polygons(ds)
        per_image_counts.append(len(polys))
        total_polys += len(polys)
        for p in polys:
            if p.size == 0:
                empty_polys += 1
                continue
            if p.size < 6 or (p.size % 2 != 0):
                invalid_polys += 1
                continue
            v = int(p.size // 2)
            vertex_counter[v] += 1

    vertex_total = int(sum(vertex_counter.values()))
    if vertex_total > 0:
        avg_vertices = float(
            sum(v * c for v, c in vertex_counter.items()) / vertex_total)
        ratio_4 = float(vertex_counter.get(4, 0) / vertex_total)
        ratio_gt4 = float(
            sum(c for v, c in vertex_counter.items() if v > 4) / vertex_total)
    else:
        avg_vertices = 0.0
        ratio_4 = 0.0
        ratio_gt4 = 0.0

    return dict(
        pkl=osp.abspath(pkl_path),
        num_images=len(preds),
        total_polygons=int(total_polys),
        avg_polygons_per_image=float(total_polys / max(len(preds), 1)),
        polygons_per_image=per_image_counts,
        vertex_stats=dict(
            total_valid_polygons=vertex_total,
            invalid_polygons=int(invalid_polys),
            empty_polygons=int(empty_polys),
            avg_vertices=avg_vertices,
            ratio_4=ratio_4,
            ratio_gt4=ratio_gt4,
            vertex_count_distribution=dict(
                sorted(((str(k), int(v)) for k, v in vertex_counter.items()),
                       key=lambda kv: int(kv[0]))),
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='统计 pkl 里 polygons 的顶点数分布/每图数量')
    parser.add_argument(
        '--pkl',
        action='append',
        required=True,
        help='输入 pkl，格式：name=/abs/path/to.pkl 或 /abs/path/to.pkl；可重复传入')
    parser.add_argument(
        '--out-json',
        default='vertex_stats.json',
        help='输出 json 路径（默认写到当前目录）')
    return parser.parse_args()


def main():
    args = parse_args()
    items = [_parse_item(s) for s in args.pkl]

    out: Dict[str, Any] = dict(items={})
    for name, p in items:
        out['items'][name] = _summarize_one(p)

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print('WROTE', str(out_path))


if __name__ == '__main__':
    main()

