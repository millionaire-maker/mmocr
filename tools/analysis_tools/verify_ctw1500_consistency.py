#!/usr/bin/env python3
"""Compare CTW1500 detection and spotting annotations.

This script checks whether the detection annotations in
``data/ctw1500/textdet_*.json`` are consistent with the detection
information embedded in ``data/ctw1500/textspotting_*.json``.

Usage:
    python tools/analysis_tools/verify_ctw1500_consistency.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

def _round_sequence(values: Sequence[float], precision: int = 2) -> Tuple[float, ...]:
    """Round all numeric entries to stabilise floating point noise."""
    return tuple(round(float(v), precision) for v in values)


def _collect_instances(instances: Iterable[dict], precision: int) -> Dict[str, List]:
    """Collect polygons, bboxes and ignore flags from annotations."""
    data = defaultdict(list)
    for inst in instances:
        if 'polygon' in inst:
            data['polygons'].append(_round_sequence(inst['polygon'], precision))
            data['polygons_raw'].append(list(inst['polygon']))
        if 'bbox' in inst:
            data['bboxes'].append(_round_sequence(inst['bbox'], precision))
            data['bboxes_raw'].append(list(inst['bbox']))
        if 'ignore' in inst:
            data['ignore'].append(bool(inst['ignore']))
    for key in ('polygons', 'polygons_raw', 'bboxes', 'bboxes_raw', 'ignore'):
        data[key]
    return data


def _load_annotations(path: Path, precision: int) -> Dict[str, dict]:
    """Load annotations and return a mapping from image path to instance info."""
    with path.open('r') as f:
        raw = json.load(f)
    mapping = {}
    for item in raw.get('data_list', []):
        mapping[item['img_path']] = _collect_instances(
            item.get('instances', []), precision=precision)
    return mapping


def _compare_split(
        det_path: Path,
        spot_path: Path,
        precision: int = 2,
        sample_count: int = 3) -> dict:
    det_map = _load_annotations(det_path, precision=precision)
    spot_map = _load_annotations(spot_path, precision=precision)

    det_keys = set(det_map.keys())
    spot_keys = set(spot_map.keys())
    shared = det_keys & spot_keys

    result = {
        'det_images': len(det_keys),
        'spot_images': len(spot_keys),
        'shared_images': len(shared),
        'det_only': sorted(det_keys - spot_keys),
        'spot_only': sorted(spot_keys - det_keys),
        'count_mismatch': [],
        'polygon_mismatch': [],
        'polygon_samples': [],
        'bbox_mismatch': [],
        'bbox_samples': [],
        'ignore_mismatch': [],
        'ignore_samples': [],
        'count_samples': [],
    }

    for key in shared:
        det_instances = det_map[key]
        spot_instances = spot_map[key]

        if len(det_instances['polygons']) != len(spot_instances['polygons']):
            result['count_mismatch'].append(key)
            if len(result['count_samples']) < sample_count:
                result['count_samples'].append({
                    'img_path': key,
                    'det_count': len(det_instances['polygons']),
                    'spot_count': len(spot_instances['polygons']),
                    'det_polygons': det_instances['polygons_raw'],
                    'spot_polygons': spot_instances['polygons_raw'],
                    'det_bboxes': det_instances['bboxes_raw'],
                    'spot_bboxes': spot_instances['bboxes_raw'],
                    'category': 'count',
                })
            continue

        # Polygon comparison (order agnostic)
        if Counter(det_instances['polygons']) != Counter(spot_instances['polygons']):
            mismatch_detail = {
                'img_path': key,
                'det_unique': len(set(det_instances['polygons']) - set(spot_instances['polygons'])),
                'spot_unique': len(set(spot_instances['polygons']) - set(det_instances['polygons'])),
            }
            result['polygon_mismatch'].append(mismatch_detail)
            if len(result['polygon_samples']) < sample_count:
                result['polygon_samples'].append({
                    'img_path': key,
                    'det_polygons': det_instances['polygons_raw'],
                    'spot_polygons': spot_instances['polygons_raw'],
                    'det_bboxes': det_instances['bboxes_raw'],
                    'spot_bboxes': spot_instances['bboxes_raw'],
                    'category': 'polygon',
                })

        # Bounding box comparison
        if Counter(det_instances['bboxes']) != Counter(spot_instances['bboxes']):
            mismatch_detail = {
                'img_path': key,
                'det_unique': len(set(det_instances['bboxes']) - set(spot_instances['bboxes'])),
                'spot_unique': len(set(spot_instances['bboxes']) - set(det_instances['bboxes'])),
            }
            result['bbox_mismatch'].append(mismatch_detail)
            if len(result['bbox_samples']) < sample_count:
                result['bbox_samples'].append({
                    'img_path': key,
                    'det_bboxes': det_instances['bboxes_raw'],
                    'spot_bboxes': spot_instances['bboxes_raw'],
                    'det_polygons': det_instances['polygons_raw'],
                    'spot_polygons': spot_instances['polygons_raw'],
                    'category': 'bbox',
                })

        # Ignore flag comparison
        if det_instances['ignore'] != spot_instances['ignore']:
            result['ignore_mismatch'].append(key)
            if len(result['ignore_samples']) < sample_count:
                result['ignore_samples'].append({
                    'img_path': key,
                    'det_ignore': det_instances['ignore'],
                    'spot_ignore': spot_instances['ignore'],
                    'det_polygons': det_instances['polygons_raw'],
                    'spot_polygons': spot_instances['polygons_raw'],
                    'det_bboxes': det_instances['bboxes_raw'],
                    'spot_bboxes': spot_instances['bboxes_raw'],
                    'category': 'ignore',
                })

    return result


def _bbox_to_polygon(bbox: Sequence[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] bbox to polygon representation."""
    if len(bbox) != 4:
        return list(bbox)
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _polygon_to_points(poly: Sequence[float]) -> List[Tuple[float, float]]:
    return [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]


def _draw_polylines(draw: ImageDraw.ImageDraw,
                    polygons: Iterable[Sequence[float]],
                    color: Tuple[int, int, int],
                    width: int = 3) -> None:
    """Draw polygons on the image."""
    for poly in polygons:
        if not poly:
            continue
        pts = _polygon_to_points(poly)
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=width)


def _prepare_sample_entry(entry: dict, sample: dict) -> None:
    """Merge sample information into visualization entry."""
    entry['categories'].add(sample.get('category', 'unknown'))
    for key in ('det_polygons', 'spot_polygons', 'det_bboxes', 'spot_bboxes'):
        data = sample.get(key)
        if data:
            entry[key] = data


def _visualize_split(data_root: Path, split: str, info: dict, out_dir: Path) -> None:
    """Render visual comparisons for a given split."""
    samples: Dict[str, dict] = {}

    def add_samples(sample_list: List[dict]) -> None:
        for sample in sample_list:
            img_path = sample['img_path']
            entry = samples.setdefault(
                img_path,
                {
                    'det_polygons': [],
                    'spot_polygons': [],
                    'det_bboxes': [],
                    'spot_bboxes': [],
                    'categories': set(),
                })
            _prepare_sample_entry(entry, sample)

    add_samples(info.get('polygon_samples', []))
    add_samples(info.get('bbox_samples', []))
    add_samples(info.get('count_samples', []))
    add_samples(info.get('ignore_samples', []))

    font = ImageFont.load_default()

    for img_rel_path, entry in samples.items():
        img_file = data_root / img_rel_path
        if not img_file.exists():
            print(f'[WARN] Image not found for visualization: {img_file}')
            continue
        try:
            image = Image.open(img_file).convert('RGB')
        except Exception as exc:  # pylint: disable=broad-except
            print(f'[WARN] Failed to read image {img_file}: {exc}')
            continue

        vis = image.copy()
        draw = ImageDraw.Draw(vis)

        polygons_det = entry.get('det_polygons') or []
        polygons_spot = entry.get('spot_polygons') or []
        bboxes_det = entry.get('det_bboxes') or []
        bboxes_spot = entry.get('spot_bboxes') or []

        if not polygons_det and bboxes_det:
            polygons_det = [_bbox_to_polygon(b) for b in bboxes_det]
        if not polygons_spot and bboxes_spot:
            polygons_spot = [_bbox_to_polygon(b) for b in bboxes_spot]

        _draw_polylines(draw, polygons_det, color=(255, 0, 0))
        _draw_polylines(draw, polygons_spot, color=(0, 255, 0))

        legend = f'Det(red) vs Spot(green) | Categories: {", ".join(sorted(entry["categories"]))}'
        bbox = draw.textbbox((0, 0), legend, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([0, 0, text_width + 20, text_height + 10], fill=(0, 0, 0))
        draw.text((10, 5), legend, fill=(255, 255, 255), font=font)

        filename = Path(img_rel_path).name
        safe_name = filename.replace('/', '_').replace('\\', '_')
        out_path = out_dir / f'{split}_{safe_name}'
        vis.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Check CTW1500 detection vs spotting annotation consistency.')
    parser.add_argument(
        '--data-root',
        default=Path('data/ctw1500'),
        type=Path,
        help='Path to CTW1500 dataset root (defaults to data/ctw1500).')
    parser.add_argument(
        '--precision',
        default=2,
        type=int,
        help='Decimal precision used when comparing float coordinates.')
    parser.add_argument(
        '--sample-count',
        default=3,
        type=int,
        help='Number of example images to display for each mismatch category.')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If set, save visual examples comparing detection and spotting boxes.')
    parser.add_argument(
        '--visualize-dir',
        type=Path,
        default=Path('work_dirs/ctw1500_vis'),
        help='Directory to save visualization images (used with --visualize).')
    args = parser.parse_args()

    data_root: Path = args.data_root
    splits = ['train', 'test']
    summary = {}

    for split in splits:
        det_path = data_root / f'textdet_{split}.json'
        spot_path = data_root / f'textspotting_{split}.json'
        if not det_path.exists() or not spot_path.exists():
            raise FileNotFoundError(f'Missing required files for split {split}')
        summary[split] = _compare_split(
            det_path,
            spot_path,
            precision=args.precision,
            sample_count=args.sample_count)

    # Pretty print summary
    for split in splits:
        info = summary[split]
        print(f'=== Split: {split} ===')
        print(f'- det images       : {info["det_images"]}')
        print(f'- spotting images  : {info["spot_images"]}')
        print(f'- shared images    : {info["shared_images"]}')
        print(f'- det-only images  : {len(info["det_only"])}')
        print(f'- spot-only images : {len(info["spot_only"])}')
        print(f'- count mismatch   : {len(info["count_mismatch"])}')
        print(f'- polygon mismatch : {len(info["polygon_mismatch"])}')
        print(f'- bbox mismatch    : {len(info["bbox_mismatch"])}')
        print(f'- ignore mismatch  : {len(info["ignore_mismatch"])}')

        if info['det_only']:
            print('  Sample det-only image:', info['det_only'][0])
        if info['spot_only']:
            print('  Sample spot-only image:', info['spot_only'][0])
        if info['polygon_mismatch']:
            sample = info['polygon_mismatch'][0]
            print(
                f'  Sample polygon mismatch: {sample["img_path"]} '
                f'(det_unique={sample["det_unique"]}, spot_unique={sample["spot_unique"]})'
            )
        if info['bbox_mismatch']:
            sample = info['bbox_mismatch'][0]
            print(
                f'  Sample bbox mismatch: {sample["img_path"]} '
                f'(det_unique={sample["det_unique"]}, spot_unique={sample["spot_unique"]})'
            )
        if info['ignore_mismatch']:
            print('  Sample ignore mismatch:', info['ignore_mismatch'][0])

        if info['count_samples']:
            print('  Detailed count mismatch examples:')
            for sample in info['count_samples']:
                print(f"    {sample['img_path']}: det={sample['det_count']}, spot={sample['spot_count']}")

        if info['polygon_samples']:
            print('  Detailed polygon mismatch examples:')
            for sample in info['polygon_samples']:
                print(f"    {sample['img_path']}:")
                print('      det polygons :', sample['det_polygons'])
                print('      spot polygons:', sample['spot_polygons'])

        if info['bbox_samples']:
            print('  Detailed bbox mismatch examples:')
            for sample in info['bbox_samples']:
                print(f"    {sample['img_path']}:")
                print('      det bboxes :', sample['det_bboxes'])
                print('      spot bboxes:', sample['spot_bboxes'])

        if info['ignore_samples']:
            print('  Detailed ignore mismatch examples:')
            for sample in info['ignore_samples']:
                print(f"    {sample['img_path']}: det={sample['det_ignore']}, spot={sample['spot_ignore']}")
        print()

    if args.visualize:
        for split in splits:
            out_dir = args.visualize_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            _visualize_split(data_root, split, summary[split], out_dir)
        print(f'Visualization images saved to {args.visualize_dir}')


if __name__ == '__main__':
    main()
