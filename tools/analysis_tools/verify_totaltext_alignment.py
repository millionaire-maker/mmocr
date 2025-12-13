#!/usr/bin/env python3
"""Check alignment between TotalText detection and recognition annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def _load_detection_bases(path: Path) -> Set[str]:
    with path.open('r') as f:
        data = json.load(f)
    bases = set()
    for item in data.get('data_list', []):
        img_path = Path(item['img_path'])
        bases.add(img_path.stem.lower())
    return bases


def _load_recog_bases(path: Path) -> Dict[str, List[str]]:
    with path.open('r') as f:
        data = json.load(f)
    mapping: Dict[str, List[str]] = defaultdict(list)
    for item in data.get('data_list', []):
        img_name = Path(item['img_path']).name
        base = img_name.split('_')[0].lower()
        mapping[base].append(img_name)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Verify TotalText detection and recognition alignment.')
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data/totaltext'),
        help='Path to TotalText data root.')
    parser.add_argument(
        '--split',
        choices=['train', 'test', 'all'],
        default='all',
        help='Which splits to inspect.')
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of unmatched examples to print per split.')
    args = parser.parse_args()

    splits = ['train', 'test'] if args.split == 'all' else [args.split]

    for split in splits:
        det_path = args.data_root / f'textdet_{split}.json'
        rec_path = args.data_root / f'textrecog_{split}.json'
        if not det_path.exists() or not rec_path.exists():
            raise FileNotFoundError(
                f'Missing required files: {det_path}, {rec_path}')

        det_bases = _load_detection_bases(det_path)
        rec_map = _load_recog_bases(rec_path)

        missing = sorted(base for base in rec_map if base not in det_bases)
        print(f'=== Split: {split} ===')
        print(f'- detection images : {len(det_bases)}')
        print(f'- recognition bases: {len(rec_map)}')
        print(f'- unmatched bases  : {len(missing)}')
        if missing:
            print('  sample unmatched :', [
                (base, rec_map[base][:3]) for base in missing[:args.limit]
            ])
        print()


if __name__ == '__main__':
    main()
