#!/usr/bin/env python3
"""Verify text consistency between CTW1500 recognition and spotting annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _load_spotting_texts(path: Path) -> Dict[str, Counter]:
    with path.open('r') as f:
        data = json.load(f)
    mapping: Dict[str, Counter] = defaultdict(Counter)
    for item in data.get('data_list', []):
        img_name = Path(item['img_path']).stem
        for inst in item.get('instances', []):
            if inst.get('ignore'):
                continue
            text = inst.get('text', '')
            mapping[img_name][text] += 1
    return mapping


def _load_recog_texts(path: Path) -> Dict[str, Counter]:
    with path.open('r') as f:
        data = json.load(f)
    mapping: Dict[str, Counter] = defaultdict(Counter)
    for item in data.get('data_list', []):
        img_path = Path(item['img_path'])
        base = img_path.stem.split('_')[0]
        for inst in item.get('instances', []):
            text = inst.get('text', '')
            mapping[base][text] += 1
    return mapping


def _compare(recog_map: Dict[str, Counter],
             spot_map: Dict[str, Counter]) -> Dict[str, List]:

    recog_only = sorted(set(recog_map) - set(spot_map))
    spot_only = sorted(set(spot_map) - set(recog_map))
    shared = set(recog_map) & set(spot_map)

    count_diff = []
    text_diff = []
    for key in sorted(shared):
        rc = recog_map[key]
        sc = spot_map[key]
        if sum(rc.values()) != sum(sc.values()):
            count_diff.append({
                'img_id': key,
                'recog_count': sum(rc.values()),
                'spot_count': sum(sc.values()),
            })
        elif rc != sc:
            text_diff.append({
                'img_id': key,
                'recog_extra': list((rc - sc).elements()),
                'spot_extra': list((sc - rc).elements()),
            })

    return {
        'recog_only': recog_only,
        'spot_only': spot_only,
        'count_diff': count_diff,
        'text_diff': text_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare CTW1500 recognition and spotting texts.')
    parser.add_argument(
        '--data-root',
        default=Path('data/ctw1500'),
        type=Path,
        help='CTW1500 data root directory.')
    parser.add_argument(
        '--split',
        choices=['train', 'test', 'all'],
        default='all',
        help='Which split(s) to compare.')
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Number of example differences to print for each category.')
    args = parser.parse_args()

    splits = ['train', 'test'] if args.split == 'all' else [args.split]

    for split in splits:
        recog_path = args.data_root / f'textrecog_{split}.json'
        spot_path = args.data_root / f'textspotting_{split}.json'
        if not recog_path.exists() or not spot_path.exists():
            raise FileNotFoundError(
                f'Missing required files: {recog_path}, {spot_path}')

        recog_map = _load_recog_texts(recog_path)
        spot_map = _load_spotting_texts(spot_path)

        result = _compare(recog_map, spot_map)

        print(f'=== Split: {split} ===')
        print(f'- recognition images: {len(recog_map)}')
        print(f'- spotting images   : {len(spot_map)}')
        print(f'- recog only images : {len(result["recog_only"])}')
        print(f'- spot only images  : {len(result["spot_only"])}')
        print(f'- count mismatches  : {len(result["count_diff"])}')
        print(f'- text mismatches   : {len(result["text_diff"])}')

        if result['recog_only']:
            print('  sample recog-only:', result['recog_only'][:args.limit])
        if result['spot_only']:
            print('  sample spot-only:', result['spot_only'][:args.limit])
        if result['count_diff']:
            print('  sample count diff:', result['count_diff'][:args.limit])
        if result['text_diff']:
            print('  sample text diff :', result['text_diff'][:args.limit])
        print()


if __name__ == '__main__':
    main()
