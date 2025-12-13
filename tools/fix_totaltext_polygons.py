#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List


def filter_instances(sample: Dict) -> Dict:
    """Remove instances whose polygon has less than 3 points."""
    instances = sample.get('instances', [])
    kept: List[Dict] = []
    removed = 0
    for inst in instances:
        polygon = inst.get('polygon')
        if polygon is None:
            kept.append(inst)
            continue
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            removed += 1
            continue
        kept.append(inst)
    sample['instances'] = kept
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Filter invalid polygons from TotalText annotations.')
    parser.add_argument(
        'ann_file',
        type=Path,
        nargs='?',
        default=Path('data/totaltext/textdet_train.json'),
        help='Path to TotalText annotation json file.')
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create a backup with suffix .bak before modifying the file.')
    args = parser.parse_args()

    ann_file: Path = args.ann_file
    if not ann_file.is_file():
        raise FileNotFoundError(f'{ann_file} not found')

    with ann_file.open('r') as f:
        data = json.load(f)

    if args.backup:
        backup_path = ann_file.with_suffix(ann_file.suffix + '.bak')
        if not backup_path.exists():
            backup_path.write_text(json.dumps(data, ensure_ascii=False))
            print(f'Backup saved to {backup_path}')

    removed_instances = 0
    removed_samples = 0
    new_data_list = []
    for sample in data.get('data_list', []):
        removed = filter_instances(sample)
        removed_instances += removed
        if not sample.get('instances'):
            removed_samples += 1
            continue
        new_data_list.append(sample)

    data['data_list'] = new_data_list
    with ann_file.open('w') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

    print(f'Removed {removed_instances} invalid instances.')
    print(f'Removed {removed_samples} samples without valid polygons.')


if __name__ == '__main__':
    main()
