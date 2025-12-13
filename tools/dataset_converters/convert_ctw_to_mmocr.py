import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mmcv

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_converters.common.textdet_mmocr_helper import (  # noqa: E402
    dump_split, rewrite_and_link, split_train_val)
from mmocr.utils import dump_ocr_data  # noqa: E402


def parse_line(line: str) -> Optional[Tuple[List[int], str]]:
    parts = [p for p in re.split(r'[,\s]+', line.strip()) if p != '']
    nums: List[float] = []
    texts: List[str] = []
    for token in parts:
        try:
            nums.append(float(token))
        except ValueError:
            texts.append(token)
    if len(nums) < 8 or len(nums) % 2 != 0:
        return None
    coords = [int(round(n)) for n in nums]
    text = ' '.join(texts) if texts else '###'
    return coords, text


def bbox_from_poly(poly: List[int]) -> List[int]:
    xs, ys = poly[::2], poly[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def clip_poly(poly: List[float], w: int, h: int) -> List[float]:
    """Clip polygon coordinates into image bounds."""
    clipped = []
    for i, v in enumerate(poly):
        if i % 2 == 0:
            clipped.append(min(max(v, 0), w - 1))
        else:
            clipped.append(min(max(v, 0), h - 1))
    return clipped


def load_txt_ann(gt_file: Path, img_dir: Path) -> Optional[Dict]:
    stem = gt_file.stem
    candidates = list(img_dir.glob(stem + '.*'))
    if not candidates:
        return None
    img_path = candidates[0]
    img = mmcv.imread(str(img_path))
    instances = []
    with gt_file.open(encoding='utf-8-sig') as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue
            poly, text = parsed
            clipped = clip_poly(poly, img.shape[1], img.shape[0])
            bbox = bbox_from_poly(clipped)
            instances.append(
                dict(
                    iscrowd=1 if text == '###' else 0,
                    category_id=0,
                    bbox=bbox,
                    area=bbox[2] * bbox[3],
                    segmentation=[clipped],
                    text=text))
    if not instances:
        return None
    return dict(
        file_name=img_path.name,
        height=img.shape[0],
        width=img.shape[1],
        anno_info=instances)


def load_jsonl_ann(jsonl_path: Path, img_root: Path) -> List[dict]:
    """解析官方 jsonl 标注格式."""
    infos = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            img_path = img_root / data['file_name']
            if not img_path.exists():
                continue
            img = mmcv.imread(str(img_path))
            h, w = img.shape[:2]
            instances = []
            for block in data.get('annotations', []):
                if not isinstance(block, list):
                    block = [block]
                for ann in block:
                    poly_pts = ann.get('polygon', [])
                    if not poly_pts:
                        continue
                    flat = []
                    for pt in poly_pts:
                        flat.extend([pt[0], pt[1]])
                    clipped = clip_poly(flat, w, h)
                    bbox = bbox_from_poly(clipped)
                    instances.append(
                        dict(
                            iscrowd=0,
                            category_id=0,
                            bbox=bbox,
                            area=bbox[2] * bbox[3],
                            segmentation=[clipped],
                            text=ann.get('text', '###')))
            for ign in data.get('ignore', []):
                poly_pts = ign.get('polygon', [])
                if not poly_pts:
                    continue
                flat = []
                for pt in poly_pts:
                    flat.extend([pt[0], pt[1]])
                clipped = clip_poly(flat, w, h)
                bbox = bbox_from_poly(clipped)
                instances.append(
                    dict(
                        iscrowd=1,
                        category_id=0,
                        bbox=bbox,
                        area=bbox[2] * bbox[3],
                        segmentation=[clipped],
                        text='###'))
            if instances:
                infos.append(
                    dict(
                        file_name=img_path.name,
                        height=h,
                        width=w,
                        anno_info=instances))
    return infos


def try_load_mmocr_json(ann_path: Path) -> Optional[List[dict]]:
    data = json.loads(ann_path.read_text())
    if 'data_list' not in data:
        return None
    infos = []
    for item in data['data_list']:
        poly_instances = []
        for inst in item.get('instances', []):
            poly = inst.get('polygon') or inst.get('poly') or inst.get(
                'segmentation')
            if not poly:
                continue
            w = item.get('width')
            h = item.get('height')
            if w and h:
                poly = clip_poly(poly, w, h)
            if len(poly) >= 4:
                xs, ys = poly[::2], poly[1::2]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            else:
                bbox = inst.get('bbox')
            poly_instances.append(
                dict(
                    iscrowd=1 if inst.get('ignore', False) else 0,
                    category_id=0,
                    bbox=bbox,
                    area=bbox[2] * bbox[3],
                    segmentation=[poly],
                    text=inst.get('text', '###')))
        if not poly_instances:
            continue
        infos.append(
            dict(
                file_name=item['img_path'],
                height=item.get('height', 0),
                width=item.get('width', 0),
                anno_info=poly_instances))
    return infos


def collect_ctw_infos(root: Path) -> List[dict]:
    ann_dir = root / 'annotations'
    img_dir = root / 'images'
    if not img_dir.exists():
        img_dir = root / 'imgs'
    mmocr_json_candidates = [
        ann_dir / 'instances_train.json',
        ann_dir / 'instances_training.json',
        root / 'instances_train.json',
        root / 'instances_training.json',
    ]
    for candidate in mmocr_json_candidates:
        if candidate.exists():
            infos = try_load_mmocr_json(candidate)
            if infos:
                print(f'[INFO] 使用已有 MMOCR 标注: {candidate}')
                return infos
    # 尝试官方 jsonl
    jsonl_train = root / 'train.jsonl'
    if jsonl_train.exists():
        infos = load_jsonl_ann(jsonl_train, img_dir)
        val_jsonl = root / 'val.jsonl'
        if val_jsonl.exists():
            infos_val = load_jsonl_ann(val_jsonl, img_dir)
            infos.extend(infos_val)
        if infos:
            print(f'[INFO] 使用 jsonl 标注: {jsonl_train}')
            return infos
    txt_files = list(ann_dir.rglob('*.txt')) if ann_dir.exists() else []
    if not txt_files:
        raise FileNotFoundError(
            f'在 {ann_dir} 未找到标注 txt/json/jsonl，无法继续转换。')
    infos: List[dict] = []
    for gt_file in txt_files:
        info = load_txt_ann(gt_file, img_dir)
        if info:
            infos.append(info)
    if not infos:
        raise RuntimeError('CTW 标注解析结果为空，请检查标注格式。')
    return infos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Chinese Text in the Wild 转 MMOCR 标准格式')
    parser.add_argument('--root', required=True, help='原始 CTW 数据根目录')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='输出目录，例如 data/ctw_mmocr （包含 imgs 与 instances_*.json）')
    parser.add_argument(
        '--val-ratio', type=float, default=0.1, help='验证集划分比例')
    parser.add_argument('--seed', type=int, default=42, help='划分随机种子')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_img_dir = out_dir / 'imgs'
    out_img_dir.mkdir(parents=True, exist_ok=True)

    all_infos = collect_ctw_infos(root)
    train_infos_raw, val_infos_raw = split_train_val(
        all_infos, args.val_ratio, args.seed)
    img_root = root / 'images'
    if not img_root.exists():
        img_root = root / 'imgs'
    train_infos = rewrite_and_link(train_infos_raw, img_root, out_img_dir,
                                   'CTW')
    val_infos = rewrite_and_link(val_infos_raw, img_root, out_img_dir, 'CTW')
    train_json, val_json = dump_split(train_infos, val_infos, out_dir,
                                      task='textdet')
    print(f'[DONE] 训练标注: {train_json}')
    print(f'[DONE] 验证标注: {val_json}')


if __name__ == '__main__':
    main()
