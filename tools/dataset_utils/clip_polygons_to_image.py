import argparse
from pathlib import Path
from typing import List, Tuple

import mmengine
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='将 MMOCR textdet 标注中的 polygon 裁剪到图像范围内')
    parser.add_argument('--ann', required=True, help='输入 annotation json')
    parser.add_argument(
        '--img-root',
        required=True,
        help='图片根目录（包含 imgs/，用于拼接 img_path）')
    parser.add_argument(
        '--out',
        default=None,
        help='输出路径（默认覆盖原文件）')
    parser.add_argument(
        '--only-source',
        default=None,
        help='仅处理指定 source 的样本（如 ctw_mmocr），为空则处理全部')
    return parser.parse_args()


def clip_poly(poly: List[float], w: int, h: int) -> Tuple[List[float], bool]:
    """Clip polygon coordinates into [0, w-1] / [0, h-1]."""
    clipped: List[float] = []
    touched = False
    for i, v in enumerate(poly):
        limit = (w - 1) if i % 2 == 0 else (h - 1)
        nv = min(max(float(v), 0.0), float(limit))
        if nv != v:
            touched = True
        clipped.append(nv)
    return clipped, touched


def main():
    args = parse_args()
    ann_path = Path(args.ann).expanduser()
    out_path = Path(args.out) if args.out else ann_path
    data = mmengine.load(str(ann_path))
    data_list = data.get('data_list', [])
    img_root = Path(args.img_root).expanduser()
    modified_inst = 0
    modified_img = 0
    total = 0
    for item in data_list:
        if args.only_source and item.get('source') != args.only_source:
            continue
        w, h = int(item.get('width', 0)), int(item.get('height', 0))
        if w <= 0 or h <= 0:
            img_path = img_root / item['img_path']
            img = mmcv.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
        touched_img = False
        for inst in item.get('instances', []):
            total += 1
            poly = inst.get('polygon') or inst.get('segmentation', [])
            if not poly:
                continue
            clipped, touched = clip_poly(poly, w, h)
            if touched:
                touched_img = True
                inst['polygon'] = clipped
                if 'segmentation' in inst:
                    inst['segmentation'] = clipped
                xs, ys = clipped[::2], clipped[1::2]
                inst['bbox'] = [
                    min(xs), min(ys),
                    max(xs) - min(xs), max(ys) - min(ys)
                ]
                modified_inst += 1
        if touched_img:
            modified_img += 1
    mmengine.dump(data, str(out_path))
    print(f'[DONE] 写回 {out_path}')
    print(
        f'[STAT] total_instances={total}, modified_instances={modified_inst}, modified_images={modified_img}'
    )


if __name__ == '__main__':
    main()
