import argparse
import random
import numpy as np
from pathlib import Path

import cv2
import mmengine


def parse_args():
    parser = argparse.ArgumentParser(description='可视化 MMOCR textdet 标注')
    parser.add_argument('--ann', required=True, help='annotation json 路径')
    parser.add_argument('--img-root', required=True, help='图片根目录')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='输出目录（保存带 polygon 的可视化图片）')
    parser.add_argument(
        '--num', type=int, default=3, help='可视化图片数量（默认 3）')
    parser.add_argument(
        '--seed', type=int, default=0, help='随机种子，0 表示按顺序取前几张')
    return parser.parse_args()


def main():
    args = parse_args()
    data = mmengine.load(args.ann)
    data_list = data.get('data_list', [])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed > 0:
        random.Random(args.seed).shuffle(data_list)
    samples = data_list[:args.num]
    for item in samples:
        img_path = Path(args.img_root) / item['img_path']
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'[WARN] 无法读取 {img_path}')
            continue
        for inst in item.get('instances', []):
            poly = inst.get('polygon') or inst.get('segmentation', [])
            if not poly:
                continue
            pts = list(zip(poly[0::2], poly[1::2]))
            pts = [(int(x), int(y)) for x, y in pts]
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], True,
                          (0, 255, 0), 2)
            if inst.get('ignore', False):
                cv2.putText(img, 'IGN',
                            (pts[0][0], pts[0][1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)
        out_file = out_dir / Path(item['img_path']).name
        cv2.imwrite(str(out_file), img)
        print(f'[SAVE] {out_file}')


if __name__ == '__main__':
    main()
