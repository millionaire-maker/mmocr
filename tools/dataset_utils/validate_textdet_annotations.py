import argparse
import json
from pathlib import Path
from typing import Dict, List

import mmcv
import mmengine
from mmengine.utils import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description='校验 MMOCR textdet 标注一致性')
    parser.add_argument('--ann', required=True, help='annotation json 路径')
    parser.add_argument('--img-root', required=True, help='图片根目录')
    parser.add_argument(
        '--out',
        default=None,
        help='统计输出路径（默认写到标注同目录 stats.json）')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='可选，限制校验的图片数量（用于快速 sanity check）')
    return parser.parse_args()


def check_instance(inst: Dict, img_w: int, img_h: int, errors: List[str]):
    poly = inst.get('polygon') or inst.get('poly') or inst.get('segmentation')
    if not poly:
        errors.append('缺少 polygon')
        return False
    if len(poly) % 2 != 0 or len(poly) < 8:
        errors.append(f'polygon 长度异常: {len(poly)}')
        return False
    xs, ys = poly[::2], poly[1::2]
    if max(xs) > img_w + 1 or max(ys) > img_h + 1 or min(xs) < -1 or min(
            ys) < -1:
        errors.append('polygon 坐标越界')
    if not isinstance(inst.get('text', '###'), str):
        errors.append('text 字段不是字符串')
    if 'ignore' in inst and not isinstance(inst['ignore'], bool):
        errors.append('ignore 字段不是 bool')
    return True


def main():
    args = parse_args()
    ann_path = Path(args.ann).expanduser()
    img_root = Path(args.img_root).expanduser()
    data = mmengine.load(str(ann_path))
    data_list = data.get('data_list', [])
    curved = 0
    total = 0
    total_images = 0
    error_records: List[Dict] = []
    limit = args.limit if args.limit is None else max(args.limit, 1)

    total_iters = len(data_list) if limit is None else min(
        len(data_list), limit)
    pbar = ProgressBar(total_iters)
    for idx, item in enumerate(data_list):
        if limit and total_images >= limit:
            break
        img_path = img_root / item['img_path']
        if not img_path.exists():
            error_records.append(
                dict(img=item['img_path'], error='图像不存在', instances=[]))
            pbar.update()
            continue
        img = mmcv.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        total_images += 1
        inst_errors: List[str] = []
        valid_instances = []
        for inst in item.get('instances', []):
            total += 1
            errors = []
            ok = check_instance(inst, img_w, img_h, errors)
            if len(inst.get('polygon', inst.get('segmentation', []))) > 8:
                curved += 1
            if errors:
                inst_errors.append('; '.join(errors))
            if ok:
                valid_instances.append(inst)
        if inst_errors:
            error_records.append(
                dict(img=item['img_path'], error='; '.join(inst_errors)))
        pbar.update()

    stats = dict(
        images=total_images,
        instances=total,
        curved_instances=curved,
        curved_ratio=0 if total == 0 else round(curved / total, 4),
        errors=error_records)
    out_path = Path(args.out) if args.out else ann_path.parent / 'stats.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'[DONE] 校验完成，统计写入 {out_path}')
    print(
        f'[STAT] images={total_images}, instances={total}, curved={curved}, errors={len(error_records)}'
    )


if __name__ == '__main__':
    main()
