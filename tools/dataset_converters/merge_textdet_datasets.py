import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional

import mmengine

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_converters.common.textdet_mmocr_helper import safe_symlink  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='合并多个 MMOCR textdet 数据集（软链接图片，合并标注并附带 source 字段）'
    )
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='要合并的数据集根目录，可用 name:path 或 path 形式（name 用于标记 source）')
    parser.add_argument('--out-dir', required=True, help='输出目录')
    parser.add_argument(
        '--train-ann',
        default='instances_train.json',
        help='默认的训练标注文件名（若不存在会回落到 instances_training.json）')
    parser.add_argument(
        '--val-ann',
        default='instances_val.json',
        help='默认的验证标注文件名（可缺省）')
    parser.add_argument(
        '--copy',
        action='store_true',
        help='默认使用软链接；若指定 --copy 则复制文件')
    return parser.parse_args()


def locate_ann(root: Path, fname: str, fallback: Optional[str] = None) -> Path:
    candidates = [
        fname,
        fallback,
        'instances_training.json',
        'instances_train.json',
        'instances_trainval.json',
        'instances_val.json',
    ]
    for cand in candidates:
        if not cand:
            continue
        target = root / cand
        if target.exists():
            return target
    raise FileNotFoundError(f'{root} 下未找到候选标注文件: {candidates}')


def load_data_list(ann_path: Path) -> List[Dict]:
    data = mmengine.load(str(ann_path))
    if 'data_list' not in data:
        raise ValueError(f'{ann_path} 不是 MMOCR data_list 格式')
    return data['data_list']


def link_image(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        import shutil
        shutil.copy(src, dst)
    else:
        safe_symlink(src, dst)


def remap_item(item: Dict, img_root: Path, tag: str, dst_img_dir: Path,
               copy_files: bool) -> Dict:
    new_item = copy.deepcopy(item)
    img_rel = item['img_path']
    src_img = img_root / img_rel
    if not src_img.exists():
        src_img = img_root / Path(img_rel).name
    new_name = f'{tag}_{Path(src_img).name}'
    dst_img = dst_img_dir / new_name
    link_image(src_img, dst_img, copy_files)
    new_item['img_path'] = str(Path('imgs') / new_name)
    new_item['source'] = tag
    return new_item


def merge_dataset(input_item: str, train_ann_name: str, val_ann_name: str,
                  dst_img_dir: Path, copy_files: bool):
    if ':' in input_item:
        tag, path = input_item.split(':', 1)
    else:
        path = input_item
        tag = Path(input_item).name.replace(' ', '_')
    root = Path(path).expanduser()
    img_root = root
    if (root / 'imgs').exists():
        img_root = root
    if (root / 'images').exists():
        img_root = root
    train_path = locate_ann(root, train_ann_name, 'instances_training.json')
    try:
        val_path = locate_ann(root, val_ann_name, None)
    except FileNotFoundError:
        val_path = None
    train_list = load_data_list(train_path)
    val_list = load_data_list(val_path) if val_path and val_path.exists(
    ) else []
    merged_train = [
        remap_item(item, root, tag, dst_img_dir, copy_files)
        for item in train_list
    ]
    merged_val = [
        remap_item(item, root, tag, dst_img_dir, copy_files)
        for item in val_list
    ]
    return merged_train, merged_val


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    dst_img_dir = out_dir / 'imgs'
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    all_train: List[Dict] = []
    all_val: List[Dict] = []
    for item in args.inputs:
        train_part, val_part = merge_dataset(item, args.train_ann,
                                             args.val_ann, dst_img_dir,
                                             args.copy)
        all_train.extend(train_part)
        all_val.extend(val_part)
    out_json = dict(
        metainfo=dict(
            dataset_type='TextDetDataset',
            task_name='textdet',
            category=[dict(id=0, name='text')]),
        data_list=all_train)
    out_val_json = dict(
        metainfo=dict(
            dataset_type='TextDetDataset',
            task_name='textdet',
            category=[dict(id=0, name='text')]),
        data_list=all_val)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / 'instances_train.json'
    val_out = out_dir / 'instances_val.json'
    mmengine.dump(out_json, str(train_out))
    mmengine.dump(out_val_json, str(val_out))
    print(f'[DONE] 合并完成，训练标注 {train_out} ，验证标注 {val_out}')


if __name__ == '__main__':
    main()
