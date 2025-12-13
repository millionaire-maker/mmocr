import argparse
import sys
from pathlib import Path
from typing import List

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_converters.common.textdet_mmocr_helper import (  # noqa: E402
    dump_split, rewrite_and_link, split_train_val)
from tools.dataset_converters.textdet.art_converter import (  # noqa: E402
    collect_art_info)


def load_all_infos(root: Path) -> List[dict]:
    # ratio=0 代表不在此处切分，我们后续统一切分
    return collect_art_info(str(root), split='train', ratio=0.0)


def parse_args():
    parser = argparse.ArgumentParser(description='ArT 转换为 MMOCR 标准格式')
    parser.add_argument('--root', required=True, help='原始 ArT 数据根目录')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='输出目录，例如 data/art_mmocr （包含 imgs 与 instances_*.json）')
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

    all_infos = load_all_infos(root)
    train_infos_raw, val_infos_raw = split_train_val(
        all_infos, args.val_ratio, args.seed)
    train_infos = rewrite_and_link(train_infos_raw, root / 'imgs',
                                   out_img_dir, 'ArT')
    val_infos = rewrite_and_link(val_infos_raw, root / 'imgs', out_img_dir,
                                 'ArT')
    train_json, val_json = dump_split(train_infos, val_infos, out_dir,
                                      task='textdet')
    print(f'[DONE] 训练标注: {train_json}')
    print(f'[DONE] 验证标注: {val_json}')


if __name__ == '__main__':
    main()
