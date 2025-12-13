import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
ARCHIVE_EXTS = {
    '.zip': 'zip',
    '.rar': 'rar',
    '.tar': 'tar',
    '.tar.gz': 'tar',
    '.tgz': 'tar',
    '.tar.bz2': 'tar',
    '.tbz': 'tar',
}
ANNOTATION_EXTS = {'.json', '.txt', '.xml'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='检查并解压原始数据集（递归扫描压缩包，统计文件情况）')
    parser.add_argument(
        '--root',
        default='data',
        help='数据集根目录（包含 ArT / RCTW / ReCTS / LSVT / CTW 五个目录）')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅打印信息而不实际解压 / 扁平化 / 移动文件')
    return parser.parse_args()


def find_archives(root: Path) -> List[Path]:
    archives = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in ARCHIVE_EXTS or path.name.lower().endswith('.tar.gz'):
            archives.append(path)
    return archives


def run_cmd(cmd: Sequence[str], cwd: Path, dry_run: bool):
    if dry_run:
        print(f'[DRY-RUN] {" ".join(cmd)} (cwd={cwd})')
        return
    print(f'[RUN] {" ".join(cmd)} (cwd={cwd})')
    subprocess.run(cmd, check=False, cwd=str(cwd))


def extract_archive(archive: Path, dry_run: bool) -> Path:
    suffix = archive.suffix.lower()
    name = archive.name.lower()
    extract_dir = archive.parent
    if name.endswith('.tar.gz') or suffix in {'.tgz', '.tar', '.tar.bz2', '.tbz'}:
        cmd = ['tar', 'xf', archive.name]
    elif suffix == '.zip':
        cmd = ['unzip', '-o', archive.name]
    elif suffix == '.rar':
        cmd = ['unrar', 'x', '-o+', archive.name]
    else:
        print(f'[WARN] 未知压缩格式，跳过: {archive}')
        return extract_dir
    run_cmd(cmd, extract_dir, dry_run)
    return extract_dir


def flatten_dir(path: Path, dry_run: bool):
    """将只有单个子目录的多级目录向上提一层，减少冗余层级。"""
    current = path
    while current.is_dir():
        items = [p for p in current.iterdir() if p.name not in {'.DS_Store'}]
        if len(items) == 1 and items[0].is_dir():
            child = items[0]
            print(f'[INFO] 扁平化 {child} -> {current}')
            if not dry_run:
                for sub_item in child.iterdir():
                    target = current / sub_item.name
                    if target.exists():
                        continue
                    shutil.move(str(sub_item), str(target))
                child.rmdir()
            current = path
        else:
            break


def count_images_and_ann(root: Path) -> Tuple[int, List[str]]:
    num_images = 0
    ann_paths: List[str] = []
    for p in root.rglob('*'):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                num_images += 1
            if ext in ANNOTATION_EXTS:
                ann_paths.append(str(p))
    return num_images, ann_paths


def detect_split_files(ann_paths: Sequence[str]) -> List[str]:
    keywords = ['train', 'val', 'test']
    candidates = []
    for p in ann_paths:
        lower = Path(p).name.lower()
        if any(k in lower for k in keywords):
            candidates.append(p)
    return candidates


def process_dataset(root: Path, dry_run: bool):
    print(f'\n===== 处理数据集目录: {root} =====')
    if not root.exists():
        print(f'[WARN] 目录不存在，跳过: {root}')
        return
    archives = find_archives(root)
    if archives:
        print(f'[INFO] 发现 {len(archives)} 个压缩包')
    for arc in archives:
        extract_archive(arc, dry_run)
    flatten_dir(root, dry_run)
    num_images, ann_paths = count_images_and_ann(root)
    split_hint = detect_split_files(ann_paths)
    print(f'[STAT] 图像数量: {num_images}')
    if ann_paths:
        print('[STAT] 标注文件列表:')
        for ap in sorted(ann_paths):
            print(f'  - {ap}')
    else:
        print('[STAT] 未发现标注文件')
    if split_hint:
        print('[STAT] 检测到可能的划分信息:')
        for sp in split_hint:
            print(f'  - {sp}')


def main():
    args = parse_args()
    root = Path(args.root).expanduser()
    targets = [
        root / 'ICDAR 2019 - ArT',
        root / 'RCTW-17',
        root / 'rects',
        root / 'ICDAR 2019 - LSVT',
        root / 'CTW',
    ]
    for dataset_root in targets:
        process_dataset(dataset_root, args.dry_run)


if __name__ == '__main__':
    main()
