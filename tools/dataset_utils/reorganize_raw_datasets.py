#!/usr/bin/env python
"""重组原始数据集目录结构以匹配转换器期望的格式."""
import argparse
import shutil
from pathlib import Path


def reorganize_art(root: Path, dry_run: bool = False):
    """重组 ArT 数据集."""
    print(f"\n===== 重组 ArT 数据集: {root} =====")
    
    # 创建 annotations 和 imgs 目录
    ann_dir = root / "annotations"
    imgs_dir = root / "imgs"
    
    if not dry_run:
        ann_dir.mkdir(exist_ok=True)
        imgs_dir.mkdir(exist_ok=True)
    
    # 移动标注文件
    for json_file in ["train_labels.json", "train_task2_labels.json"]:
        src = root / json_file
        dst = ann_dir / json_file
        if src.exists() and not dst.exists():
            print(f"  移动: {json_file} -> annotations/{json_file}")
            if not dry_run:
                shutil.move(str(src), str(dst))
    
    # 移动或软链接图片 - 收集所有训练图片目录
    train_img_dirs = [
        root / "train_images",
        root / "train_task2_images"
    ]
    for train_img_dir in train_img_dirs:
        if train_img_dir.exists():
            print(f"  处理训练图片: {train_img_dir.name}/ -> imgs/")
            if not dry_run:
                for img in train_img_dir.glob("*.jpg"):
                    dst = imgs_dir / img.name
                    if not dst.exists():
                        shutil.copy(str(img), str(dst))
    
    # 统计
    if imgs_dir.exists():
        img_count = len(list(imgs_dir.glob("*.jpg")))
        print(f"  --> imgs/ 下共 {img_count} 张图片")


def reorganize_rctw(root: Path, dry_run: bool = False):
    """重组 RCTW-17 数据集."""
    print(f"\n===== 重组 RCTW-17 数据集: {root} =====")
    
    # train_gts -> annotations
    train_gts = root / "train_gts"
    annotations = root / "annotations"
    imgs_dir = root / "imgs"
    
    if train_gts.exists() and not annotations.exists():
        print(f"  重命名: train_gts/ -> annotations/")
        if not dry_run:
            shutil.move(str(train_gts), str(annotations))
    
    # train_images -> imgs
    train_images = root / "train_images"
    if train_images.exists() and not imgs_dir.exists():
        print(f"  重命名: train_images/ -> imgs/")
        if not dry_run:
            shutil.move(str(train_images), str(imgs_dir))
    
    # 统计
    if annotations and annotations.exists():
        ann_count = len(list(annotations.glob("*.txt")))
        print(f"  --> annotations/ 下共 {ann_count} 个标注")
    if imgs_dir and imgs_dir.exists():
        img_count = len(list(imgs_dir.glob("*.jpg")))
        print(f"  --> imgs/ 下共 {img_count} 张图片")


def reorganize_rects(root: Path, dry_run: bool = False):
    """重组 ReCTS 数据集."""
    print(f"\n===== 重组 ReCTS 数据集: {root} =====")
    
    # 检查是否有 rects 子目录
    rects_subdir = root / "rects"
    if rects_subdir.exists() and rects_subdir.is_dir():
        print(f"  发现嵌套目录 rects/,向上提升内容")
        if not dry_run:
            # 将 rects/ 下的内容移到根目录
            for item in rects_subdir.iterdir():
                dst = root / item.name
                if not dst.exists():
                    shutil.move(str(item), str(dst))
            if not any(rects_subdir.iterdir()):
                rects_subdir.rmdir()
    
    # 寻找 img 或 img_train 等目录
    possible_img_dirs = ["img", "img_train", "imgs"]
    annotations_dir = root / "annotations"
    imgs_dir = root / "imgs"
    
    for poss in possible_img_dirs:
        candidate = root / poss
        if candidate.exists() and candidate.is_dir() and candidate != imgs_dir:
            print(f"  重命名: {poss}/ -> imgs/")
            if not dry_run:
                if not imgs_dir.exists():
                    shutil.move(str(candidate), str(imgs_dir))
            break
    
    # 寻找 gt 或 gt_train 等目录
    possible_gt_dirs = ["gt", "gt_train", "annotations"]
    for poss in possible_gt_dirs:
        candidate = root / poss
        if candidate.exists() and candidate.is_dir() and candidate != annotations_dir:
            print(f"  重命名: {poss}/ -> annotations/")
            if not dry_run:
                if not annotations_dir.exists():
                    shutil.move(str(candidate), str(annotations_dir))
            break
    
    # 统计
    if annotations_dir and annotations_dir.exists():
        ann_count = len(list(annotations_dir.glob("*.json")))
        print(f"  --> annotations/ 下共 {ann_count} 个标注")
    if imgs_dir and imgs_dir.exists():
        img_count = len(list(imgs_dir.glob("*.jpg")))
        print(f"  --> imgs/ 下共 {img_count} 张图片")


def reorganize_lsvt(root: Path, dry_run: bool = False):
    """重组 LSVT 数据集."""
    print(f"\n===== 重组 LSVT 数据集: {root} =====")
    
    # 创建 annotations 和 imgs 目录
    ann_dir = root / "annotations"
    imgs_dir = root / "imgs"
    
    if not dry_run:
        ann_dir.mkdir(exist_ok=True)
        imgs_dir.mkdir(exist_ok=True)
    
    # 移动标注文件
    for json_file in ["train_full_labels.json", "train_weak_labels.json"]:
        src = root / json_file
        dst = ann_dir / json_file
        if src.exists() and not dst.exists():
            print(f"  移动: {json_file} -> annotations/{json_file}")
            if not dry_run:
                shutil.move(str(src), str(dst))
    
    # 合并多个训练图片目录
    for img_dir_name in ["train_full_images_0", "train_full_images_1"]:
        img_dir = root / img_dir_name
        if img_dir.exists():
            print(f"  合并: {img_dir_name}/ -> imgs/")
            if not dry_run:
                for img in img_dir.glob("*.jpg"):
                    dst = imgs_dir / img.name
                    if not dst.exists():
                        shutil.copy(str(img), str(dst))
    
    # 统计
    if imgs_dir.exists():
        img_count = len(list(imgs_dir.glob("*.jpg")))
        print(f"  --> imgs/ 下共 {img_count} 张图片")


def reorganize_ctw(root: Path, dry_run: bool = False):
    """重组 CTW 数据集."""
    print(f"\n===== 重组 CTW 数据集: {root} =====")
    
    # 检查 info.json 是否存在
    info_json = root / "info.json"
    if not info_json.exists():
        print("  警告: 未找到 info.json,可能标注格式不标准")
    
    # 创建 imgs 目录
    imgs_dir = root / "imgs"
    if not dry_run:
        imgs_dir.mkdir(exist_ok=True)
    
    # 从 ctw_images-trainval_datasets 目录收集图片
    ctw_trainval = root / "ctw_images-trainval_datasets"
    if ctw_trainval.exists():
        print(f"  合并: ctw_images-trainval_datasets/ -> imgs/")
        if not dry_run:
            # 递归查找所有 jpg 文件
            for img in ctw_trainval.rglob("*.jpg"):
                dst = imgs_dir / img.name
                if not dst.exists():
                    shutil.copy(str(img), str(dst))
    
    # 统计
    if imgs_dir.exists():
        img_count = len(list(imgs_dir.glob("*.jpg")))
        print(f"  --> imgs/ 下共 {img_count} 张图片")


def main():
    parser = argparse.ArgumentParser(description="重组原始数据集目录结构")
    parser.add_argument(
        "--root", default="data", help="数据集根目录"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="仅打印操作而不实际执行"
    )
    args = parser.parse_args()
    
    root = Path(args.root).expanduser()
    
    # 重组各个数据集
    reorganize_art(root / "ICDAR 2019 - ArT", args.dry_run)
    reorganize_rctw(root / "RCTW-17", args.dry_run)
    reorganize_rects(root / "rects", args.dry_run)
    reorganize_lsvt(root / "ICDAR 2019 - LSVT", args.dry_run)
    reorganize_ctw(root / "CTW", args.dry_run)
    
    print("\n===== 重组完成 =====")


if __name__ == "__main__":
    main()
