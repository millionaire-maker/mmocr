import os
import shutil
from pathlib import Path
from mmengine.config import Config
from mmocr.datasets.preparers.data_preparer import DatasetPreparer


def main():
    cfg_path = '/home/yzy/mmocr/dataset_zoo/totaltext/textrecog.py'
    cfg = Config.fromfile(cfg_path)

    # Use user's dataset root
    cfg.data_root = '/home/yzy/mmocr/datasets/totaltext'
    cfg.dataset_name = 'totaltext'
    cfg.task = 'textrecog'

    # Reuse existing images in Train/Test; skip online obtaining
    for split_key in ['train_preparer', 'test_preparer']:
        preparer = cfg.get(split_key, None)
        if not preparer:
            continue
        # Disable obtainer to avoid network download
        if 'obtainer' in preparer:
            preparer['obtainer'] = None

        # Point gatherer to user's existing image dirs
        gatherer = preparer.get('gatherer', None)
        if gatherer:
            if split_key == 'train_preparer':
                gatherer['img_dir'] = 'Train'
            else:
                gatherer['img_dir'] = 'Test'
            # Use annotations placed under annotations/{split}
            gatherer['ann_dir'] = 'annotations'

    # Ensure annotations are available under data_root/annotations/{train,test}
    data_root = Path(cfg.data_root)
    ann_train_dst = data_root / 'annotations' / 'train'
    ann_test_dst = data_root / 'annotations' / 'test'
    ann_train_dst.mkdir(parents=True, exist_ok=True)
    ann_test_dst.mkdir(parents=True, exist_ok=True)

    # Prefer user-provided txt_format folder
    local_txt_format = data_root / 'txt_format'
    if local_txt_format.exists():
        for src_dir, dst_dir in [
            (local_txt_format / 'Train', ann_train_dst),
            (local_txt_format / 'Test', ann_test_dst),
        ]:
            if src_dir.exists():
                for p in src_dir.glob('*.txt'):
                    shutil.copy2(p, dst_dir / p.name)

    # If still missing, try GitHub Groundtruth fallback (may be incomplete)
    if not any(ann_train_dst.glob('*.txt')) or not any(ann_test_dst.glob('*.txt')):
        gt_repo_root = Path('/home/yzy/mmocr/datasets/Total-Text-Dataset')
        gt_train_src = gt_repo_root / 'Groundtruth' / 'Polygon' / 'Train'
        gt_test_src = gt_repo_root / 'Groundtruth' / 'Polygon' / 'Test'
        if gt_train_src.exists() and gt_test_src.exists():
            for src_dir, dst_dir in [(gt_train_src, ann_train_dst), (gt_test_src, ann_test_dst)]:
                for p in src_dir.glob('*.txt'):
                    shutil.copy2(p, dst_dir / p.name)

    # Run preparer (will download annotations, pair with Train/Test images, crop for textrecog)
    preparer = DatasetPreparer.from_file(cfg)
    preparer.run(splits=['train', 'test'])


if __name__ == '__main__':
    os.makedirs('/home/yzy/mmocr/datasets/cache', exist_ok=True)
    main()


