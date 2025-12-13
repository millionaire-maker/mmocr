import argparse
from mmengine.config import Config
import os
from mmocr.datasets.preparers.data_preparer import DatasetPreparer


def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'cache'), exist_ok=True)


def _fix_known_urls(cfg: Config, name: str):
    # 修复已知 308 重定向或不可直连的链接
    # 1) MJSynth: thor.robots → www.robots
    if name.lower() == 'mjsynth':
        tp = cfg.get('train_preparer', {}).get('obtainer', {})
        files = tp.get('files', [])
        for f in files:
            url = f.get('url', '')
            if 'thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz' in url:
                f['url'] = url.replace('thor.robots.ox.ac.uk', 'www.robots.ox.ac.uk')
    return cfg


def prepare_one(name: str, dataset_zoo_path: str, out_root: str, splits=None):
    cfg_path = f"{dataset_zoo_path}/{name}/textrecog.py"
    cfg = Config.fromfile(cfg_path)
    # MJSynth 配方内部解压会创建一个 mjsynth 子目录，避免重复路径导致交互
    if name.lower() == 'mjsynth':
        cfg.data_root = f"{out_root}"
    else:
        cfg.data_root = f"{out_root}/{name}"
    _ensure_dirs(out_root)
    cfg = _fix_known_urls(cfg, name)
    preparer = DatasetPreparer.from_file(cfg)
    if splits is None:
        splits = []
        if cfg.get('train_preparer', None) is not None:
            splits.append('train')
        if cfg.get('test_preparer', None) is not None:
            splits.append('test')
    preparer.run(splits=splits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', help='dataset names under dataset_zoo')
    parser.add_argument('--dataset-zoo-path', default='/home/yzy/mmocr/dataset_zoo')
    parser.add_argument('--out-root', default='/home/yzy/mmocr/data')
    parser.add_argument('--splits', nargs='*', default=None)
    args = parser.parse_args()

    for name in args.datasets:
        print(f"==== Preparing {name} ====")
        prepare_one(name, args.dataset_zoo_path, args.out_root, args.splits)
        print(f"==== Done {name} ====")


if __name__ == '__main__':
    main()


