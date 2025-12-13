import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='从配置构建 dataloader 并迭代若干 batch 做 sanity check')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument(
        '--num-batches',
        type=int,
        default=1,
        help='迭代多少个 batch（默认 1）')
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    cfg.setdefault('work_dir', './work_dirs/tmp_try_build')
    runner = Runner.from_cfg(cfg)
    loader = runner.build_dataloader(cfg.train_dataloader)
    for i, data in enumerate(loader):
        print(f'[OK] batch {i} keys: {list(data.keys())}')
        if i + 1 >= args.num_batches:
            break


if __name__ == '__main__':
    main()
