import argparse
import gc
import sys

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='自动探测可运行的最大 batch_size')
    parser.add_argument('--config', required=True, help='模型配置文件')
    parser.add_argument(
        '--candidates',
        type=int,
        nargs='+',
        default=[4, 3, 2, 1],
        help='尝试的 batch_size 列表，按顺序探测')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='通过命令行修改配置，格式同 tools/train.py')
    return parser.parse_args()


def try_batch(cfg: Config, bs: int):
    cfg.train_dataloader.batch_size = bs
    if 'auto_scale_lr' in cfg:
        cfg.auto_scale_lr['base_batch_size'] = bs
    cfg.setdefault('work_dir', './work_dirs/tmp_auto_bs')
    init_default_scope('mmocr')
    runner = Runner.from_cfg(cfg)
    loader = runner.build_dataloader(cfg.train_dataloader)
    data = next(iter(loader))
    runner.call_hook('before_train_epoch')
    data = runner.data_preprocessor(data, True)
    with runner.optim_wrapper.optim_context(data):
        with runner.optim_wrapper.optim_context(model=runner.model):
            runner.model.train()
            loss = runner.model.train_step(data)
            loss_value = sum(v.item() for v in loss.values())
    del runner
    gc.collect()
    return loss_value


def main():
    args = parse_args()
    base_cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        base_cfg.merge_from_dict(args.cfg_options)
    for bs in args.candidates:
        cfg = base_cfg.copy()
        try:
            loss = try_batch(cfg, bs)
            print(f'[OK] batch_size={bs} 运行成功，loss={loss:.4f}')
            return
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'[OOM] batch_size={bs}，尝试更小 batch')
            else:
                print(f'[FAIL] batch_size={bs} 异常: {e}')
        except Exception as e:  # noqa
            print(f'[FAIL] batch_size={bs} 异常: {e}')
    print('[WARN] 所有候选 batch_size 均失败，请检查数据与模型设置')


if __name__ == '__main__':
    sys.exit(main())
