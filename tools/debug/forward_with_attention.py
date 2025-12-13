import argparse
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mmengine.registry import init_default_scope
from mmocr.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='使用 ResNetWithAttention 做一次前向 sanity check')
    parser.add_argument(
        '--attention',
        default='se',
        choices=['se', 'cbam'],
        help='注意力类型')
    parser.add_argument(
        '--depth', type=int, default=50, help='ResNet depth，默认 50')
    parser.add_argument(
        '--size', type=int, default=640, help='输入短边尺寸，默认 640')
    return parser.parse_args()


def main():
    args = parse_args()
    init_default_scope('mmocr')
    backbone = MODELS.build(
        dict(
            type='ResNetWithAttention',
            depth=args.depth,
            out_indices=(0, 1, 2, 3),
            attention_type=args.attention))
    backbone.eval()
    dummy = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        outs = backbone(dummy)
    shapes = [tuple(o.shape) for o in outs]
    print(f'[OK] forward 成功，输出 shape: {shapes}')


if __name__ == '__main__':
    main()
