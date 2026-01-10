#!/usr/bin/env python3
import argparse
import os
from typing import Dict

import torch


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            continue
        out[k[len('module.'):]] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Export the RAW (non-EMA) model state_dict from a checkpoint saved with mmengine EMAHook.'
    )
    parser.add_argument('--in-ckpt', required=True, help='Input checkpoint path (e.g. epoch_*.pth).')
    parser.add_argument('--out-ckpt', default=None, help='Output checkpoint path. Defaults to "<in>.raw.pth".')
    parser.add_argument(
        '--keep-ema-state',
        action='store_true',
        help='Keep original "ema_state_dict" in the output checkpoint (default: drop it).',
    )
    args = parser.parse_args()

    in_ckpt = args.in_ckpt
    out_ckpt = args.out_ckpt or (in_ckpt[:-4] + '.raw.pth' if in_ckpt.endswith('.pth') else in_ckpt + '.raw.pth')

    if not os.path.exists(in_ckpt):
        raise FileNotFoundError(in_ckpt)

    ckpt = torch.load(in_ckpt, map_location='cpu')
    if 'ema_state_dict' not in ckpt:
        raise KeyError(
            f'"ema_state_dict" not found in {in_ckpt}. This checkpoint may not be saved with EMAHook.'
        )

    raw_state_dict = _strip_module_prefix(ckpt['ema_state_dict'])
    if not raw_state_dict:
        raise RuntimeError(
            f'Failed to extract raw state_dict from {in_ckpt}: no "module.*" keys in ema_state_dict.'
        )

    ckpt_out = dict(ckpt)
    ckpt_out['state_dict'] = raw_state_dict
    if not args.keep_ema_state:
        ckpt_out.pop('ema_state_dict', None)

    os.makedirs(os.path.dirname(os.path.abspath(out_ckpt)), exist_ok=True)
    torch.save(ckpt_out, out_ckpt)
    print(f'[OK] Wrote raw checkpoint: {out_ckpt}')


if __name__ == '__main__':
    main()

