#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import csv
import json
import os
import os.path as osp
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from mmengine.fileio import dump, load


STAGE1_FIXED_CFG_OPTIONS = [
    'model.det_head.postprocessor.text_repr_type=poly',
    'model.det_head.postprocessor.unclip_ratio=1.8',
    'model.det_head.postprocessor.epsilon_ratio=0.002',
]


def _balanced_indices_cfg_options(per_dataset: int) -> List[str]:
    opts: List[str] = []
    for split_key in ['val_dataloader', 'test_dataloader']:
        for i in range(3):
            opts.append(f'{split_key}.dataset.datasets.{i}.indices={per_dataset}')
    return opts


def parse_args():
    parser = argparse.ArgumentParser(
        description='refine_ratio sweep + trade-off 曲线产物（离线 gated refine）')
    parser.add_argument('--stage1-config', required=True)
    parser.add_argument('--stage1-ckpt', required=True)
    parser.add_argument('--stage2-config', required=True)
    parser.add_argument('--stage2-ckpt', default='')
    parser.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        default=[0, 0.1, 0.3, 0.5, 1.0],
        help='refine_ratio 列表')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument(
        '--per-dataset',
        type=int,
        default=0,
        help='balanced 子采样：每个子数据集取前 N 张（ConcatDataset 顺序）')
    return parser.parse_args()


def _run(cmd: List[str], cwd: str) -> None:
    print('[CMD]', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _read_metric_json(path: Path) -> Dict:
    data = load(str(path))
    if not isinstance(data, dict):
        raise TypeError(f'metrics json 期望 dict: {path}')
    return data


def _ensure_plots(out_dir: Path, ratios: List[float], metrics_rows: List[Dict],
                  speed_rows: List[Dict]) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # hmean vs ratio
    def _get(xs, key):
        ys = []
        for x in xs:
            v = x.get(key, None)
            try:
                ys.append(float(v))
            except Exception:
                ys.append(float('nan'))
        return ys

    plt.figure()
    plt.plot(ratios, _get(metrics_rows, 'icdar/hmean'), label='overall')
    if any(r.get('icdar/art/hmean') is not None for r in metrics_rows):
        plt.plot(ratios, _get(metrics_rows, 'icdar/art/hmean'), label='art')
    if any(r.get('icdar/rctw/hmean') is not None for r in metrics_rows):
        plt.plot(ratios, _get(metrics_rows, 'icdar/rctw/hmean'), label='rctw')
    if any(r.get('icdar/rects/hmean') is not None for r in metrics_rows):
        plt.plot(
            ratios, _get(metrics_rows, 'icdar/rects/hmean'), label='rects')
    plt.xlabel('refine_ratio')
    plt.ylabel('hmean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(plots_dir / 'hmean_vs_ratio.png'), dpi=150)
    plt.close()

    # ms/img vs ratio
    plt.figure()
    plt.plot(ratios, [float(r['ms_per_img']) for r in speed_rows])
    plt.xlabel('refine_ratio')
    plt.ylabel('ms/img')
    plt.tight_layout()
    plt.savefig(str(plots_dir / 'ms_per_img_vs_ratio.png'), dpi=150)
    plt.close()


def _write_readme(out_dir: Path, args) -> None:
    stage2_ckpt_note = args.stage2_ckpt if args.stage2_ckpt else '<path/to/fcenet_ckpt.pth>'
    smoke_per_dataset = int(args.per_dataset) if getattr(args, 'per_dataset', 0) else 20
    readme = f"""# Cascade Refine (DBNet++ -> FCENet)

本实验实现 **离线** coarse-to-fine gated refine：

- Stage-1：DBNet++（coarse），由 `tools/test.py --save-preds` 生成预测 `pkl`
- Stage-2：FCENet（fine），对 Stage-1 的部分实例裁 patch、推理、回映射、融合/NMS
- 输出：refine_ratio sweep 的 **精度-速度 trade-off 曲线**（overall + art/rctw/rects）

## 一行命令（可直接复制）

说明：

- 所有命令均使用 `conda run -n openmmlab ...`；每条命令都是**单行**（无反斜杠续行）
- 建议固定 `CUDA_VISIBLE_DEVICES=0`

1) baseline 生成 Stage-1 pkl（全量）：

`cd /home/yzy/mmocr && CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python tools/test.py {args.stage1_config} {args.stage1_ckpt} --work-dir {out_dir}/stage1_preds --save-preds --cfg-options model.det_head.postprocessor.text_repr_type=poly model.det_head.postprocessor.unclip_ratio=1.8 model.det_head.postprocessor.epsilon_ratio=0.002`

2) baseline 离线评测（全量）：

`cd /home/yzy/mmocr && conda run -n openmmlab python tools/cascade/eval_saved_preds.py --config {args.stage1_config} --pred-pkl {out_dir}/stage1_preds/{Path(args.stage1_ckpt).name}_predictions.pkl --split val --device cpu --out-json {out_dir}/stage1_preds/baseline_metrics.json`

3) refine smoke（max-images=20；默认 dry-run，不依赖 Stage-2 权重）：

`cd /home/yzy/mmocr && CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python tools/cascade/gated_refine_fcenet.py --stage1-pred-pkl {out_dir}/stage1_preds/{Path(args.stage1_ckpt).name}_predictions.pkl --out-pkl {out_dir}/refine_smoke_ratio_0.1/refined_predictions.pkl --stage2-config {args.stage2_config} --device cuda:0 --gating-mode ratio --refine-ratio 0.1 --max-images 20 --dry-run-refine`

4) sweep smoke（ratios=0/0.1，balanced subsample={smoke_per_dataset}；默认 dry-run）：

`cd /home/yzy/mmocr && CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python tools/cascade/run_sweep_gated_refine.py --stage1-config {args.stage1_config} --stage1-ckpt {args.stage1_ckpt} --stage2-config {args.stage2_config} --ratios 0 0.1 --out-dir {out_dir} --per-dataset {smoke_per_dataset}`

5) sweep full（ratios=0/0.1/0.3/0.5/1.0，全量；需要 Stage-2 权重才有真实提升）：

`cd /home/yzy/mmocr && CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python tools/cascade/run_sweep_gated_refine.py --stage1-config {args.stage1_config} --stage1-ckpt {args.stage1_ckpt} --stage2-config {args.stage2_config} --stage2-ckpt {stage2_ckpt_note} --ratios 0 0.1 0.3 0.5 1.0 --out-dir {out_dir}`

## 实验依据（已验证结论）

1) Stage-1 后处理路径：`cfg.model.det_head.postprocessor` 存在且默认是 `quad`。
2) 单阶段调参：`text_repr_type=poly + unclip_ratio=1.8` 优于默认 `quad`。
3) 在 `(poly, u=1.8)` 下，`epsilon_ratio=0.002` 最优，因此 Stage-1 固定：

`text_repr_type=poly, unclip_ratio=1.8, epsilon_ratio=0.002`

## 关键实现

### Gating（离线选框）

- 统计每个 coarse polygon 的 `score / minAreaRect 长宽比 / area`
- 候选集合：`(score in [low,high]) OR (aspect_ratio > thr)`
- `gating-mode=ratio`：按 `abs(score-0.5)` 从小到大（越接近 0.5 越“疑难”）选前 `refine_ratio`
- 每张图最多 refine `topk-per-image`

### Patch 裁剪与回映射（稳健性补丁）

- polygon -> `cv2.minAreaRect` 得到旋转 ROI
- ROI 以 `expand_ratio` 扩张
- `cv2.getPerspectiveTransform` + `cv2.warpPerspective` 裁出 patch（方向一致）
- patch 统一缩放到长边 `max_patch_long_edge=1024`，并 pad 到 `pad_divisor=32` 的倍数
- Stage-2 在 patch 坐标输出 polygons，再用逆变换矩阵映射回原图坐标

### Fallback（防止劣化）

对每个被 refine 的实例：

- FCENet 无输出 / 映射失败 -> 回退 Stage-1 原框
- refine 后 polygon 面积异常（<50 或 > minAreaRect_area*1.5）-> 回退
- refine 后与原框 IoU < 0.1 -> 回退

### Polygon NMS

对整图 polygons 做 greedy NMS（基于 `mmocr.utils.poly_iou`），阈值 `nms-iou-thr`。

## Debug Vis 快速验收（裁 patch / 回映射 / NMS / fallback）

建议在少量样本上启用 `--save-debug-vis` 快速看对不对（脚本默认只保存前 20 张图，避免爆目录）：

`cd /home/yzy/mmocr && CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python tools/cascade/gated_refine_fcenet.py --stage1-pred-pkl {out_dir}/stage1_preds/{Path(args.stage1_ckpt).name}_predictions.pkl --out-pkl {out_dir}/debug_vis/refined_predictions.pkl --stage2-config {args.stage2_config} --device cuda:0 --gating-mode ratio --refine-ratio 0.1 --save-debug-vis --max-images 20 --dry-run-refine`

验收要点（建议顺序）：

1) 看 `{out_dir}/debug_vis/vis/` 下的原图叠图：coarse（红） vs 最终（绿），确认绿框没有整体偏移/旋转错位（回映射正确）。
2) 看 `{out_dir}/debug_vis/vis/patches/` 下的 patch：文本应该在 patch 内居中，polygons 不应被裁断（裁 patch 稳健）。
3) 同一文本区域重复框应被抑制（NMS 生效）。
4) `refine_summary.json` 中 `refine_fallback` 不应为 0（在真实 Stage-2 下常见），且失败时最终结果仍应保留 coarse（fallback 生效）。

有 Stage-2 权重时：去掉 `--dry-run-refine` 并补上 `--stage2-ckpt {stage2_ckpt_note}`，即可验收“真实 refine”效果。

## 输出目录结构

以本次 sweep 输出目录 `{out_dir}` 为例：

```
{out_dir.name}/
  stage1_preds/
    <stage1_ckpt_basename>_predictions.pkl
    baseline_metrics.json
    *.log
  ratio_0/
    refined_predictions.pkl
    refine_summary.json
    metrics.json
  ratio_0.1/
    ...
  results.csv
  results.json
  plots/
    hmean_vs_ratio.png
    ms_per_img_vs_ratio.png
  debug_vis/
    refined_predictions.pkl
    refine_summary.json
    vis/
      *.jpg
      patches/
        *.jpg
  README.md
```

## 说明：Stage-2 权重

本仓库默认不包含 FCENet ckpt。若本地没有权重：

- refine 可用 `--dry-run-refine` 模式跑通裁剪/映射/NMS/fallback（但不会带来精度提升）
- 需要真实 refine 时，请下载/准备 FCENet 权重文件并传给 `--stage2-ckpt`

可参考 `/home/yzy/mmocr/configs/textdet/fcenet/README.md` 中的官方权重下载链接（OpenMMLab download）。
"""
    (out_dir / 'README.md').write_text(readme, encoding='utf-8')


def main():
    args = parse_args()
    if args.per_dataset and args.per_dataset > 0 and args.max_images and args.max_images > 0:
        raise ValueError('请不要同时使用 --per-dataset 与 --max-images（会导致评测/预测不一致）')

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = out_dir / 'stage1_preds'
    stage1_dir.mkdir(parents=True, exist_ok=True)

    stage1_pkl = stage1_dir / f'{Path(args.stage1_ckpt).name}_predictions.pkl'
    if not stage1_pkl.exists():
        extra_cfg_opts: List[str] = []
        if args.per_dataset and args.per_dataset > 0:
            extra_cfg_opts = _balanced_indices_cfg_options(args.per_dataset)
        cmd = [
            sys.executable, 'tools/test.py', args.stage1_config,
            args.stage1_ckpt, '--work-dir',
            str(stage1_dir), '--save-preds', '--cfg-options'
        ] + STAGE1_FIXED_CFG_OPTIONS + extra_cfg_opts
        _run(cmd, cwd=str(repo_root))
        if not stage1_pkl.exists():
            raise FileNotFoundError(
                f'未找到 stage1 pkl: {stage1_pkl}（检查 tools/test.py 输出）')

    baseline_metrics = stage1_dir / 'baseline_metrics.json'
    if not baseline_metrics.exists():
        cmd = [
            sys.executable, 'tools/cascade/eval_saved_preds.py', '--config',
            args.stage1_config, '--pred-pkl',
            str(stage1_pkl), '--split', 'val', '--device', 'cpu', '--out-json',
            str(baseline_metrics)
        ]
        if args.per_dataset and args.per_dataset > 0:
            cmd += ['--cfg-options'] + _balanced_indices_cfg_options(
                args.per_dataset)
        elif args.max_images and args.max_images > 0:
            cmd += ['--max-images', str(args.max_images)]
        _run(cmd, cwd=str(repo_root))

    ratios = [float(r) for r in args.ratios]
    results_rows: List[Dict] = []
    speed_rows: List[Dict] = []

    for ratio in ratios:
        ratio_dir = out_dir / f'ratio_{ratio}'
        ratio_dir.mkdir(parents=True, exist_ok=True)

        refined_pkl = ratio_dir / 'refined_predictions.pkl'
        summary_json = ratio_dir / 'refine_summary.json'
        metrics_json = ratio_dir / 'metrics.json'

        if not refined_pkl.exists():
            cmd = [
                sys.executable, 'tools/cascade/gated_refine_fcenet.py',
                '--stage1-pred-pkl',
                str(stage1_pkl),
                '--out-pkl',
                str(refined_pkl),
                '--stage2-config',
                args.stage2_config,
                '--device',
                'cuda:0',
                '--gating-mode',
                'ratio',
                '--refine-ratio',
                str(ratio),
                '--expand-ratio',
                '0.2',
                '--max-patch-long-edge',
                '1024',
                '--pad-divisor',
                '32',
                '--nms-iou-thr',
                '0.2',
            ]
            if args.max_images and args.max_images > 0:
                cmd += ['--max-images', str(args.max_images)]

            stage2_ckpt_exists = bool(args.stage2_ckpt) and osp.isfile(
                args.stage2_ckpt)
            if stage2_ckpt_exists:
                cmd += ['--stage2-ckpt', args.stage2_ckpt]
            else:
                cmd += ['--dry-run-refine']
                if args.stage2_ckpt:
                    cmd += ['--stage2-ckpt', args.stage2_ckpt]

            _run(cmd, cwd=str(repo_root))
            if not summary_json.exists():
                raise FileNotFoundError(f'缺少 refine_summary.json: {summary_json}')

        if not metrics_json.exists():
            cmd = [
                sys.executable, 'tools/cascade/eval_saved_preds.py', '--config',
                args.stage1_config, '--pred-pkl',
                str(refined_pkl), '--split', 'val', '--device', 'cpu',
                '--out-json', str(metrics_json)
            ]
            if args.per_dataset and args.per_dataset > 0:
                cmd += ['--cfg-options'] + _balanced_indices_cfg_options(
                    args.per_dataset)
            elif args.max_images and args.max_images > 0:
                cmd += ['--max-images', str(args.max_images)]
            _run(cmd, cwd=str(repo_root))

        m = _read_metric_json(metrics_json)
        row = {
            'ratio': ratio,
            'icdar/hmean': m.get('icdar/hmean', None),
            'icdar/art/hmean': m.get('icdar/art/hmean', None),
            'icdar/rctw/hmean': m.get('icdar/rctw/hmean', None),
            'icdar/rects/hmean': m.get('icdar/rects/hmean', None),
        }
        results_rows.append(row)

        s = load(str(summary_json))
        ms_per_img = None
        stage2_ms_per_patch = None
        if isinstance(s, dict):
            timing = s.get('timing_ms', {})
            if isinstance(timing, dict):
                ms_per_img = timing.get('avg_per_image', None)
                stage2_ms_per_patch = timing.get('stage2_avg_per_patch', None)
        speed_rows.append(
            dict(
                ratio=ratio,
                ms_per_img=ms_per_img if ms_per_img is not None else float('nan'),
                stage2_ms_per_patch=stage2_ms_per_patch
                if stage2_ms_per_patch is not None else float('nan'),
                dry_run=bool(s.get('dry_run_refine', False))
                if isinstance(s, dict) else False,
            ))

    # write results.json/csv
    results_json = {
        'stage1_pred_pkl': str(stage1_pkl),
        'baseline_metrics_json': str(baseline_metrics),
        'per_dataset': int(args.per_dataset),
        'ratios': ratios,
        'results': results_rows,
        'speed': speed_rows,
    }
    dump(results_json, str(out_dir / 'results.json'))

    with (out_dir / 'results.csv').open('w', newline='',
                                        encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'ratio',
                'icdar/hmean',
                'icdar/art/hmean',
                'icdar/rctw/hmean',
                'icdar/rects/hmean',
                'ms_per_img',
                'stage2_ms_per_patch',
                'dry_run',
            ],
        )
        writer.writeheader()
        for r, s in zip(results_rows, speed_rows):
            writer.writerow({
                **r,
                'ms_per_img': s['ms_per_img'],
                'stage2_ms_per_patch': s['stage2_ms_per_patch'],
                'dry_run': s['dry_run'],
            })

    _ensure_plots(out_dir, ratios, results_rows, speed_rows)
    _write_readme(out_dir, args)

    print('[DONE] out_dir =', out_dir)


if __name__ == '__main__':
    main()
