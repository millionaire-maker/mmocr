#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import json
import os.path as osp
from pathlib import Path
from typing import Dict, Optional

from mmengine.config import Config
from mmengine.fileio import dump, load
from mmengine.runner import Runner
from mmengine.structures import InstanceData

from mmocr.registry import METRICS
from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='离线评测保存的 pkl 预测（复用 MultiDatasetHmeanIOUMetric）')
    parser.add_argument('--config', required=True, help='Stage-1 config 路径')
    parser.add_argument('--pred-pkl', required=True, help='预测 pkl（stage1/refined）')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out-json', required=True, help='输出 metrics.json')
    return parser.parse_args()


def _get_img_path(data_sample) -> str:
    if data_sample is None:
        return ''
    if isinstance(data_sample, dict):
        img_path = data_sample.get('img_path', '')
        if img_path:
            return img_path
        metainfo = data_sample.get('metainfo', {})
        if isinstance(metainfo, dict):
            return metainfo.get('img_path', '')
        return ''
    metainfo = getattr(data_sample, 'metainfo', None)
    if isinstance(metainfo, dict) and metainfo.get('img_path', ''):
        return metainfo['img_path']
    if hasattr(data_sample, 'get'):
        img_path = data_sample.get('img_path', '')
        if img_path:
            return img_path
        metainfo = data_sample.get('metainfo', {})
        if isinstance(metainfo, dict):
            return metainfo.get('img_path', '')
    return ''


def _pick_metric_cfg(evaluator_cfg):
    # evaluator_cfg can be dict or list[dict]
    if isinstance(evaluator_cfg, (list, tuple)):
        for cfg in evaluator_cfg:
            if not isinstance(cfg, dict):
                continue
            t = str(cfg.get('type', ''))
            if 'MultiDatasetHmeanIOUMetric' in t or 'HmeanIOUMetric' in t:
                return cfg
        # fallback to first dict
        for cfg in evaluator_cfg:
            if isinstance(cfg, dict):
                return cfg
        raise ValueError('evaluator 为空，或无可用 metric 配置')
    if isinstance(evaluator_cfg, dict):
        return evaluator_cfg
    raise TypeError('val_evaluator/test_evaluator 需为 dict 或 list[dict]')


def main():
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    if args.split == 'val':
        dataloader_cfg = cfg.val_dataloader
        evaluator_cfg = cfg.val_evaluator
    else:
        dataloader_cfg = cfg.test_dataloader
        evaluator_cfg = cfg.test_evaluator

    dataloader = Runner.build_dataloader(dataloader_cfg, seed=0)

    metric_cfg = _pick_metric_cfg(evaluator_cfg)
    metric = METRICS.build(metric_cfg)
    if hasattr(dataloader.dataset, 'metainfo'):
        metric.dataset_meta = dataloader.dataset.metainfo

    preds = load(args.pred_pkl)
    if not isinstance(preds, list):
        raise TypeError('pred pkl 期望为 list[DataSample]')

    pred_map: Dict[str, object] = {}
    for pred in preds:
        img_path = _get_img_path(pred)
        if not img_path:
            continue
        key = osp.abspath(osp.normpath(img_path))
        if hasattr(pred, 'get'):
            pred_map[key] = pred.get('pred_instances')
        elif isinstance(pred, dict):
            pred_map[key] = pred.get('pred_instances')

    missing = 0
    for data_batch in dataloader:
        data_samples = data_batch.get('data_samples')
        if data_samples is None:
            continue
        # batch_size=1 by default, but keep generic
        for ds in data_samples:
            img_path = _get_img_path(ds)
            key = osp.abspath(osp.normpath(img_path))
            pred_instances = pred_map.get(key, None)
            if pred_instances is None:
                missing += 1
                empty = InstanceData()
                empty.polygons = []
                empty.scores = []
                ds.pred_instances = empty
            else:
                ds.pred_instances = pred_instances
        metric.process(data_batch=data_batch, data_samples=data_samples)

    metrics = metric.evaluate(len(dataloader.dataset))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    dump(metrics, str(out_json))

    print(f'[EVAL] missing_pred_samples={missing}')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
