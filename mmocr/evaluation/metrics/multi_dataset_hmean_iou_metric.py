# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from mmocr.evaluation.functional import compute_hmean
from mmocr.registry import METRICS

from .hmean_iou_metric import HmeanIOUMetric


@METRICS.register_module()
class MultiDatasetHmeanIOUMetric(HmeanIOUMetric):
    """在拼接验证集(ConcatDataset)上同时输出总分与分数据集得分的 HmeanIOU 指标。

    该指标会保留原始的 ``icdar/hmean``（用于 best ckpt / early stop），并额外输出：

    - ``icdar/<dataset_name>/precision``
    - ``icdar/<dataset_name>/recall``
    - ``icdar/<dataset_name>/hmean``

    通过样本的 ``metainfo.img_path`` 与 ``dataset_prefixes`` 的路径前缀/目录名进行匹配
    来判断样本属于哪个子数据集。

    Args:
        dataset_prefixes (dict): 子数据集名称到路径前缀(或列表)的映射，用于根据
            ``img_path`` 归属分组。例如：
            ``dict(art='data/art_mmocr', rctw='data/rctw17_mmocr')``。
        unknown_label (str): 无法匹配时的分组名。Defaults to 'unknown'.
        compute_unknown (bool): 是否对 unknown 分组也计算指标。Defaults to False.
    """

    def __init__(self,
                 dataset_prefixes: Dict[str, Union[str, Sequence[str]]],
                 match_iou_thr: float = 0.5,
                 ignore_precision_thr: float = 0.5,
                 pred_score_thrs: Dict = dict(start=0.3, stop=0.9, step=0.1),
                 strategy: str = 'vanilla',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 unknown_label: str = 'unknown',
                 compute_unknown: bool = False) -> None:
        super().__init__(
            match_iou_thr=match_iou_thr,
            ignore_precision_thr=ignore_precision_thr,
            pred_score_thrs=pred_score_thrs,
            strategy=strategy,
            collect_device=collect_device,
            prefix=prefix)
        self.unknown_label = unknown_label
        self.compute_unknown = compute_unknown

        normalized: Dict[str, List[str]] = {}
        for name, prefixes in dataset_prefixes.items():
            if isinstance(prefixes, str):
                normalized[name] = [prefixes]
            else:
                normalized[name] = list(prefixes)
        self.dataset_prefixes = normalized
        self.dataset_names = list(normalized.keys())

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        prev_len = len(self.results)
        super().process(data_batch=data_batch, data_samples=data_samples)

        new_results = self.results[prev_len:]
        for result, data_sample in zip(new_results, data_samples):
            metainfo = data_sample.get('metainfo', {})
            img_path = metainfo.get('img_path', '') if isinstance(
                metainfo, dict) else ''
            dataset_name = self._infer_dataset_name(img_path)
            result['dataset_name'] = dataset_name
            result['img_path'] = img_path

    def compute_metrics(self, results: List[Dict]) -> Dict:
        logger: MMLogger = MMLogger.get_current_instance()

        # 先计算“总分”，保留与原版 HmeanIOUMetric 一致的日志输出与 best-thr 搜索。
        metrics = super().compute_metrics(results)

        grouped: Dict[str, List[Dict]] = {}
        for res in results:
            name = res.get('dataset_name', self.unknown_label)
            grouped.setdefault(name, []).append(res)

        for name in self.dataset_names:
            subset = grouped.get(name, [])
            if not subset:
                logger.warning(
                    f'MultiDatasetHmeanIOUMetric: 未找到匹配 "{name}" 的样本，'
                    '请检查 dataset_prefixes 与 img_path 是否一致。')
                continue
            sub_metrics = self._compute_metrics_no_log(subset)
            for k, v in sub_metrics.items():
                metrics[f'{name}/{k}'] = v
            logger.info(
                f'[{name}] hmean: {sub_metrics["hmean"]:.4f}, '
                f'precision: {sub_metrics["precision"]:.4f}, '
                f'recall: {sub_metrics["recall"]:.4f}')

        unknown_subset = grouped.get(self.unknown_label, [])
        if unknown_subset and not self.compute_unknown:
            logger.warning(
                f'MultiDatasetHmeanIOUMetric: 有 {len(unknown_subset)} 个样本'
                f'无法匹配到任何 dataset_prefixes（标记为 "{self.unknown_label}"）。'
            )
        elif unknown_subset and self.compute_unknown:
            sub_metrics = self._compute_metrics_no_log(unknown_subset)
            for k, v in sub_metrics.items():
                metrics[f'{self.unknown_label}/{k}'] = v
            logger.info(
                f'[{self.unknown_label}] hmean: {sub_metrics["hmean"]:.4f}, '
                f'precision: {sub_metrics["precision"]:.4f}, '
                f'recall: {sub_metrics["recall"]:.4f}')

        return metrics

    def _infer_dataset_name(self, img_path: str) -> str:
        if not isinstance(img_path, str) or not img_path:
            return self.unknown_label

        img_path_abs = osp.abspath(img_path)
        img_path_parts = set(osp.normpath(img_path).split(osp.sep))

        best_score = -1
        best_name = self.unknown_label

        for name, prefixes in self.dataset_prefixes.items():
            for prefix in prefixes:
                if not prefix:
                    continue
                prefix_norm = osp.normpath(prefix)
                prefix_abs = osp.abspath(prefix_norm)

                # 1) 最可靠：按目录前缀判断（绝对路径）
                try:
                    if osp.commonpath([img_path_abs, prefix_abs]) == prefix_abs:
                        score = len(prefix_abs)
                        if score > best_score:
                            best_score = score
                            best_name = name
                        continue
                except Exception:
                    pass

                # 2) 兜底：按 data_root 目录名（basename）判断
                base = osp.basename(prefix_norm)
                if base and base in img_path_parts:
                    score = len(base)
                    if score > best_score:
                        best_score = score
                        best_name = name

        return best_name

    def _compute_metrics_no_log(self, results: List[Dict]) -> Dict:
        """复用 HmeanIOU 的计算逻辑，但不打印逐阈值日志。"""
        best_eval_results = dict(hmean=-1.0, precision=0.0, recall=0.0)

        dataset_pred_num = np.zeros_like(self.pred_score_thrs)
        dataset_hit_num = np.zeros_like(self.pred_score_thrs)
        dataset_gt_num = 0

        for result in results:
            iou_metric = result['iou_metric']  # (gt_num, pred_num)
            pred_scores = result['pred_scores']  # (pred_num)
            dataset_gt_num += iou_metric.shape[0]

            for i, pred_score_thr in enumerate(self.pred_score_thrs):
                pred_ignore_flags = pred_scores < pred_score_thr
                matched_metric = iou_metric[:, ~pred_ignore_flags] > \
                    self.match_iou_thr

                if self.strategy == 'max_matching':
                    csr_matched_metric = csr_matrix(matched_metric)
                    matched_preds = maximum_bipartite_matching(
                        csr_matched_metric, perm_type='row')
                    dataset_hit_num[i] += np.sum(matched_preds != -1)
                else:
                    matched_gt_indexes = set()
                    matched_pred_indexes = set()
                    for gt_idx, pred_idx in zip(*np.nonzero(matched_metric)):
                        if gt_idx in matched_gt_indexes or \
                                pred_idx in matched_pred_indexes:
                            continue
                        matched_gt_indexes.add(gt_idx)
                        matched_pred_indexes.add(pred_idx)
                    dataset_hit_num[i] += len(matched_gt_indexes)

                dataset_pred_num[i] += np.sum(~pred_ignore_flags)

        for i, _ in enumerate(self.pred_score_thrs):
            recall, precision, hmean = compute_hmean(
                int(dataset_hit_num[i]), int(dataset_hit_num[i]),
                int(dataset_gt_num), int(dataset_pred_num[i]))
            if hmean > best_eval_results['hmean']:
                best_eval_results = dict(
                    precision=precision, recall=recall, hmean=hmean)

        return best_eval_results
