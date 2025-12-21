# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.evaluation import MultiDatasetHmeanIOUMetric
from mmocr.structures import TextDetDataSample


class TestMultiDatasetHmeanIOU(unittest.TestCase):

    def _build_perfect_sample(self, img_path: str) -> dict:
        data_sample = TextDetDataSample()
        data_sample.set_metainfo(dict(img_path=img_path))

        gt_instances = InstanceData()
        gt_instances.polygons = [torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1])]
        gt_instances.ignored = np.bool_([False])

        pred_instances = InstanceData()
        pred_instances.polygons = [torch.FloatTensor([0, 0, 1, 0, 1, 1, 0,
                                                      1])]
        pred_instances.scores = torch.FloatTensor([1.0])

        data_sample.gt_instances = gt_instances
        data_sample.pred_instances = pred_instances
        return data_sample.to_dict()

    def test_multi_dataset_hmean_iou_with_flat_img_path(self):
        # NOTE: mmengine Evaluator 会把 BaseDataElement 转成 dict，并把 metainfo
        # “拍平”为顶层 key（如 'img_path'）。该用例确保指标能从 dict 顶层读取
        # img_path 并正确分组计算分数据集指标。
        predictions = [
            self._build_perfect_sample('data/art_mmocr/imgs/a.jpg'),
            self._build_perfect_sample('data/rctw17_mmocr/imgs/b.jpg'),
        ]

        metric = MultiDatasetHmeanIOUMetric(
            dataset_prefixes=dict(
                art='data/art_mmocr',
                rctw='data/rctw17_mmocr',
            ))
        metric.process(None, predictions)
        eval_results = metric.evaluate(size=2)

        self.assertEqual(eval_results['icdar/hmean'], 1.0)
        self.assertEqual(eval_results['icdar/art/hmean'], 1.0)
        self.assertEqual(eval_results['icdar/rctw/hmean'], 1.0)

