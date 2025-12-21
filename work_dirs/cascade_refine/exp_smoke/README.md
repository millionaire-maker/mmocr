# Cascade Refine (DBNet++ -> FCENet)

本实验实现 **离线** coarse-to-fine gated refine：

- Stage-1：DBNet++（coarse），由 `tools/test.py --save-preds` 生成预测 `pkl`
- Stage-2：FCENet（fine），对 Stage-1 的部分实例裁 patch、推理、回映射、融合/NMS
- 输出：refine_ratio sweep 的 **精度-速度 trade-off 曲线**（overall + art/rctw/rects）

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

## 输出目录结构

以本次 sweep 输出目录 `/home/yzy/mmocr/work_dirs/cascade_refine/exp_smoke` 为例：

```
exp_smoke/
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
  README.md
```

## 说明：Stage-2 权重

本仓库默认不包含 FCENet ckpt。若本地没有权重：

- refine 可用 `--dry-run-refine` 模式跑通裁剪/映射/NMS/fallback（但不会带来精度提升）
- 需要真实 refine 时，请下载/准备 FCENet 权重文件并传给 `--stage2-ckpt`

可参考 `configs/textdet/fcenet/README.md` 中的官方权重下载链接（OpenMMLab download）。
