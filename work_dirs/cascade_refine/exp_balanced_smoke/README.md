# Cascade Refine (DBNet++ -> FCENet)

本实验实现 **离线** coarse-to-fine gated refine：

- Stage-1：DBNet++（coarse），由 `tools/test.py --save-preds` 生成预测 `pkl`
- Stage-2：FCENet（fine），对 Stage-1 的部分实例裁 patch、推理、回映射、融合/NMS
- 输出：refine_ratio sweep 的 **精度-速度 trade-off 曲线**（overall + art/rctw/rects）

## 一行命令（可直接复制）

说明：

- 所有命令均使用 `conda run -n openmmlab ...`；每条命令都是**单行**（无反斜杠续行）
- 建议固定 `CUDA_VISIBLE_DEVICES=0`

1) baseline 生成 Stage-1 pkl（全量）：

`CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python /home/yzy/mmocr/tools/test.py /home/yzy/mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py /home/yzy/mmocr/work_dirs/dbnetpp_r50_finetune_art_rctw_rects/best_icdar_hmean_epoch_69.pth --work-dir /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/stage1_preds --save-preds --cfg-options model.det_head.postprocessor.text_repr_type=poly model.det_head.postprocessor.unclip_ratio=1.8 model.det_head.postprocessor.epsilon_ratio=0.002`

2) baseline 离线评测（全量）：

`conda run -n openmmlab python /home/yzy/mmocr/tools/cascade/eval_saved_preds.py --config /home/yzy/mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py --pred-pkl /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/stage1_preds/best_icdar_hmean_epoch_69.pth_predictions.pkl --split val --device cpu --out-json /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/stage1_preds/baseline_metrics.json`

3) refine smoke（max-images=20；默认 dry-run，不依赖 Stage-2 权重）：

`CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python /home/yzy/mmocr/tools/cascade/gated_refine_fcenet.py --stage1-pred-pkl /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/stage1_preds/best_icdar_hmean_epoch_69.pth_predictions.pkl --out-pkl /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/refine_smoke_ratio_0.1/refined_predictions.pkl --stage2-config /home/yzy/mmocr/configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py --device cuda:0 --gating-mode ratio --refine-ratio 0.1 --max-images 20 --dry-run-refine`

4) sweep smoke（ratios=0/0.1，balanced subsample=2；默认 dry-run）：

`CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python /home/yzy/mmocr/tools/cascade/run_sweep_gated_refine.py --stage1-config /home/yzy/mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py --stage1-ckpt /home/yzy/mmocr/work_dirs/dbnetpp_r50_finetune_art_rctw_rects/best_icdar_hmean_epoch_69.pth --stage2-config /home/yzy/mmocr/configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py --ratios 0 0.1 --out-dir /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke --per-dataset 2`

5) sweep full（ratios=0/0.1/0.3/0.5/1.0，全量；需要 Stage-2 权重才有真实提升）：

`CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python /home/yzy/mmocr/tools/cascade/run_sweep_gated_refine.py --stage1-config /home/yzy/mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py --stage1-ckpt /home/yzy/mmocr/work_dirs/dbnetpp_r50_finetune_art_rctw_rects/best_icdar_hmean_epoch_69.pth --stage2-config /home/yzy/mmocr/configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py --stage2-ckpt <path/to/fcenet_ckpt.pth> --ratios 0 0.1 0.3 0.5 1.0 --out-dir /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke`

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

`CUDA_VISIBLE_DEVICES=0 conda run -n openmmlab python /home/yzy/mmocr/tools/cascade/gated_refine_fcenet.py --stage1-pred-pkl /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/stage1_preds/best_icdar_hmean_epoch_69.pth_predictions.pkl --out-pkl /home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/debug_vis/refined_predictions.pkl --stage2-config /home/yzy/mmocr/configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py --device cuda:0 --gating-mode ratio --refine-ratio 0.1 --save-debug-vis --max-images 20 --dry-run-refine`

验收要点（建议顺序）：

1) 看 `/home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/debug_vis/vis/` 下的原图叠图：coarse（红） vs 最终（绿），确认绿框没有整体偏移/旋转错位（回映射正确）。
2) 看 `/home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke/debug_vis/vis/patches/` 下的 patch：文本应该在 patch 内居中，polygons 不应被裁断（裁 patch 稳健）。
3) 同一文本区域重复框应被抑制（NMS 生效）。
4) `refine_summary.json` 中 `refine_fallback` 不应为 0（在真实 Stage-2 下常见），且失败时最终结果仍应保留 coarse（fallback 生效）。

有 Stage-2 权重时：去掉 `--dry-run-refine` 并补上 `--stage2-ckpt <path/to/fcenet_ckpt.pth>`，即可验收“真实 refine”效果。

## 输出目录结构

以本次 sweep 输出目录 `/home/yzy/mmocr/work_dirs/cascade_refine/exp_balanced_smoke` 为例：

```
exp_balanced_smoke/
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
