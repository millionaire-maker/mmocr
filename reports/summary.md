# 文本检测评测汇总

生成时间：2025-11-19 10:52:13


## 指标总览（Precision / Recall / Hmean）

### ctw1500
- DBNet: P=0.7717, R=0.6603, F1=0.7117
- FCENet: P=0.8608, R=0.8144, F1=0.8369
- DBNet-FT-CTW1500: P=0.7666, R=0.6499, F1=0.7034

### icdar2015
- DBNet: P=0.5949, R=0.3698, F1=0.4561
- FCENet: P=0.3759, R=0.2691, F1=0.3137
- DBNet-FT-IC15: P=0.8543, R=0.7650, F1=0.8072
- FCENet-FT-IC15: P=0.8311, R=0.7323, F1=0.7786

### totaltext
- DBNet: P=0.5756, R=0.3264, F1=0.4166
- FCENet: P=0.4796, R=0.3025, F1=0.3710
- DBNet-FT-TotalText: P=0.8500, R=0.8158, F1=0.8325
- FCENet-FT-TotalText: P=0.8804, R=0.8239, F1=0.8512

## 可视化与日志

- DBNet 可视化：workdir_det/dbnet/viz_* / viz_*_sample
- FCENet 可视化：workdir_det/fcenet/viz_* / viz_*_sample
- Finetune 可视化：workdir_det/*_ft_*/viz_* / viz_*_sample
- 评测日志：workdir_det/*/eval_all.log, eval_totaltext.log
