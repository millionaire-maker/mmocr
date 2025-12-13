# 评测命令与日志清单

| 日期 | 模型/数据集 | 命令 | 日志位置 | 备注 |
| --- | --- | --- | --- | --- |
| 2025-11-18 15:22 | FCENet 基线（ICDAR2015） | `python workdir_det/scripts/run_eval.py --config workdir_det/fcenet_cfg.py --checkpoint workdir_det/fcenet/fcenet/best_icdar_hmean_epoch_220.pth --work-dir workdir_det/fcenet --datasets icdar2015 --viz-root workdir_det/fcenet --metrics-root workdir_det/fcenet --sample-count 50` | `workdir_det/fcenet/20251118_145846/20251118_145846.log` | GPU=RTX4060，需预先 `pip install rapidfuzz imgaug` |
| 2025-11-18 15:36 | FCENet 基线（Total-Text） | 同上但 `--datasets totaltext` | `workdir_det/fcenet/20251118_155642/20251118_155642.log` | CUDA 运行，viz_totaltext/_sample 已更新 |
| 2025-11-18 16:06 | DBNet 基线（ICDAR2015） | `LD_LIBRARY_PATH=/home/yzy/anaconda3/lib:$LD_LIBRARY_PATH python workdir_det/scripts/run_eval.py --config workdir_det/dbnet_cfg.py --checkpoint workdir_det/dbnet/dbnet/best_icdar_hmean_epoch_140.pth --work-dir workdir_det/dbnet --datasets icdar2015 --viz-root workdir_det/dbnet --metrics-root workdir_det/dbnet --sample-count 50` | `workdir_det/dbnet/20251118_160601/20251118_160601.log` | 解决 SciPy/GLIBCXX 报错，产出 `viz_icdar2015(_sample)` |
| 2025-11-18 16:11 | DBNet 基线（CTW1500 + Total-Text） | `LD_LIBRARY_PATH=/home/yzy/anaconda3/lib:$LD_LIBRARY_PATH python workdir_det/scripts/run_eval.py --config workdir_det/dbnet_cfg.py --checkpoint workdir_det/dbnet/dbnet/best_icdar_hmean_epoch_140.pth --work-dir workdir_det/dbnet --datasets ctw1500 totaltext --viz-root workdir_det/dbnet --metrics-root workdir_det/dbnet --sample-count 50` | `workdir_det/dbnet/20251118_161106/20251118_161106.log` | 同步生成 `viz_ctw1500/_sample`、`viz_totaltext/_sample` |
| 2025-11-18 16:15 | 汇总刷新 | `python workdir_det/scripts/gen_summary.py` | 控制台输出 | 刷新 `reports/summary.md` |
