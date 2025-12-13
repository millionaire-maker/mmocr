# 检测硬案例清单（误检/漏检）

> 目录中保留了每个模型×数据集×FP/FN 的 20 张示例，以下对典型问题做归纳。

## DBNet

- **CTW1500**  
  - 误检：`workdir_det/dbnet/viz_hardcases/ctw1500_fp/`。弯曲路面纹理或护栏阴影与文字形似（如 `1004_412.png`、`1010_192.png`）。  
  - 漏检：`workdir_det/dbnet/viz_hardcases/ctw1500_fn/`。长弧形句子被切断、遮挡或亮度过低（`1022_376.png`、`1036_353.png`）。

- **ICDAR2015**  
  - 误检：`workdir_det/dbnet/viz_hardcases/icdar2015_fp/`。高速场景路牌反光、车辆纹理带来伪响应（`img_101_3.png` 等）。  
  - 漏检：`workdir_det/dbnet/viz_hardcases/icdar2015_fn/`。运动模糊或极度倾斜的数字区域信号不足（`img_121_25.png`、`img_130_35.png`）。

- **Total-Text**  
  - 误检：`workdir_det/dbnet/viz_hardcases/totaltext_fp/`。地面砖缝/喷绘线条形成闭合结构（`img1000_3.png`、`img1098_12.png`）。  
  - 漏检：`workdir_det/dbnet/viz_hardcases/totaltext_fn/`。低对比度手写弯曲文本与背景混合（`img1197_21.png`、`img1210_32.png`）。

## FCENet

- **CTW1500**  
  - 误检：`workdir_det/fcenet/viz_hardcases/ctw1500_fp/`。复杂地面纹理/屋顶曲线仍被识别（`1001_169.png` 等）。  
  - 漏检：`workdir_det/fcenet/viz_hardcases/ctw1500_fn/`。细长曲线因响应断裂或交叉遮挡而丢失（`1023_112.png`、`1034_142.png`）。

- **ICDAR2015**  
  - 误检：`workdir_det/fcenet/viz_hardcases/icdar2015_fp/`。与 DBNet 相同，多源自车灯/广告牌反光（`img_107_9.png`、`img_112_15.png`）。  
  - 漏检：`workdir_det/fcenet/viz_hardcases/icdar2015_fn/`。运动模糊或倾斜字体未闭合（`img_124_28.png`、`img_132_37.png`）。

- **Total-Text**  
  - 误检：`workdir_det/fcenet/viz_hardcases/totaltext_fp/`。地面条纹、喷绘线条依旧触发（`img1091_5.png`、`img1196_20.png`）。  
  - 漏检：`workdir_det/fcenet/viz_hardcases/totaltext_fn/`。手写弯曲文本被阴影/高光覆盖（`img1202_26.png`、`img1215_39.png`）。
