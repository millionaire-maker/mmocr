# synth_chinese_ocr 配置修改与使用指南（面向当前仓库）

本指南对应：
- 合成器：`3rdparty/synth_chinese_ocr`
- Wrapper：`tools/recog_synth/run_synth_chinese_ocr.py`
- 默认配置：`3rdparty/synth_chinese_ocr/configs/my_cn_scene.yaml`

---

## 1. 两种“改配置”的方式

### A) 直接改 YAML（推荐：可控、可复现）

你可以复制/新建多个 yaml，例如：
- `my_cn_scene_main.yaml`（主集：更干净、低弯曲、低噪声）
- `my_cn_scene_curve_boost.yaml`（子集：高弯曲、更强增强）

然后用 wrapper 指定 `--base-config <yaml>`。

### B) 用 wrapper 的参数覆盖（仅支持少数项）

当前 wrapper 只会覆盖：
- `--curve-ratio`：写入 `curve.fraction`（并根据 >0 自动开 `curve.enable`）

也就是说：`curve.min/max/period`、`seamless_clone`、`noise/blur` 等都需要你改 yaml（或你让我把 wrapper 扩展成支持更多参数）。

---

## 2. wrapper 常用参数（最重要的几个）

生成命令骨架：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/recog_synth/run_synth_chinese_ocr.py \
  --num-images 500 \
  --out-root /root/lanyun-tmp/mmocr/data/synth_rec_ch \
  --tag debug500 \
  --base-config /root/lanyun-tmp/mmocr/3rdparty/synth_chinese_ocr/configs/my_cn_scene.yaml \
  --curve-ratio 0.7 \
  --chars-file /root/lanyun-tmp/mmocr/data/charset/charset_rec_cn_en.txt \
  --fonts-list /root/lanyun-tmp/mmocr/data/synth_assets/fonts_list/chn.txt \
  --corpus-mode list \
  --corpus-dir /root/lanyun-tmp/mmocr/data/synth_assets/list_corpus/full_main_fontok \
  --bg-dir /root/lanyun-tmp/mmocr/data/synth_assets/bg_proc \
  --strict \
  --clean \
  --num-processes 14'
```

说明：
- `--corpus-mode list`：会递归读取 `--corpus-dir` 下所有 `.txt`。因此建议传 `full_main_fontok/` 或 `full_curve_mix_fontok/` 这种“目录里只有目标 txt”的目录。
- `--strict`：启用“字体缺字重试”，因此必须使用 `*.fontok.txt`（否则可能无限重试/速度极慢）。
- `--img-width 0`（wrapper 默认）：输出可变宽；若你训练管线要求固定宽，可在 wrapper 里加参数或直接修改 wrapper 默认值。
- `--num-processes`：CPU 并行度。建议设为 CPU 核数（如 14），或留空让其自动使用全部核。

---

## 3. YAML 的“概率模型”怎么理解

`synth_chinese_ocr` 里大多数增强都是：
- `enable: true/false`
- `fraction: 0~1`

当 `enable=true` 且抽样命中 `fraction` 时，才会应用该增强（见 `3rdparty/synth_chinese_ocr/libs/utils.py:apply`）。

因此：
- 想“更多发生” ⇒ 提高 `fraction`
- 想“更强形变/更强噪声” ⇒ 看具体模块的强度参数（例如 `curve.min/max/period`）

---

## 4. 关键字段怎么改（面向自然场景 + 弯曲文本）

下面字段都在 `my_cn_scene.yaml` 中：

### 4.1 背景相关（决定“自然感”）

- `img_bg.enable / img_bg.fraction`
  - 更真实：`fraction` 接近 1（几乎全用 bg 图）
  - 更干净：降低它（更多随机纯色背景）
- `seamless_clone.enable / seamless_clone.fraction`
  - 更真实：适度提高（但会变慢，也可能偶发“融合怪”）
  - 更快更稳定：降低或关闭

### 4.2 弯曲相关（你的重点）

字段：`curve.enable / curve.fraction / curve.period / curve.min / curve.max`

直觉：
- `curve.fraction`：有多少比例样本会弯
- `curve.min/max`：弯曲幅值范围，越大越弯
- `curve.period`：波形周期（单位 degree），越小越“波浪/频繁”，越大越“缓弯/近似单弧”

建议：
- 主集：`curve.fraction 0.10~0.25`，`max 6~10`，`period 1200~1800`
- 强化集：`curve.fraction 0.70~0.85`，`max 10~16`，`period 600~1000`

### 4.3 模糊/降采样/噪声（你看到“看不清”的主要来源）

- `blur.fraction`：越大越容易模糊
- `prydown.fraction / prydown.max_scale`：模拟小图放大（越大越糊）
- `noise.fraction` + 子噪声分配：越大越脏

若你觉得“人都看不清”的样本比例偏高：
- 先把 `blur.fraction` 降到 `0.05~0.10`
- `prydown.fraction` 降到 `0.10~0.20`，`max_scale` 先用 `1.3~1.8`
- `noise.fraction` 降到 `0.20~0.30`

### 4.4 低对比度（浅色字 + 背景接近）

主要由 `font_color` 和 `seamless_clone` 引起：
- 想减少低对比：降低 `font_color.white.fraction`，提高 `font_color.black/dark_gray` 占比
- 或提高 `text_border.fraction`（描边更容易“读得清”）
- 或降低 `seamless_clone.fraction`（减少融合导致的“吃字”）

### 4.5 裁切误差（模拟检测框不准）

字段：`crop.enable / crop.fraction / crop.top/bottom`
- 轻度建议：`fraction 0.10~0.25`，`top/bottom 1~4`
- 过强会切断笔画，变成标签噪声

---

## 5. 最推荐的用法：主集 + 弯曲强化子集

用两个 tag 分开生成，然后训练时混合或拼接 label：

- 主集（更干净）：`--curve-ratio 0.15` + `corpus_dir=full_main_fontok`
- 子集（弯曲强化）：`--curve-ratio 0.75` + `corpus_dir=full_curve_mix_fontok`

---

## 6. 常见坑

- `corpus_mode=list` 会把 `corpus_dir` 下所有 `.txt` 都读进来：目录里别放中间产物。
- 开了 `--strict` 但语料里有字符所有字体都不支持：会疯狂重试/变慢。当前我们已用 `*.fontok.txt` 过滤过（仍建议保留该流程）。
- `GPU 生图`：当前仓库的 `--gpu` 依赖 OpenCV CUDA + Cython 扩展 `libs/gpu/GpuWrapper`，你的环境里该扩展未编译且 OpenCV 无 CUDA，因此现阶段无法用 GPU 生图（CPU 多进程是推荐方案）。

