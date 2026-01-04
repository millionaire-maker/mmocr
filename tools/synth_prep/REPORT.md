# synth_chinese_ocr（3rdparty/synth_chinese_ocr）合成预训练数据：准备 + debug500 验收 + 全量生成方案

仓库根目录：`/root/lanyun-tmp/mmocr`  
合成器：`/root/lanyun-tmp/mmocr/3rdparty/synth_chinese_ocr`  
Wrapper：`/root/lanyun-tmp/mmocr/tools/recog_synth/run_synth_chinese_ocr.py`  
目标：**弯曲（curved text）优先**，不做竖排排版。

> 重要约定（口径统一）
> - 统一字符集：`/root/lanyun-tmp/mmocr/data/charset/charset_rec_cn_en.txt`（每行取第 1 个字符；共 5883；**不包含空格**；英文为小写）
> - 全部文本处理口径：`NFKC + lowercase`，且因 charset 不含空格，默认**移除所有空白字符**
> - 所有 Python 命令均在 conda 环境 `openmmlab` 执行（下文命令已写好）

---

## 0) 自检与环境

- conda 环境：`openmmlab`
- Python：`3.8.20`
- 依赖可用：`lmdb` / `PIL` 可正常 import

检查命令（原样可运行）：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && python -V'
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && python -c "import lmdb, PIL; print(\"lmdb/PIL ok\")"'
```

> 若 conda 激活失败，可直接用：`/root/miniconda/envs/openmmlab/bin/python` 运行本文所有脚本。

---

## 1) 背景图预处理（适配竖长图）

输入：`/root/lanyun-tmp/mmocr/data/synth_assets/bg`（共 9 张）  
输出：`/root/lanyun-tmp/mmocr/data/synth_assets/bg_proc`（共 9 张，**统一 RGB JPG**，最长边缩放到 1280，保持比例不裁剪）  
明细报告：`/root/lanyun-tmp/mmocr/data/synth_assets/bg_proc_report.txt`

可复用脚本：`/root/lanyun-tmp/mmocr/tools/synth_prep/preprocess_bg.py`

执行命令：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/preprocess_bg.py --src data/synth_assets/bg --dst data/synth_assets/bg_proc --max-side 1280 --report data/synth_assets/bg_proc_report.txt'
```

---

## 2) 语料格式统一（base / hard_curve）

输入：
- `data/synth_assets/list_corpus/base.txt`
- `data/synth_assets/list_corpus/hard_curve.txt`

输出（不覆盖原文件；已自动备份 `*.bak`）：
- `data/synth_assets/list_corpus/base.cleaned.txt`
- `data/synth_assets/list_corpus/hard_curve.cleaned.txt`

可复用脚本：`tools/synth_prep/clean_corpus.py`

清洗动作：
- 去 BOM、strip 行首尾空白、移除空行
- 去“序号前缀”（最大鲁棒，命中规则会统计）
- `NFKC + lowercase`
- 因 charset **不含空格**：移除所有空白字符
- 去重（完全相同行）

“序号前缀”判定规则（脚本内 `PREFIX_RULES`，按顺序匹配，命中则剥离）：
- `digits_tab`：`001\t...` / `001\t\t...`
- `bracket_digits`：`(23) ...` / `（23）...` / `[23] ...` / `【23】...`
- `digits_rparen`：`23) ...` / `23）...`
- `digits_punct`：`23. ...` / `23、...` / `23: ...` / `23：...` / `23-...` / `23—...`
- `digits_rbracket`：`23] ...` / `23】...`
- `digits_space_short`：`23 ...`（仅剥离 1~3 位数字+空格，避免误删类似 “2024 …”）

统计结果（见 `tools/synth_prep/*.stats.json`）：

- base：
  - 原始行数：2826
  - 清洗后（未去重）：2438
  - 去重后：2398
  - 规则命中：`bracket_digits=138`，`digits_punct=2181`
- hard_curve：
  - 原始行数：354
  - 清洗后（未去重）：354
  - 去重后：354

执行命令：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/clean_corpus.py --in data/synth_assets/list_corpus/base.txt --out data/synth_assets/list_corpus/base.cleaned.txt --charset data/charset/charset_rec_cn_en.txt --normalize nfkc_lower --stats tools/synth_prep/base.cleaned.stats.json'
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/clean_corpus.py --in data/synth_assets/list_corpus/hard_curve.txt --out data/synth_assets/list_corpus/hard_curve.cleaned.txt --charset data/charset/charset_rec_cn_en.txt --normalize nfkc_lower --stats tools/synth_prep/hard_curve.cleaned.stats.json'
```

---

## 3) 从 Fudan scene_train LMDB 抽取 label 并融合到 base

LMDB：`/root/lanyun-tmp/mmocr/data/fudan/scene/scene_train`  
实现：直接读取 LMDB（兼容 `num-samples` / `label-%09d`），并做 `NFKC + lowercase` +（因 charset 不含空格）移除空白 + charset 过滤。

输出：
- 抽取的 labels（去重后）：`tools/synth_prep/fudan_scene_train.labels.txt`
- 融合结果：`data/synth_assets/list_corpus/base.plus_fudan.txt`
- 统计：`tools/synth_prep/base.plus_fudan.stats.json`

关键统计（来自 `base.plus_fudan.stats.json`）：
- LMDB 样本数：509164（`num-samples`）
- label key：`label-%09d`（start index=1）
- 读到 label：509164
- charset 过滤掉：11
- 抽取后去重：230180
- base.cleaned 输入：2398（其中 4 条在 charset 过滤时被剔除）
- 合并后总条数：232187（新增 229793，重复跳过 387）

执行命令：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/extract_fudan_labels.py --lmdb-dir data/fudan/scene/scene_train --out tools/synth_prep/fudan_scene_train.labels.txt --charset data/charset/charset_rec_cn_en.txt --normalize nfkc_lower --merge-base data/synth_assets/list_corpus/base.cleaned.txt --out-merged data/synth_assets/list_corpus/base.plus_fudan.txt --stats tools/synth_prep/base.plus_fudan.stats.json'
```

---

## 4) 字符集覆盖检查：删除含字典外字符的行

对以下文件逐行检查（`NFKC + lowercase` 后再检查；并移除空白后检查）：
- `base.cleaned.txt`
- `hard_curve.cleaned.txt`
- `base.plus_fudan.txt`

输出：
- `base.cleaned.filtered.txt`（2394/2398）
- `hard_curve.cleaned.filtered.txt`（352/354）
- `base.plus_fudan.filtered.txt`（232187/232187）

Top-OOV（最多 50，来自 `tools/synth_prep/*filtered.stats.json`）：
- base.cleaned.filtered：`–(2)`、`珐(1)`、`缀(1)`、`讫(1)`
- hard_curve.cleaned.filtered：`鹄(2)`

可复用脚本：`tools/synth_prep/filter_by_charset.py`

执行命令：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/filter_by_charset.py --in data/synth_assets/list_corpus/base.cleaned.txt --out data/synth_assets/list_corpus/base.cleaned.filtered.txt --charset data/charset/charset_rec_cn_en.txt --normalize nfkc_lower --stats tools/synth_prep/base.cleaned.filtered.stats.json'
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/filter_by_charset.py --in data/synth_assets/list_corpus/hard_curve.cleaned.txt --out data/synth_assets/list_corpus/hard_curve.cleaned.filtered.txt --charset data/charset/charset_rec_cn_en.txt --normalize nfkc_lower --stats tools/synth_prep/hard_curve.cleaned.filtered.stats.json'
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/filter_by_charset.py --in data/synth_assets/list_corpus/base.plus_fudan.txt --out data/synth_assets/list_corpus/base.plus_fudan.filtered.txt --charset data/charset/charset_rec_cn_en.txt --normalize nfkc_lower --stats tools/synth_prep/base.plus_fudan.filtered.stats.json'
```

---

## 4b) 字体覆盖过滤（避免 `--strict` 因缺字无限重试）

现状：`fonts_list/chn.txt` 的 9 个字体**联合**仍无法渲染 charset 中 21 个字符：

```
čřšžƨǝɔɛʃʌḍḥṭℰ∙▬❋❤𠝹𣇉𦠿
```

在 `base.plus_fudan.filtered.txt` 中实际命中这些字符的行数：48 行（例：`succeʃʃ`、`1997∙华夫冰淇淋∙`、`麻辣小𣇉肝...`）。  
为避免生成阶段死循环，已剔除并输出 **font-ok** 版本：

- `data/synth_assets/list_corpus/base.plus_fudan.filtered.fontok.txt`（232139/232187）
- `data/synth_assets/list_corpus/hard_curve.cleaned.filtered.fontok.txt`（352/352）

可复用脚本：`tools/synth_prep/filter_by_fonts.py`

执行命令：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/filter_by_fonts.py --corpus-in data/synth_assets/list_corpus/base.plus_fudan.filtered.txt --corpus-out data/synth_assets/list_corpus/base.plus_fudan.filtered.fontok.txt --fonts-list data/synth_assets/fonts_list/chn.txt --normalize nfkc_lower --stats tools/synth_prep/base.plus_fudan.filtered.fontok.stats.json'
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/filter_by_fonts.py --corpus-in data/synth_assets/list_corpus/hard_curve.cleaned.filtered.txt --corpus-out data/synth_assets/list_corpus/hard_curve.cleaned.filtered.fontok.txt --fonts-list data/synth_assets/fonts_list/chn.txt --normalize nfkc_lower --stats tools/synth_prep/hard_curve.cleaned.filtered.fontok.stats.json'
```

---

## 5) 字体准备与 fonts_list 生成（店招/美术字）

输出目录：
- 字体：`/root/lanyun-tmp/mmocr/data/synth_assets/fonts/chn/`
- 字体列表（绝对路径，一行一个）：`/root/lanyun-tmp/mmocr/data/synth_assets/fonts_list/chn.txt`

实现策略：
- A) 联网可用时：自动从 Google Fonts 下载（**SIL OFL 1.1**，可商用开源），覆盖：
  - 常规：Noto Sans/Serif SC（可替代思源黑/思源宋风格）
  - 显示/店招/美术：ZCOOL 系列、MaShanZheng、LiuJianMaoCao、LongCang、ZhiMangXing 等
- B) 若下载失败：自动扫描系统字体目录并导入（脚本已实现兜底；本次未触发）

可用性检查（写入 `tools/synth_prep/fonts_report.json`）：
- PIL/FreeType 可加载
- 从 charset 中随机抽取 1000 字符做覆盖检查，阈值：`missing_ratio_thr=0.25`（缺字 >25% 的字体剔除）

本次结果：
- 通过检查字体：9/9
- 详细来源/许可/缺字比例：`tools/synth_prep/fonts_report.json`

可复用脚本：`tools/synth_prep/prepare_fonts.py`

执行命令：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/prepare_fonts.py --charset data/charset/charset_rec_cn_en.txt --dst-dir data/synth_assets/fonts/chn --list-out data/synth_assets/fonts_list/chn.txt --report tools/synth_prep/fonts_report.json --mode auto --sample-size 1000 --missing-ratio-thr 0.25 --seed 0'
```

若你希望后续补充更“楷/圆”等风格（手工下载后放入 `fonts/chn`，再重跑脚本生成列表）：
- `LXGW WenKai / 霞鹜文楷`（OFL）
- `Smiley Sans / 得意黑`（OFL）
- `Source Han Sans/Serif SC / 思源黑体/思源宋体`（OFL）

---

## 6) 合成配置 my_cn_scene.yaml（弯曲优先 + 自然背景优先）

配置文件：`/root/lanyun-tmp/mmocr/3rdparty/synth_chinese_ocr/configs/my_cn_scene.yaml`

关键参数（字段名=值，默认“中档”）：
- 背景：
  - `img_bg.enable=true`
  - `img_bg.fraction=0.98`（几乎全用自然背景）
  - `seamless_clone.enable=true`
  - `seamless_clone.fraction=0.18`（少量使用更真实融合；过高会显著变慢）
- 弯曲（Sin remap）：
  - `curve.enable=true`
  - `curve.fraction=0.55`（默认中档）
  - `curve.period=1000`（越小越“波浪/频繁”，越大越“缓弯/近似单弧”）
  - `curve.min=2`、`curve.max=10`（幅值范围；越大弯曲越强，也越容易“合成味”）
- 透视/旋转：
  - `perspective_transform.max_x=18`
  - `perspective_transform.max_y=18`
  - `perspective_transform.max_z=7`（适度旋转增强鲁棒性；过大会出现不真实倾斜）
- 模糊/降采样/噪声：
  - `blur.fraction=0.12`（少量 blur）
  - `prydown.fraction=0.25`，`prydown.max_scale=2.0`（模拟小图放大）
  - `noise.fraction=0.35`（适度加噪）
- 裁切误差（模拟检测裁框）：
  - `crop.enable=true`
  - `crop.fraction=0.25`
  - `crop.top/bottom: 1~4`（轻度切边；过大容易切断笔画）

轻/中/强三档建议（你可以直接改 `my_cn_scene.yaml`，或用 wrapper 的 `--curve-ratio` 覆盖 `curve.fraction`）：

- 轻（更快、更干净，适合作主集）：
  - `curve.fraction=0.10~0.25`
  - `curve.max=6~8`，`curve.period=1200~1600`
  - `seamless_clone.fraction=0.10~0.15`
  - `noise.fraction=0.25~0.30`，`blur.fraction=0.08~0.10`
- 中（当前默认，兼顾真实与强度）：
  - `curve.fraction=0.45~0.60`
  - `curve.max=10`，`curve.period≈1000`
  - `seamless_clone.fraction≈0.18`
- 强（更偏弯曲强化；速度更慢，合成风险更高）：
  - `curve.fraction=0.70~0.85`
  - `curve.max=12~16`，`curve.period=600~900`
  - `seamless_clone.fraction=0.20~0.30`
  - `noise.fraction=0.35~0.45`，`prydown.fraction=0.30~0.40`

增强对“弯曲文本识别预训练”的意义与风险（简述）：
- 意义：提前让识别模型见过“非直线排布 + 轻透视 + 真实背景 + 轻模糊/噪声/压缩”的组合，提高泛化。
- 风险：`curve.max` 太大或 `period` 太小会出现强烈波浪，容易偏离真实分布；`seamless_clone` 过高会拖慢且偶发不自然边缘；噪声/模糊过强可能引入负迁移。

---

## 7) 生成 500 张验收集（debug500）

目标输出：
- 图片目录：`/root/lanyun-tmp/mmocr/data/synth_rec_ch/debug500/`（500 张 jpg）
- labels：`/root/lanyun-tmp/mmocr/data/synth_rec_ch/debug500.txt`（500 行，`rel_path<TAB>label`）

### 7.1 生成 debug500 的加权语料（40% 来自 hard_curve）

输入：
- 主语料：`data/synth_assets/list_corpus/base.plus_fudan.filtered.txt`
- 弯曲难例：`data/synth_assets/list_corpus/hard_curve.cleaned.filtered.txt`

输出：
- `data/synth_assets/list_corpus/debug500/merged_debug500.txt`（固定 500 行；hard=200/base=300；seed=0）

命令：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/build_weighted_corpus.py --base data/synth_assets/list_corpus/base.plus_fudan.filtered.txt --hard data/synth_assets/list_corpus/hard_curve.cleaned.filtered.txt --out data/synth_assets/list_corpus/debug500/merged_debug500.txt --total 500 --hard-ratio 0.4 --seed 0 --min-len 2 --stats tools/synth_prep/merged_debug500.stats.json'
```

### 7.2 生成命令（原样可复制）

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/recog_synth/run_synth_chinese_ocr.py --num-images 500 --out-root /root/lanyun-tmp/mmocr/data/synth_rec_ch --tag debug500 --curve-ratio 0.7 --chars-file /root/lanyun-tmp/mmocr/data/charset/charset_rec_cn_en.txt --fonts-list /root/lanyun-tmp/mmocr/data/synth_assets/fonts_list/chn.txt --corpus-dir /root/lanyun-tmp/mmocr/data/synth_assets/list_corpus/debug500 --corpus-mode list --bg-dir /root/lanyun-tmp/mmocr/data/synth_assets/bg_proc --base-config /root/lanyun-tmp/mmocr/3rdparty/synth_chinese_ocr/configs/my_cn_scene.yaml --strict --clean --num-processes 8'
```

### 7.3 自动验收统计

- 图片数量：500（`find data/synth_rec_ch/debug500 -name "*.jpg" | wc -l`）
- labels 条目：500（`wc -l data/synth_rec_ch/debug500.txt`）

随机抽查 20 条（seed=0）：

```
debug500/00000453.jpg	revolution
debug500/00000159.jpg	瑞康大药房有限公司
debug500/00000346.jpg	到店自取请核对订单
debug500/00000455.jpg	合作洽谈请联系招商主管
debug500/00000258.jpg	人可以住的别墅
debug500/00000003.jpg	mediationroom
debug500/00000132.jpg	美地
debug500/00000476.jpg	320
debug500/00000281.jpg	免费加饭一次
debug500/00000264.jpg	四季鲜果便利店
debug500/00000143.jpg	400-654-0459
debug500/00000413.jpg	续航六十公里左右
debug500/00000347.jpg	滇味过桥米线馆
debug500/00000421.jpg	电饭锅等
debug500/00000124.jpg	acrosier
debug500/00000431.jpg	锦璟全屋定制有限公司
debug500/00000221.jpg	武汉职业技术学院招生与就业指导处
debug500/00000187.jpg	童装童鞋折扣店
debug500/00000291.jpg	锦澜臻选生活馆
debug500/00000442.jpg	窝头
```

---

## 8) 验收通过后的“全量生成”方案（主集 + 弯曲强化子集）

### 8.1 输出目录结构（wrapper 固定行为）

若 `--out-root data/synth_rec_ch --tag <TAG>`，则输出：
- 图片：`data/synth_rec_ch/<TAG>/*.jpg`
- 标签：`data/synth_rec_ch/<TAG>.txt`（每行：`<TAG>/<img>.jpg<TAB><label>`）

### 8.2 推荐拆分（可按算力/存储调整）

- 主集（低弯曲比例、跑得快）：`num_images=2,000,000`，`curve_ratio=0.15`
- 弯曲强化子集（高弯曲比例）：`num_images=200,000~500,000`，`curve_ratio=0.75`

> 说明：当前 hard_curve 只有 352 行，若弯曲强化集做得太大，会出现 hard 样本重复次数很高；建议后续持续扩充 `hard_curve` 语料再放大规模。

### 8.3 已准备好的 corpus_dir（直接用）

为避免 `list` 模式下加载到无关 txt，本次已创建专用目录：
- 主集语料目录：`data/synth_assets/list_corpus/full_main_fontok/`
  - 仅包含 `base.plus_fudan.filtered.fontok.txt` 的符号链接
- 弯曲强化语料目录：`data/synth_assets/list_corpus/full_curve_mix_fontok/`
  - 仅包含 `mixed_curve_mix.txt`（100000 行，hard_ratio=0.4；统计见 `tools/synth_prep/full_curve_mix.stats.json`）

如需重新生成弯曲强化混合语料（调整 hard_ratio/总行数）：

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/synth_prep/build_weighted_corpus.py --base data/synth_assets/list_corpus/base.plus_fudan.filtered.fontok.txt --hard data/synth_assets/list_corpus/hard_curve.cleaned.filtered.fontok.txt --out data/synth_assets/list_corpus/full_curve_mix_fontok/mixed_curve_mix.txt --total 100000 --hard-ratio 0.4 --seed 1 --min-len 2 --stats tools/synth_prep/full_curve_mix.stats.json'
```

### 8.4 两条“可直接执行”的全量生成命令（原样可复制）

命令 1：主集（低弯曲比例）

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/recog_synth/run_synth_chinese_ocr.py --num-images 2000000 --out-root /root/lanyun-tmp/mmocr/data/synth_rec_ch --tag cn_scene_main --curve-ratio 0.15 --chars-file /root/lanyun-tmp/mmocr/data/charset/charset_rec_cn_en.txt --fonts-list /root/lanyun-tmp/mmocr/data/synth_assets/fonts_list/chn.txt --corpus-dir /root/lanyun-tmp/mmocr/data/synth_assets/list_corpus/full_main_fontok --corpus-mode list --bg-dir /root/lanyun-tmp/mmocr/data/synth_assets/bg_proc --base-config /root/lanyun-tmp/mmocr/3rdparty/synth_chinese_ocr/configs/my_cn_scene.yaml --strict --clean --num-processes 8'
```

命令 2：弯曲强化子集（高弯曲比例）

```bash
/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && /root/miniconda/envs/openmmlab/bin/python tools/recog_synth/run_synth_chinese_ocr.py --num-images 300000 --out-root /root/lanyun-tmp/mmocr/data/synth_rec_ch --tag cn_scene_curve_boost --curve-ratio 0.75 --chars-file /root/lanyun-tmp/mmocr/data/charset/charset_rec_cn_en.txt --fonts-list /root/lanyun-tmp/mmocr/data/synth_assets/fonts_list/chn.txt --corpus-dir /root/lanyun-tmp/mmocr/data/synth_assets/list_corpus/full_curve_mix_fontok --corpus-mode list --bg-dir /root/lanyun-tmp/mmocr/data/synth_assets/bg_proc --base-config /root/lanyun-tmp/mmocr/3rdparty/synth_chinese_ocr/configs/my_cn_scene.yaml --strict --clean --num-processes 8'
```

合并使用方式（示例）：训练时同时读两份 label txt；或先合并：

```bash
cat data/synth_rec_ch/cn_scene_main.txt data/synth_rec_ch/cn_scene_curve_boost.txt > data/synth_rec_ch/cn_scene_all.txt
```

