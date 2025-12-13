# Fudan Scene LMDB 兼容性说明

## MMOCR 期望的 LMDB 结构
- 参考 `mmocr/datasets/recog_lmdb_dataset.py`，`RecogLMDBDataset` 读取的 LMDB 需要包含键：`num-samples`、`image-XXXXXXXXX`、`label-XXXXXXXXX`（X 为 9 位编号）。
- `num-samples` 存总样本数，`image-*` 保存图像二进制，`label-*` 保存 UTF-8 文本。
- 数据管线需配合 `LoadImageFromNDArray` / `LoadOCRAnnotations` 等文字识别 pipeline。

## Fudan Scene 实际结构
- 目录：`/home/yzy/mmocr/data/fudan/scene/scene_{train,val,test}`，内部由 LMDB 默认的 `data.mdb`、`lock.mdb` 组成。
- `data/fudan/tools/lmdbReader.py` 与 `tools/inspect_fudan_lmdb.py` 均验证键命名采用 `image-%09d` / `label-%09d`，并存在 `num-samples`。
- 标签均为 UTF-8，可混合中文、英文、符号，样例与尺寸见 `reports/fudan_scene_lmdb_samples.md`。

## 兼容结论
- Fudan Scene LMDB 与 `RecogLMDBDataset` 要求完全一致，可直接作为 MMOCR 中文文本识别输入，无需额外转换。
- 推荐配置：`type='RecogLMDBDataset'`、`ann_file='/home/yzy/mmocr/data/fudan/scene/scene_train'`（val/test 同理），pipeline 以 `LoadImageFromNDArray` 开头，后续沿用 ABINet 默认的增强与打包步骤。
- 因此在总结中会直接给出 `RecogLMDBDataset + ABINet pipeline` 的使用方式。
