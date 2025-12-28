_base_ = ['fcenet_r50dcnv2_fpn_1500e_lsvt_ctw_pretrain.py']

# 2 卡训练 + 提速版：
# - MMOCR/MMEngine 中 dataloader.batch_size 是“每张 GPU 的 batch size”
# - 设为 32 时，2 卡总 batch = 32 * 2 = 64
work_dir = 'work_dirs/fcenet_r50dcnv2_pretrain_lsvt_ctw_2gpu_bs64'

train_dataloader = dict(batch_size=32)

