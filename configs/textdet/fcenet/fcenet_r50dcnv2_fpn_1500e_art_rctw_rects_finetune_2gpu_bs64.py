_base_ = ['fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py']

# 2 卡训练 + 总 batch size=64：
# - MMOCR/MMEngine 中 dataloader.batch_size 是“每张 GPU 的 batch size”
# - 设为 32 时，2 卡总 batch = 32 * 2 = 64
work_dir = 'work_dirs/fcenet_r50dcnv2_fpn_finetune_art_rctw_rects_2gpu_bs64'

# 使用 LSVT+CTW 预训练得到的最优权重作为初始化（finetune 只加载权重，不恢复 optimizer/scheduler）
load_from = 'work_dirs/fcenet_r50dcnv2_fpn_pretrain_lsvt_ctw/best_icdar_hmean_epoch_50.pth'

train_dataloader = dict(batch_size=32)

