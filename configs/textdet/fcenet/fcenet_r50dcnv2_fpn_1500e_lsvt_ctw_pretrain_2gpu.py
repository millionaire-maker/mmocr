_base_ = ['fcenet_r50dcnv2_fpn_1500e_lsvt_ctw_pretrain.py']

# 注意：MMEngine/MMOCR 中的 dataloader.batch_size 是“每张 GPU 的 batch size”。
# 你原来单卡配置里 train_dataloader.batch_size=32（总 batch=32）。
# 现在单机 2 卡训练，为了保持“总 batch size”不变（=32），把每卡 batch 调成 16。
work_dir = 'work_dirs/fcenet_r50dcnv2_pretrain_lsvt_ctw_2gpu'

train_dataloader = dict(batch_size=16)

