_base_ = ['dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py']

# 直接在 ART+RCTW+RECTS 上训练（不进行 LSVT+CTW 预训练阶段）：
# - 仅保留 backbone 的 ImageNet 初始化（见 base 配置中的 torchvision://resnet50）
# - 不加载任何 textdet 预训练权重
work_dir = 'work_dirs/dbnetpp_r50_direct_finetune_art_rctw_rects'
load_from = None

