_base_ = ['textdet.py']

task = 'textspotting'

_base_.train_preparer.obtainer = None
_base_.test_preparer.obtainer = None

# 指向用户提供的新原始数据路径
_base_.train_preparer.gatherer.img_dir = 'train/text_image'
_base_.train_preparer.gatherer.ann_dir = '原始数据/ctw1500_train_labels'
_base_.test_preparer.gatherer.img_dir = 'test/text_image'
_base_.test_preparer.gatherer.ann_dir = '原始数据/evaluation_det_e2e_offline/datasets/evaluation/gt_ctw1500'

_base_.train_preparer.packer.type = 'TextSpottingPacker'
_base_.test_preparer.packer.type = 'TextSpottingPacker'

# 端到端不强制下载lexicons
_base_.test_preparer.obtainer = None


config_generator = dict(type='TextSpottingConfigGenerator')
