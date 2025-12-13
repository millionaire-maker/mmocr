_base_ = ['textdet.py']

task = 'textrecog'

_base_.train_preparer.obtainer = None
_base_.test_preparer.obtainer = None

_base_.train_preparer.gatherer.img_dir = 'train/text_image'
_base_.train_preparer.gatherer.ann_dir = 'ctw1500_train_labels'
_base_.test_preparer.gatherer.img_dir = 'test/text_image'
_base_.test_preparer.gatherer.ann_dir = 'gt_ctw1500'

_base_.train_preparer.packer.type = 'TextRecogCropPacker'
_base_.test_preparer.packer.type = 'TextRecogCropPacker'

config_generator = dict(type='TextRecogConfigGenerator')
