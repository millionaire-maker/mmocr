import torch.nn as nn
from mmdet.models.backbones import ResNet

from mmocr.models.common.attentions import CBAMBlock, SEBlock
from mmocr.registry import MODELS


def _stage_channels(depth: int):
    if depth in (18, 34):
        return (64, 128, 256, 512)
    return (256, 512, 1024, 2048)


@MODELS.register_module()
class ResNetWithAttention(ResNet):
    """在 ResNet 输出特征后插入 SE / CBAM 等注意力模块。"""

    def __init__(self,
                 attention_type: str = 'se',
                 attn_stages=(1, 2, 3),
                 reduction: int = 16,
                 spatial_kernel: int = 7,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention_type = attention_type
        self.attn_stages = attn_stages
        channels = _stage_channels(kwargs.get('depth', 50))
        self.attentions = nn.ModuleList()
        for idx, ch in enumerate(channels):
            if idx in attn_stages and attention_type:
                self.attentions.append(
                    self._build_attention(ch, attention_type, reduction,
                                          spatial_kernel))
            else:
                self.attentions.append(None)

    @staticmethod
    def _build_attention(channels: int, attention_type: str, reduction: int,
                         spatial_kernel: int):
        if attention_type.lower() == 'se':
            return SEBlock(channels, reduction=reduction)
        if attention_type.lower() == 'cbam':
            return CBAMBlock(
                channels, reduction=reduction, spatial_kernel=spatial_kernel)
        return None

    def forward(self, x):
        outs = list(super().forward(x))
        for i, attn in enumerate(self.attentions):
            if attn is not None and i < len(outs):
                outs[i] = attn(outs[i])
        return tuple(outs)
