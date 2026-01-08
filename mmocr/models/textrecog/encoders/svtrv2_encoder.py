# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_init

from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample


class ConvBNLayer(BaseModule):
    """Conv2d + BatchNorm2d + GELU.

    This block is migrated from OpenOCR SVTRv2 implementation:
    `openrec/modeling/encoders/svtrv2_lnconv_two33.py::ConvBNLayer`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: int = 0,
                 bias: bool = False,
                 groups: int = 1,
                 act: type[nn.Module] = nn.GELU,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Mlp(BaseModule):
    """MLP block used by SVTRv2 global mixing."""

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: type[nn.Module] = nn.GELU,
                 drop: float = 0.0,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(BaseModule):
    """Multi-head self-attention used in SVTRv2 global mixing."""

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads,
                                  self.dim // self.num_heads).permute(
                                      2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalBlock(BaseModule):
    """Transformer block for SVTRv2 global mixing."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 act_layer: type[nn.Module] = nn.GELU,
                 norm_layer: type[nn.Module] = nn.LayerNorm,
                 eps: float = 1e-6,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FlattenBlockRe2D(GlobalBlock):
    """Global block that takes 2D features (B, C, H, W) and returns 2D."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).reshape(bsz, channels, height, width)
        return x


class ConvBlock(BaseModule):
    """Conv-based local mixing block (Conv2 in SVTRv2).

    It replaces the local mixer with several depthwise (grouped) 2D convs,
    without intermediate norm/act, matching OpenOCR SVTRv2 LNConvTwo33.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.0,
                 drop_path: float = 0.0,
                 act_layer: type[nn.Module] = nn.GELU,
                 norm_layer: type[nn.Module] = nn.LayerNorm,
                 eps: float = 1e-6,
                 num_conv: int = 2,
                 kernel_size: int = 3,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = nn.Sequential(*[
            nn.Conv2d(dim,
                      dim,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      groups=num_heads) for _ in range(num_conv)
        ])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels, height, width = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose(1, 2).reshape(-1, channels, height, width)
        return x


class FlattenTranspose(BaseModule):
    """Flatten spatial dims and transpose to (B, N, C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)


class SubSample2D(BaseModule):
    """Downsample module for 2D feature maps."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Sequence[int] = (2, 1),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor,
                sz: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
        x = self.conv(x)
        channels, height, width = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, channels, height, width)
        return x, [height, width]


class SubSample1D(BaseModule):
    """Downsample module for token sequences."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Sequence[int] = (2, 1),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor,
                sz: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
        channels = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, channels, sz[0], sz[1])
        x = self.conv(x)
        channels, height, width = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [height, width]


class IdentitySize(BaseModule):
    """Identity with size passthrough."""

    def forward(self, x: torch.Tensor,
                sz: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
        return x, list(sz)


class SVTRv2Stage(BaseModule):
    """SVTRv2 stage consisting of mixed Conv/Global blocks and downsample."""

    def __init__(self,
                 dim: int,
                 out_dim: int,
                 depth: int,
                 mixer: Sequence[str],
                 kernel_sizes: Sequence[int],
                 sub_k: Sequence[int],
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path: Sequence[float] = (0.0, ),
                 norm_layer: type[nn.Module] = nn.LayerNorm,
                 act: type[nn.Module] = nn.GELU,
                 eps: float = 1e-6,
                 num_conv: Sequence[int] = (2, ),
                 downsample: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.dim = dim

        blocks: List[nn.Module] = []
        for i in range(depth):
            mix = mixer[i]
            if mix == 'Conv':
                blocks.append(
                    ConvBlock(
                        dim=dim,
                        kernel_size=kernel_sizes[i],
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        act_layer=act,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        eps=eps,
                        num_conv=num_conv[i]))
            else:
                if mix == 'Global':
                    block_cls = GlobalBlock
                elif mix == 'FGlobal':
                    blocks.append(FlattenTranspose())
                    block_cls = GlobalBlock
                elif mix == 'FGlobalRe2D':
                    block_cls = FlattenBlockRe2D
                else:
                    raise ValueError(
                        "SVTRv2 mixer must be one of ['Conv','Global','FGlobal','FGlobalRe2D'], "
                        f'but got {mix!r}')
                blocks.append(
                    block_cls(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        act_layer=act,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        eps=eps,
                    ))
        self.blocks = nn.ModuleList(blocks)

        if downsample:
            if mixer[-1] in ('Conv', 'FGlobalRe2D'):
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x: torch.Tensor,
                sz: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
        for blk in self.blocks:
            x = blk(x)
        x, sz = self.downsample(x, sz)
        return x, sz


class ADDPosEmbed(BaseModule):
    """Add learnable 2D positional embedding with dynamic slicing."""

    def __init__(self,
                 feat_max_size: Sequence[int],
                 embed_dim: int,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        pos_embed = torch.zeros(
            [1, feat_max_size[0] * feat_max_size[1], embed_dim],
            dtype=torch.float32)
        trunc_normal_init(pos_embed, mean=0, std=0.02)
        self.pos_embed = nn.Parameter(
            pos_embed.transpose(1, 2).reshape(1, embed_dim, feat_max_size[0],
                                              feat_max_size[1]),
            requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        return x + self.pos_embed[:, :, :height, :width]


class POPatchEmbed(BaseModule):
    """Progressive overlapping patch embedding for SVTRv2."""

    def __init__(self,
                 in_channels: int,
                 feat_max_size: Sequence[int],
                 embed_dim: int,
                 use_pos_embed: bool = False,
                 flatten: bool = False,
                 bias: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        layers: List[nn.Module] = [
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=bias,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=bias,
            ),
        ]
        if use_pos_embed:
            layers.append(ADDPosEmbed(feat_max_size, embed_dim))
        if flatten:
            layers.append(FlattenTranspose())
        self.patch_embed = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        x = self.patch_embed(x)
        if x.dim() == 4:
            sz = list(x.shape[2:])
        else:
            raise AssertionError('Unexpected patch embedding output shape.')
        return x, sz


class LastStage(BaseModule):
    """Optional last stage to map features to out_channels and pool height."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 last_drop: float = 0.1,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.last_conv = nn.Linear(in_channels, out_channels, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x: torch.Tensor,
                sz: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
        x = x.reshape(-1, sz[0], sz[1], x.shape[-1])
        x = x.mean(1)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x, [1, sz[1]]


class Feat2D(BaseModule):
    """Convert token sequence (B, N, C) to feature map (B, C, H, W)."""

    def forward(self, x: torch.Tensor,
                sz: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
        channels = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, channels, sz[0], sz[1])
        return x, list(sz)


@MODELS.register_module()
class SVTRv2Backbone(BaseModule):
    """SVTRv2 backbone/encoder (LNConvTwo33 variant).

    This module is migrated from OpenOCR:
    `openrec/modeling/encoders/svtrv2_lnconv_two33.py::SVTRv2LNConvTwo33`.

    It supports:
    - Conv2 local mixing (group conv x2/3) via ``mixer='Conv'`` blocks.
    - Higher feature height with 2D outputs (``feat2d=True``).

    Note:
        In MMOCR text recognition, the visual model is usually configured as
        ``model.encoder``. Despite the name, this class is registered as an
        encoder and can be used as a drop-in replacement of ``SVTREncoder``.
    """

    def __init__(self,
                 max_sz: Sequence[int] = (32, 128),
                 in_channels: int = 3,
                 out_channels: int = 192,
                 depths: Sequence[int] = (3, 6, 3),
                 dims: Sequence[int] = (64, 128, 256),
                 mixer: Sequence[Sequence[str]] = (('Conv', ) * 3,
                                                   ('Conv', ) * 3 +
                                                   ('Global', ) * 3,
                                                   ('Global', ) * 3),
                 use_pos_embed: bool = False,
                 sub_k: Sequence[Sequence[int]] = ((1, 1), (2, 1), (1, 1)),
                 num_heads: Sequence[int] = (2, 4, 8),
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.0,
                 last_drop: float = 0.1,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 norm_layer: type[nn.Module] = nn.LayerNorm,
                 act: type[nn.Module] = nn.GELU,
                 last_stage: bool = False,
                 feat2d: bool = True,
                 eps: float = 1e-6,
                 num_convs: Sequence[Sequence[int]] = ((2, ) * 3,
                                                      (2, ) * 3 + (3, ) * 3,
                                                      (3, ) * 3),
                 kernel_sizes: Sequence[Sequence[int]] = ((3, ) * 3,
                                                         (3, ) * 6,
                                                         (3, ) * 3),
                 pope_bias: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [int(max_sz[0]) // 4, int(max_sz[1]) // 4]
        self.pope = POPatchEmbed(
            in_channels=in_channels,
            feat_max_size=feat_max_size,
            embed_dim=dims[0],
            use_pos_embed=use_pos_embed,
            flatten=(mixer[0][0] != 'Conv'),
            bias=pope_bias,
        )

        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth

        stages: List[nn.Module] = []
        for i_stage in range(num_stages):
            stage = SVTRv2Stage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=kernel_sizes[i_stage]
                if len(kernel_sizes[i_stage]) == len(mixer[i_stage]) else
                [3] * len(mixer[i_stage]),
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=(False if i_stage == num_stages - 1 else True),
                eps=eps,
                num_conv=num_convs[i_stage]
                if len(num_convs[i_stage]) == len(mixer[i_stage]) else
                [2] * len(mixer[i_stage]),
            )
            stages.append(stage)

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            stages.append(LastStage(self.num_features, out_channels, last_drop))
        if feat2d:
            stages.append(Feat2D())

        self.stages = nn.ModuleList(stages)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')

    def forward(
        self,
        x: torch.Tensor,
        data_samples: Optional[List[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        if x.dim() == 5:
            x = x.flatten(0, 1)
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x

