# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath

from mmocr.registry import MODELS
from .base import BasePreprocessor


class Mlp(nn.Module):
    """MLP block used by DGAB.

    This is a minimal adaptation from the official TPS++ implementation.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: nn.Module = nn.GELU,
                 drop: float = 0.) -> None:
        super().__init__()
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


class DGAB_Block(nn.Module):
    """Dynamic Gated Attention Block (DGAB) used in TPS++.

    Note: The official TPS++ implementation applies Linear layers on the
    feature map width dimension, therefore it requires `width == dim`.
    """

    def __init__(self,
                 dim: int,
                 point: int,
                 height: int,
                 width: int,
                 qkv_bias: bool = False,
                 proj_drop: float = 0.) -> None:
        super().__init__()
        if width != dim:
            raise ValueError(
                'TPS++ DGAB requires `img_size[1] == num_img_channel` to match '
                f'the official implementation, but got width={width}, dim={dim}'
            )

        self.mlp_h = nn.Linear(height + point, height + 1, bias=qkv_bias)
        self.mlp_w = nn.Linear(width + point, width + 1, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); y: (B, T, C)
        y = y.permute(0, 2, 1)  # (B, C, T)

        w = self.mlp_w(torch.cat([x.mean(2), y], 2))  # (B, C, W+1)
        v_w = w[:, :, :-1].softmax(dim=-1).unsqueeze(2)  # (B, C, 1, W)

        h = self.mlp_h(torch.cat([x.mean(3), y], 2))  # (B, C, H+1)
        v_h = h[:, :, :-1].softmax(dim=-1).unsqueeze(3)  # (B, C, H, 1)

        x = (
            v_h * x * h[:, :, -1].unsqueeze(-1).unsqueeze(-1) +
            v_w * x * w[:, :, -1].unsqueeze(-1).unsqueeze(-1))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DGAB(nn.Module):

    def __init__(self,
                 dim: int,
                 width: int,
                 high: int,
                 point: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 skip_lam: float = 1.0) -> None:
        super().__init__()
        tuple_dim = [high, width]
        self.norm1 = norm_layer(tuple_dim)
        self.attn = DGAB_Block(
            dim=dim,
            point=point,
            height=high,
            width=width,
            qkv_bias=qkv_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(tuple_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.skip_lam = skip_lam

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), y)) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class ChannelAttentionModule(nn.Module):

    def __init__(self, channel: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if ratio > 0:
            self.shared_mlp = nn.Sequential(
                nn.Conv2d(channel, channel // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel // ratio, channel, 1, bias=False),
            )
        else:
            self.shared_mlp = nn.Sequential(
                nn.Conv2d(channel, channel * -ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel * -ratio, channel, 1, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttentionModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):

    def __init__(self, channel: int, ratio: int = 16) -> None:
        super().__init__()
        self.ratio = ratio
        self.channel_attention = ChannelAttentionModule(channel, ratio)
        self.spatial_attention = SpatialAttentionModule()
        if ratio < 0:
            self.down = nn.Conv2d(channel, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        if self.ratio < 0:
            out = self.down(out).squeeze(1)
        return out


class Encoder_Decoder_Feature_Extractor(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 num_channels: int = 64,
                 attn_mode: str = 'nearest',
                 stride: int = 2,
                 ratio: List[int] = [1, 1, 1],
                 u_channel: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.k_encoder = nn.Sequential(
            self._encoder_layer(
                in_channels * u_channel,
                num_channels * ratio[0],
                stride=1),
            self._encoder_layer(num_channels * ratio[0],
                                num_channels * ratio[1],
                                stride=2),
            self._encoder_layer(num_channels * ratio[1],
                                num_channels * ratio[2],
                                stride=stride),
            self._encoder_layer(num_channels, num_channels, stride=(2, 1)),
        )

        self.atten = CBAM(num_channels * ratio[2])

        self.k_decoder = nn.Sequential(
            self._decoder_layer(
                num_channels, num_channels, scale_factor=(2, 1), mode=attn_mode),
            self._decoder_layer(
                num_channels * ratio[2],
                num_channels * ratio[1],
                scale_factor=stride,
                mode=attn_mode),
            self._decoder_layer(
                num_channels * ratio[1],
                num_channels * ratio[0],
                scale_factor=2,
                mode=attn_mode),
            self._decoder_layer(
                num_channels * ratio[0],
                in_channels,
                scale_factor=1,
                mode=attn_mode),
        )

    def _encoder_layer(self,
                       in_channels: int,
                       out_channels: int,
                       kernel_size: int = 3,
                       stride: Union[int, Tuple[int, int]] = 2,
                       padding: int = 1) -> ConvModule:
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def _decoder_layer(self,
                       in_channels: int,
                       out_channels: int,
                       kernel_size: int = 3,
                       stride: int = 1,
                       padding: int = 1,
                       mode: str = 'nearest',
                       scale_factor: Optional[Union[int,
                                                    Tuple[int,
                                                          int]]] = None,
                       size: Optional[Tuple[int, int]] = None) -> nn.Sequential:
        align_corners = None if mode == 'nearest' else True
        return nn.Sequential(
            nn.Upsample(
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))

    def forward(self, k: torch.Tensor) -> dict:
        features: List[torch.Tensor] = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        point = features[-1]

        k = self.atten(point)

        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)
        return {'decoded_feature': k, 'encoded_feature': point}


class Multi_Scale_Fearue_Aggregation(nn.Module):

    def __init__(self,
                 num_img_channel: int,
                 point_size: Tuple[int, int],
                 p_stride: int,
                 num_map: int = 2) -> None:
        super().__init__()
        self.num_img_channel = num_img_channel
        self.point_x = point_size[1]
        self.point_y = point_size[0]
        self.tf_ratio = 4

        self.conv = Encoder_Decoder_Feature_Extractor(
            in_channels=num_img_channel,
            num_channels=64,
            stride=p_stride,
            u_channel=num_map,
        )

    def forward(self, batch_img: torch.Tensor) -> dict:
        logits = self.conv(batch_img)
        return {'de_feat': logits['decoded_feature'], 'en_feat': logits['encoded_feature']}


class Transformation_Parameter_Estimation(nn.Module):

    def __init__(self,
                 img_channel: int,
                 point_channel: int,
                 num_img_channel: int,
                 point_size: Tuple[int, int],
                 img_size: Tuple[int, int]) -> None:
        super().__init__()

        self.num_img_channel = num_img_channel
        self.point_x = point_size[1]
        self.point_y = point_size[0]
        self.tf_layers = 1
        self.scale = num_img_channel**-0.5
        self.without_as = False

        self.num_fiducial = self.point_y * self.point_x

        self.p_linear = nn.Sequential(
            nn.Linear(point_channel, 32),
            nn.Linear(32, 64 * 2),
        )
        self.feat_linear = nn.Sequential(
            nn.Linear(img_channel, 32),
            nn.Linear(32, 64 * 2),
        )

        self.atten = nn.ModuleList([
            DGAB(
                dim=num_img_channel,
                point=self.num_fiducial,
                width=img_size[1],
                high=img_size[0],
            ) for _ in range(self.tf_layers)
        ])

        self.localization_fc1 = nn.Sequential(
            nn.Linear(num_img_channel, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
            nn.ReLU(True),
        )
        self.localization_fc2 = nn.Linear(2 * self.num_fiducial,
                                          self.num_fiducial * 2)

        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(0.1, self.point_x - 0.1,
                                 num=int(self.point_x)) / self.point_x
        ctrl_pts_y = np.linspace(0.1, self.point_y - 0.1,
                                 num=int(self.point_y)) / self.point_y
        initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

    def atten_score(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        attn = torch.einsum('b m c, b n c -> b m n', a, b)
        attn = attn.mul(self.scale)
        attn = torch.tanh(attn)
        return attn

    def get_score(self, point: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        feat = feat.flatten(2).transpose(1, 2)
        p1 = self.p_linear(point)
        f = self.feat_linear(feat)
        pc_score = self.atten_score(f, p1)
        if self.without_as:
            pc_score = torch.zeros_like(pc_score)
        return pc_score

    def forward(
        self, en_feat: torch.Tensor, de_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = en_feat.size(0)
        en_feat = en_feat.flatten(2).transpose(1, 2)  # (B, num_fiducial, C)
        for atten_layer in self.atten:
            de_feat = atten_layer(de_feat, en_feat)

        control_point = self.localization_fc2(
            self.localization_fc1(en_feat).view(batch_size, -1)).view(
                batch_size, self.num_fiducial, 2)
        atten_score = self.get_score(en_feat, de_feat)
        return control_point, atten_score


class Attention_Enhanced_TPS(nn.Module):

    def __init__(self, rectified_img_size: Tuple[int, int],
                 point_size: Tuple[int, int]) -> None:
        super().__init__()
        self.eps = 1e-6
        self.thela = 0.5
        self.point_size = point_size

        self.point_y = point_size[0]
        self.point_x = point_size[1]
        self.num_fiducial = self.point_y * self.point_x

        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]

        self.C = self._build_C()
        self.P = self._build_P(self.rectified_img_width,
                               self.rectified_img_height)

        self.register_buffer(
            'hat_C',
            torch.tensor(self._build_hat_C(self.num_fiducial,
                                           self.C)).float())
        self.register_buffer(
            'P_hat',
            torch.tensor(
                self._build_P_hat(self.num_fiducial, self.C,
                                  self.P)).float())
        self.register_buffer('P_tensor', torch.tensor(self.P).float())

    def _build_C(self) -> np.ndarray:
        ctrl_pts_x = np.linspace(0.5, self.point_x - 0.5,
                                 num=int(self.point_x)) / self.point_x
        ctrl_pts_y = np.linspace(0.5, self.point_y - 0.5,
                                 num=int(self.point_y)) / self.point_y
        C = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2).reshape(
            [-1, 2])
        return C

    def _build_hat_C(self, num_fiducial: int, C: np.ndarray) -> np.ndarray:
        hat_C = np.zeros((num_fiducial, num_fiducial), dtype=float)
        for i in range(num_fiducial):
            for j in range(i, num_fiducial):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        delta_C = np.concatenate([
            np.concatenate([np.ones((num_fiducial, 1)), C, hat_C], axis=1),
            np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
            np.concatenate([np.zeros((1, 3)), np.ones((1, num_fiducial))],
                           axis=1),
        ],
                                 axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, rectified_img_width: int,
                 rectified_img_height: int) -> np.ndarray:
        rectified_img_grid_x = np.linspace(
            0.5, rectified_img_width - 0.5,
            num=int(rectified_img_width)) / rectified_img_width
        rectified_img_grid_y = np.linspace(
            0.5, rectified_img_height - 0.5,
            num=int(rectified_img_height)) / rectified_img_height
        P = np.stack(np.meshgrid(rectified_img_grid_x, rectified_img_grid_y),
                     axis=2)
        return P.reshape([-1, 2])

    def _build_P_hat(self, num_fiducial: int, C: np.ndarray,
                     P: np.ndarray) -> np.ndarray:
        n = P.shape[0]
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, num_fiducial, 1))
        C_tile = np.expand_dims(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        P_hat = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        return P_hat

    def P_hat_score_process(self, P_hat: torch.Tensor, pc_score: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
        B, n, _ = pc_score.size()
        dtype = P_hat.dtype
        pc_score = pc_score.to(dtype=dtype)
        P = self.P_tensor.to(device=device, dtype=dtype).unsqueeze(0).repeat(
            B, 1, 1)
        P_hat = P_hat * (pc_score * self.thela + 1)
        P_hat = torch.cat(
            [torch.ones((B, n, 1), device=device, dtype=dtype), P, P_hat],
            dim=2)
        return P_hat

    def build_P_prime(self, batch_C_prime: torch.Tensor, pc_score: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        """Generate sampling grid P_prime.

        Args:
            batch_C_prime (Tensor): (B, num_fiducial, 2)
            pc_score (Tensor): (B, n, num_fiducial), n = H_r * W_r
        """
        batch_size = batch_C_prime.size(0)
        dtype = batch_C_prime.dtype
        batch_inv_delta_C = self.hat_C.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1)

        batch_P_hat = self.P_hat.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1)
        batch_P_hat = self.P_hat_score_process(batch_P_hat, pc_score, device)

        batch_C_prime_with_zeros = torch.cat([
            batch_C_prime,
            torch.zeros(batch_size, 3, 2, device=device, dtype=dtype),
        ],
                                             dim=1)
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)
        return batch_P_prime


class _TPSPPFeatureExtractor(nn.Module):

    def __init__(self, in_channels: int = 3, init_cfg: Optional[Union[dict,
                                                                      List[
                                                                          dict]]] = None) -> None:
        super().__init__()
        del init_cfg

        self.conv1 = ConvModule(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            32,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv3 = ConvModule(
            32,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feat0 = self.conv1(x)  # (B, 32, H, W)
        feat1 = self.conv2(feat0)  # (B, 32, H/2, W/2)
        feat2 = self.conv3(feat1)  # (B, 64, H/2, W/2)
        return feat2, [feat0, feat1]


class TPSPPCore(nn.Module):
    """TPS++ core that generates sampling grids.

    This is adapted from the official TPS++ implementation in
    `3rdparty/TPS_PP/mmocr/.../tps_pp.py`.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (16, 64),
                 rectified_img_size: Tuple[int, int] = (32, 100),
                 num_img_channel: int = 64,
                 point_size: Tuple[int, int] = (2, 16),
                 p_stride: int = 2,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__()
        del init_cfg

        self.img_size = img_size
        self.rectified_img_size = rectified_img_size

        if img_size[1] != num_img_channel:
            raise ValueError(
                'TPS++ requires `img_size[1] == num_img_channel` to match DGAB '
                f'behavior, but got img_size={img_size}, num_img_channel={num_img_channel}'
            )

        pc_ratio = 1
        ic_ratio = 1
        self.num_fiducial = point_size[0] * point_size[1]
        self.num_img_channel = num_img_channel
        self.point_channel = num_img_channel * pc_ratio
        self.img_channel = num_img_channel * ic_ratio

        self.MSFA = Multi_Scale_Fearue_Aggregation(
            num_img_channel=self.num_img_channel,
            point_size=point_size,
            p_stride=p_stride,
            num_map=3,
        )
        self.TPE = Transformation_Parameter_Estimation(
            img_channel=self.point_channel,
            point_channel=self.img_channel,
            num_img_channel=self.num_img_channel,
            point_size=point_size,
            img_size=self.img_size,
        )

        self.down0 = ConvModule(32, self.img_channel, kernel_size=3, stride=2, padding=1)
        self.down1 = ConvModule(32, self.img_channel, kernel_size=1, stride=1)
        self.down2 = ConvModule(64, self.img_channel, kernel_size=1, stride=1)

        self.atten_tps = Attention_Enhanced_TPS(self.rectified_img_size, point_size)

    def forward(self, batch_img: torch.Tensor,
                outs: List[torch.Tensor]) -> torch.Tensor:
        """Generate TPS++ grid.

        Args:
            batch_img (Tensor): (B, 64, H, W), where (H, W) == `img_size`.
            outs (list[Tensor]): A list with two tensors:
                outs[0] (B, 32, H*2, W*2), outs[1] (B, 32, H, W).

        Returns:
            Tensor: Grid with shape (B, H_r, W_r, 2) in [0, 1] coordinates.
        """
        feat0 = self.down0(outs[0])
        feat1 = self.down1(outs[1])
        feat2 = self.down2(batch_img)

        feat_cat = torch.cat((feat0, feat1, feat2), dim=1)
        logits = self.MSFA(feat_cat)
        de_feat = logits['de_feat']
        en_feat = logits['en_feat']

        control_point, pc_score = self.TPE(en_feat, de_feat)

        h_in, w_in = self.img_size
        pc_score = pc_score.view(control_point.size(0), h_in, w_in,
                                 self.num_fiducial).permute(0, 3, 1, 2)
        if self.rectified_img_size != self.img_size:
            pc_score = F.interpolate(
                pc_score,
                size=self.rectified_img_size,
                mode='bilinear',
                align_corners=True,
            )
        pc_score = pc_score.permute(0, 2, 3, 1).reshape(
            control_point.size(0), -1, self.num_fiducial)

        build_P_prime = self.atten_tps.build_P_prime(control_point, pc_score,
                                                     batch_img.device)
        grid = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])
        return grid


@MODELS.register_module()
class TPSPP(BasePreprocessor):
    """TPS++ rectification preprocessor (IJCAI 2023).

    This module is designed to be plug-and-play in MMOCR recognizers via
    ``model.preprocessor`` (same entry as the built-in TPS/STN).

    Args:
        in_channels (int): Number of input image channels. Defaults to 3.
        resized_image_size (Tuple[int, int]): Size (H, W) used inside TPS++
            for parameter estimation. Defaults to (32, 128).
        output_image_size (Tuple[int, int]): Output rectified image size
            (H, W), should match ``encoder.img_size``. Defaults to (32, 100).
        num_img_channel (int): Internal feature channels for TPS++. The
            official implementation uses 64. Defaults to 64.
        point_size (Tuple[int, int]): Grid size (Y, X) of fiducial points.
            Defaults to (2, 16) (32 points).
        p_stride (int): Stride used in MSFA encoder. Defaults to 2.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels: int = 3,
                 resized_image_size: Tuple[int, int] = (32, 128),
                 output_image_size: Tuple[int, int] = (32, 100),
                 num_img_channel: int = 64,
                 point_size: Tuple[int, int] = (2, 16),
                 p_stride: int = 2,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.resized_image_size = resized_image_size
        self.output_image_size = output_image_size

        img_size = (resized_image_size[0] // 2, resized_image_size[1] // 2)
        if img_size[0] <= 0 or img_size[1] <= 0:
            raise ValueError(
                f'Invalid resized_image_size={resized_image_size}; '
                'it must be positive and divisible by 2.')

        self.feat_extractor = _TPSPPFeatureExtractor(in_channels=in_channels)
        self.core = TPSPPCore(
            img_size=img_size,
            rectified_img_size=output_image_size,
            num_img_channel=num_img_channel,
            point_size=point_size,
            p_stride=p_stride,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        resized_img = F.interpolate(
            img,
            size=self.resized_image_size,
            mode='bilinear',
            align_corners=True)
        feat, outs = self.feat_extractor(resized_img)
        grid_01 = self.core(feat, outs)

        grid = grid_01 * 2.0 - 1.0
        rectified = F.grid_sample(
            img, grid, padding_mode='border', align_corners=True)
        return rectified
