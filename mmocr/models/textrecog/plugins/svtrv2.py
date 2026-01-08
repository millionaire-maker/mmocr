# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_init

from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import MODELS, TASK_UTILS
from mmocr.structures import TextRecogDataSample


class _Mlp(BaseModule):

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
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


class _Attention(BaseModule):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        qkv = (self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads,
                                   channels // self.num_heads).permute(
                                       2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _TransformerBlock(BaseModule):

    def __init__(
        self,
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
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODELS.register_module()
class FeatureRearrangementModule(BaseModule):
    """Feature Rearrangement Module (FRM) for SVTRv2.

    Migrated from OpenOCR:
    `openrec/modeling/decoders/rctc_decoder.py::RCTCDecoder` (FRM part).

    It performs:
    1) Horizontal rearrangement via width-wise Transformer block.
    2) Vertical rearrangement via height attention, collapsing H -> 1.

    Args:
        in_channels (int): Feature channels.
        enabled (bool): Whether to enable FRM. If False, it falls back to
            mean-pooling along height. Defaults to True.
        num_heads (int, optional): Attention heads for width-wise Transformer.
            Defaults to ``max(1, in_channels // 32)``.
    """

    def __init__(self,
                 in_channels: int,
                 enabled: bool = True,
                 num_heads: Optional[int] = None,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.enabled = enabled
        self.in_channels = in_channels
        if num_heads is None:
            num_heads = max(1, in_channels // 32)
        self.num_heads = num_heads

        self.char_token = nn.Parameter(
            torch.zeros([1, 1, in_channels], dtype=torch.float32),
            requires_grad=True,
        )
        trunc_normal_init(self.char_token, mean=0, std=0.02)

        self.fc_kv = nn.Linear(in_channels, 2 * in_channels, bias=True)
        self.w_atten_block = _TransformerBlock(
            dim=in_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange features.

        Args:
            x (Tensor): Feature map of shape (B, C, H, W).

        Returns:
            Tensor: Rearranged sequence features of shape (B, W, C).
        """
        if x.dim() != 4:
            raise ValueError(f'FRM expects 4D feature map (B,C,H,W), got {x.shape}')
        if not self.enabled:
            return x.mean(dim=2).permute(0, 2, 1)

        bsz, channels, height, width = x.shape
        x = self.w_atten_block(
            x.permute(0, 2, 3, 1).reshape(-1, width, channels)).reshape(
                bsz, height, width, channels).permute(0, 3, 1, 2)

        x_kv = self.fc_kv(x.flatten(2).transpose(1, 2)).reshape(
            bsz, height * width, 2, channels).permute(2, 0, 3, 1)
        x_k, x_v = x_kv.unbind(0)  # (B, C, HW)
        char_token = self.char_token.expand(bsz, -1, -1)  # (B, 1, C)
        attn_ctc2d = char_token @ x_k  # (B, 1, HW)
        attn_ctc2d = attn_ctc2d.reshape(bsz, 1, height, width)
        attn_ctc2d = F.softmax(attn_ctc2d, dim=2)  # softmax over H
        attn_ctc2d = attn_ctc2d.permute(0, 3, 1, 2)  # (B, W, 1, H)
        x_v = x_v.reshape(bsz, channels, height, width)

        feats = attn_ctc2d @ x_v.permute(0, 3, 2, 1)  # (B, W, 1, C)
        feats = feats.squeeze(2)  # (B, W, C)
        return feats


class _Embeddings(BaseModule):

    def __init__(self,
                 d_model: int,
                 vocab: int,
                 padding_idx: Optional[int] = None,
                 scale_embedding: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        if self.scale_embedding:
            return x * (self.d_model**0.5)
        return x


class _CrossAttention(BaseModule):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                q: torch.Tensor,
                kv: torch.Tensor,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_kv, channels = kv.shape[1:]
        num_q = q.shape[1]

        q = self.q(q).reshape(
            [-1, num_q, self.num_heads,
             channels // self.num_heads]).transpose(1, 2)
        q = q * self.scale
        k, v = self.kv(kv).reshape(
            [-1, num_kv, 2, self.num_heads,
             channels // self.num_heads]).permute(2, 0, 3, 1, 4)

        attn = q.matmul(k.transpose(2, 3))
        if key_mask is not None:
            attn = attn + key_mask.unsqueeze(1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose(1, 2).reshape((-1, num_q, channels))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _SSMatchLayer(BaseModule):

    def __init__(
        self,
        dim: int,
        nextq2subs_heads: Optional[int] = None,
        dynq2img_heads: int = 2,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        epsilon: float = 1e-6,
        is_last_layer: bool = False,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.dim = dim
        if nextq2subs_heads is None:
            nextq2subs_heads = max(1, dim // 32)

        self.normq1 = nn.LayerNorm(dim, eps=epsilon)
        self.normkv1 = nn.LayerNorm(dim, eps=epsilon)
        self.images_to_question_cross_attn = _CrossAttention(
            dim,
            num_heads=nextq2subs_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.normq2 = nn.LayerNorm(dim, eps=epsilon)
        self.normkv2 = nn.LayerNorm(dim, eps=epsilon)
        self.question_to_images_cross_attn = _CrossAttention(
            dim,
            num_heads=dynq2img_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.is_last_layer = is_last_layer

    def forward(self, question_f: torch.Tensor, prompt_f: torch.Tensor,
                visual_f: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        question_f = question_f + self.drop_path(
            self.images_to_question_cross_attn(self.normq1(question_f),
                                               self.normkv1(prompt_f), mask))
        question_f = question_f.reshape(visual_f.shape[0], -1, self.dim)
        question_f = self.question_to_images_cross_attn(
            self.normq2(question_f), self.normkv2(visual_f))
        if self.is_last_layer:
            return question_f
        return question_f.flatten(0, 1).unsqueeze(1)


@MODELS.register_module()
class SemanticGuidanceModule(BaseModule):
    """Semantic Guidance Module (SGM) for SVTRv2 (training-only auxiliary loss).

    This module is migrated from OpenOCR SMTR/GTC branch:
    - `openrec/modeling/decoders/smtr_decoder.py::SMTRDecoder` (core logic)
    - `openrec/modeling/decoders/__init__.py::GTCDecoder` (training-only usage)

    In MMOCR, this module is designed to be called only in ``model.loss()``
    (i.e. training). It does not participate in ``predict`` and therefore
    introduces no inference overhead.
    """

    def __init__(self,
                 in_channels: int,
                 dictionary: Union[Dict, Dictionary],
                 enabled: bool = True,
                 loss_weight: float = 0.1,
                 sub_str_len: int = 5,
                 num_layer: int = 1,
                 nextq2subs_heads: Optional[int] = None,
                 dynq2img_heads: int = 2,
                 drop_path_rate: float = 0.1,
                 detach_visual: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.enabled = enabled
        self.loss_weight = float(loss_weight)
        self.sub_str_len = int(sub_str_len)
        self.detach_visual = detach_visual

        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        elif isinstance(dictionary, Dictionary):
            self.dictionary = dictionary
        else:
            raise TypeError(
                'The type of dictionary should be `Dictionary` or dict, '
                f'but got {type(dictionary)}')

        self.num_classes = self.dictionary.num_classes
        if self.dictionary.padding_idx is None:
            raise ValueError('SGM requires dictionary.with_padding=True so that '
                             '`dictionary.padding_idx` is available.')
        self.ignore_index = int(self.dictionary.padding_idx)

        dim = in_channels
        self.char_embed = _Embeddings(
            d_model=dim,
            vocab=self.num_classes,
            padding_idx=self.ignore_index,
            scale_embedding=True,
        )

        # Learnable query tokens and prompt embeddings.
        self.next_token = nn.Parameter(
            torch.zeros([1, 1, dim], dtype=torch.float32), requires_grad=True)
        self.pre_token = nn.Parameter(
            torch.zeros([1, 1, dim], dtype=torch.float32), requires_grad=True)
        self.prompt_next_embed = nn.Parameter(
            torch.zeros([1, 1, self.sub_str_len + 1, dim], dtype=torch.float32),
            requires_grad=True)
        self.prompt_pre_embed = nn.Parameter(
            torch.zeros([1, 1, self.sub_str_len + 1, dim], dtype=torch.float32),
            requires_grad=True)
        trunc_normal_init(self.next_token, mean=0, std=0.02)
        trunc_normal_init(self.pre_token, mean=0, std=0.02)
        trunc_normal_init(self.prompt_next_embed, mean=0, std=0.02)
        trunc_normal_init(self.prompt_pre_embed, mean=0, std=0.02)

        dpr = np.linspace(0, drop_path_rate, num_layer).tolist()
        self.cmff_decoder = nn.ModuleList([
            _SSMatchLayer(
                dim=dim,
                nextq2subs_heads=nextq2subs_heads,
                dynq2img_heads=dynq2img_heads,
                drop_path=dpr[i],
                is_last_layer=(i == num_layer - 1),
            ) for i in range(num_layer)
        ])
        self.norm_pred = nn.LayerNorm(dim, eps=1e-6)
        self.ques_head = nn.Linear(dim, self.num_classes, bias=True)

    def _build_context_windows(
            self, gt_ids: List[torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(gt_ids)
        lengths = torch.tensor([int(t.numel()) for t in gt_ids],
                               device=device,
                               dtype=torch.long)
        max_len = int(lengths.max().item()) if batch_size > 0 else 0
        if max_len == 0:
            empty = torch.empty((batch_size, 0), device=device, dtype=torch.long)
            return empty, empty.view(batch_size, 0, self.sub_str_len), empty.view(
                batch_size, 0, self.sub_str_len)

        labels = torch.full((batch_size, max_len),
                            fill_value=self.ignore_index,
                            dtype=torch.long,
                            device=device)
        left_ctx = torch.full((batch_size, max_len, self.sub_str_len),
                              fill_value=self.ignore_index,
                              dtype=torch.long,
                              device=device)
        right_ctx = torch.full((batch_size, max_len, self.sub_str_len),
                               fill_value=self.ignore_index,
                               dtype=torch.long,
                               device=device)
        for b, ids in enumerate(gt_ids):
            l = int(ids.numel())
            if l == 0:
                continue
            labels[b, :l] = ids[:l]
            for t in range(l):
                l_ctx = ids[max(0, t - self.sub_str_len):t]
                if l_ctx.numel() > 0:
                    left_ctx[b, t, -l_ctx.numel():] = l_ctx
                r_ctx = ids[t + 1:t + 1 + self.sub_str_len]
                if r_ctx.numel() > 0:
                    right_ctx[b, t, :r_ctx.numel()] = r_ctx
        return labels, left_ctx, right_ctx

    def loss(self, visual_feat: torch.Tensor,
             data_samples: Sequence[TextRecogDataSample]) -> Dict[str,
                                                                torch.Tensor]:
        if (not self.enabled) or self.loss_weight <= 0:
            return {}
        if data_samples is None:
            return {}

        if visual_feat.dim() == 4:
            visual_feat = visual_feat.flatten(2).transpose(1, 2)
        if visual_feat.dim() != 3:
            raise ValueError(
                f'SGM expects visual_feat as (B,N,C) or (B,C,H,W), got {visual_feat.shape}'
            )
        if self.detach_visual:
            visual_feat = visual_feat.detach()

        device = visual_feat.device
        gt_ids: List[torch.Tensor] = []
        for sample in data_samples:
            if getattr(sample.gt_text, 'indexes', None) is None:
                raise ValueError(
                    'SGM requires `data_sample.gt_text.indexes`; make sure '
                    'CTCModuleLoss.get_targets() is called before SGM loss.'
                )
            ids = sample.gt_text.indexes.to(device=device, dtype=torch.long)
            gt_ids.append(ids)

        labels, left_ctx, right_ctx = self._build_context_windows(gt_ids,
                                                                  device=device)
        if labels.numel() == 0:
            return {}

        bsz, max_len = labels.shape
        dim = visual_feat.shape[-1]
        if dim != self.next_token.shape[-1]:
            raise ValueError(
                f'visual_feat dim ({dim}) != SGM in_channels ({self.next_token.shape[-1]})'
            )

        # Build prompt features (B, max_len, sub_len+1, dim)
        prompt_next_embed = self.prompt_next_embed.expand(bsz, max_len, -1, -1)
        prompt_pre_embed = self.prompt_pre_embed.expand(bsz, max_len, -1, -1)
        prompt_char_next = torch.cat([
            prompt_next_embed[:, :, :1, :],
            prompt_next_embed[:, :, 1:, :] + self.char_embed(left_ctx),
        ], 2)
        prompt_char_pre = torch.cat([
            prompt_pre_embed[:, :, :1, :],
            prompt_pre_embed[:, :, 1:, :] + self.char_embed(right_ctx),
        ], 2)

        ques_next = self.next_token.expand(bsz, max_len, -1, -1)
        ques_pre = self.pre_token.expand(bsz, max_len, -1, -1)

        # Masks: ignore padded context tokens.
        mask_next = torch.where(left_ctx == self.ignore_index, float('-inf'), 0.0)
        mask_pre = torch.where(right_ctx == self.ignore_index, float('-inf'), 0.0)

        prompt_char = torch.cat([prompt_char_next, prompt_char_pre], 1)  # B, 2L, S, D
        questions = torch.cat([ques_next, ques_pre], 1)  # B, 2L, 1, D
        mask = torch.cat([mask_next, mask_pre], 1).flatten(0, 1)  # (B*2L, S-1)

        # Add mask for the first prompt token (never masked).
        mask_pad = torch.zeros((mask.shape[0], 1), dtype=torch.float32, device=device)
        mask = torch.cat([mask_pad, mask], 1)  # (B*2L, S)

        questions = questions.flatten(0, 1)  # (B*2L, 1, D)
        prompt_char = prompt_char.flatten(0, 1)  # (B*2L, S, D)

        for layer in self.cmff_decoder:
            questions = layer(questions, prompt_char, visual_feat,
                              mask.unsqueeze(1))

        logits = self.ques_head(self.norm_pred(questions))  # (B, 2L, C)
        targets = torch.cat([labels, labels], 1)  # (B, 2L)
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_classes),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )
        return {'loss_sgm': loss * self.loss_weight}

