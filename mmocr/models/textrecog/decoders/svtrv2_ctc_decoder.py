# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseDecoder


@MODELS.register_module()
class SVTRv2CTCDecoder(BaseDecoder):
    """CTC decoder for SVTRv2 with optional FRM/SGM.

    This decoder keeps MMOCR's CTC loss/postprocess interfaces while enabling
    SVTRv2's key plug-ins:

    - FRM (Feature Rearrangement Module): expects 2D features (B,C,H,W) and
      outputs a 1D sequence (B,W,C) for CTC.
    - SGM (Semantic Guidance Module): training-only auxiliary loss branch.

    Args:
        in_channels (int): Feature channels of the input sequence/features.
        dictionary (dict or :obj:`Dictionary`): Dictionary config or instance.
        frm (dict, optional): Config to build FRM. Set to None to disable.
        sgm (dict, optional): Config to build SGM. Set to None to disable.
    """

    def __init__(self,
                 in_channels: int,
                 dictionary: Union[Dict, Dictionary] = None,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 max_seq_len: int = 40,
                 frm: Optional[Dict] = None,
                 sgm: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        self.in_channels = int(in_channels)
        self.ctc_head = nn.Linear(
            in_features=self.in_channels,
            out_features=self.dictionary.num_classes,
        )
        self.softmax = nn.Softmax(dim=-1)

        self.frm = None
        if frm is not None:
            frm_cfg = frm.copy()
            frm_cfg.setdefault('in_channels', self.in_channels)
            self.frm = MODELS.build(frm_cfg)

        self.sgm = None
        if sgm is not None:
            sgm_cfg = sgm.copy()
            sgm_cfg.setdefault('in_channels', self.in_channels)
            sgm_cfg.setdefault('dictionary', dictionary)
            self.sgm = MODELS.build(sgm_cfg)

    def _split_feats(
        self, out_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare features for CTC logits and for SGM.

        Returns:
            tuple:
            - ctc_feat (Tensor): (B, T, C)
            - sgm_feat (Tensor): (B, N, C)
        """
        if out_enc is None:
            raise ValueError('`out_enc` must not be None for SVTRv2CTCDecoder.')

        if out_enc.dim() == 4:
            # (B, C, H, W) -> (B, HW, C)
            sgm_feat = out_enc.flatten(2).permute(0, 2, 1)
            if self.frm is not None:
                ctc_feat = self.frm(out_enc)
            else:
                if out_enc.size(2) == 1:
                    ctc_feat = out_enc.squeeze(2).permute(0, 2, 1)
                else:
                    ctc_feat = out_enc.mean(dim=2).permute(0, 2, 1)
            return ctc_feat, sgm_feat

        if out_enc.dim() == 3:
            # (B, T, C)
            return out_enc, out_enc

        raise ValueError(
            f'Unsupported out_enc shape for SVTRv2CTCDecoder: {out_enc.shape}')

    def forward_train(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        ctc_feat, _ = self._split_feats(out_enc)
        return self.ctc_head(ctc_feat)

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        return self.softmax(self.forward_train(feat, out_enc, data_samples))

    def loss(self,
             feat: Optional[torch.Tensor] = None,
             out_enc: Optional[torch.Tensor] = None,
             data_samples: Optional[Sequence[TextRecogDataSample]] = None
             ) -> Dict:
        """Calculate CTC loss (and optional SGM auxiliary loss)."""
        if self.module_loss is None:
            raise ValueError('`module_loss` must be set for loss computation.')
        if data_samples is None:
            raise ValueError('`data_samples` must not be None for loss.')

        # Prepare targets for CTC (also provides `gt_text.indexes` for SGM).
        data_samples = self.module_loss.get_targets(data_samples)

        ctc_feat, sgm_feat = self._split_feats(out_enc)
        logits = self.ctc_head(ctc_feat)
        losses = self.module_loss(logits, data_samples)

        if self.sgm is not None and self.training:
            if hasattr(self.sgm, 'loss'):
                losses.update(self.sgm.loss(sgm_feat, data_samples))
            else:
                raise AttributeError(
                    f'{type(self.sgm)} must implement a `loss()` method.')
        return losses

