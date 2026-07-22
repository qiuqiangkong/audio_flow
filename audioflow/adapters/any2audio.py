import torch
import torch.nn as nn
from torch import Tensor

from audioflow.utils.misc import get_single_value
from audioflow.utils.xml import batch_get_xml_attr, batch_xml_to_text
from audioflow.adapters.wrappers import T5Wrapper, ClapTextWrapper, ConvNeXtWrapper

import time


class Any2AudioAdapter(nn.Module): 
    def __init__(self, in_dim: int, dim: int, **kwargs):
        super().__init__()

        self.t5 = T5Wrapper(dim)
        self.clap_text = ClapTextWrapper(dim)
        self.feature_encoder = ConvNeXtWrapper(in_dim, dim)

        self.dt = 0.1  # Token duration in seconds (100 ms per token)

    def forward(self, data: dict) -> Tensor:
        r"""
        b: batch_size
        l_v: cond_seq_len (value)
        l_q: target_seq_len (query)
        d: dim

        Args:
            data: dict

        Returns:
            controls: dict
        """
        device = next(self.parameters()).device
        text = batch_xml_to_text(data["text"])
        prompt = batch_get_xml_attr(data["text"], attr="prompt")

        # Prompt
        text, text_mask = self.t5(prompt)  # (b, l_v, d), (b, l_v)

        # Input audio
        feat = self.feature_encoder(data["input_feature"])  # (b, l_v, d)
        feat_mask = data["input_mask"]  # (b, l_v)

        # Input sequence
        seq = torch.cat([text, feat], dim=1)  # (b, l_v, d)
        in_mask = torch.cat([text_mask, feat_mask], dim=1)  # (b, l_v)        

        # Attention masks
        tgt_mask = data["target_mask"]  # (b, l_q)
        self_mask = tgt_mask[:, :, None] & tgt_mask[:, None, :]  # (b, l_q, l_q)
        cross_mask = tgt_mask[:, :, None] & in_mask[:, None, :]  # (b, l_q, l_v)

        # Positions
        text_pos = torch.arange(text.shape[1], device=device)  # (l1,)
        feat_fps = get_single_value(data["input_fps"].tolist())
        feat_pos = torch.arange(data["input_feature"].shape[1], device=device) / feat_fps / self.dt  # (l_q,)
        in_pos = torch.cat([text_pos, feat_pos], dim=0)  # (l_v,)
        
        tgt_fps = get_single_value(data["target_fps"].tolist())
        tgt_pos = torch.arange(data["target_mask"].shape[1], device=device) / tgt_fps / self.dt  # (l_q,)

        # Global embedding
        clap = self.clap_text(prompt)  # (b, 1, d)
        c = clap

        controls = {
            "c": c,  # (b, 1, d)
            "seq": seq,  # (b, l_v, d)
            "self_attn_mask": self_mask.unsqueeze(1),  # (b, 1, l_q, l_q)
            "cross_attn_mask": cross_mask.unsqueeze(1),  # (b, 1, l_q, l_v)
            "input_pos": in_pos,  # (b, l_v)
            "target_pos": tgt_pos,  # (b, l_q)
        }
        
        return controls 