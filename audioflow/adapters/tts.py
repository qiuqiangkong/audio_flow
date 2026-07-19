import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from audioflow.utils.misc import get_single_value
from audioflow.utils.xml import batch_get_xml_attr, batch_xml_to_text
from audioflow.adapters.wrappers import T5Wrapper, CharWrapper


class TTSAdapter(nn.Module): 
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        self.t5 = T5Wrapper(dim)
        self.content_encoder = CharWrapper(dim)

        # Token duration in seconds (40 ms per token)
        self.dt = 0.04

    def forward(self, data: dict) -> Tensor:

        device = data["target"].device
        text = batch_xml_to_text(data["text"])
        content = batch_get_xml_attr(data["text"], attr="content")

        # Text & Content
        text, text_mask = self.t5(text)  # (b, l1, d), (b, l1)
        content, content_mask = self.content_encoder(content)  # (b, l2, d), (b, l2)

        # Conditional sequence
        seq = torch.cat([text, content], dim=1)  # (b, l_v, d)
        in_mask = torch.cat([text_mask, content_mask], dim=1)  # (b, l_v)

        # Self attention mask
        tgt_mask = data["target_mask"]  # (b, l_q)
        self_mask = tgt_mask[:, :, None] & tgt_mask[:, None, :]  # (b, l_q, l_q)

        # Cross attention mask
        cross_mask = tgt_mask[:, :, None] & in_mask[:, None, :]  # (b, l_q, l_v)

        # Positions
        text_pos = torch.arange(text.shape[1], device=device)  # (l1,)
        content_pos = torch.arange(content.shape[1], device=device)  # (l2,)
        in_pos = torch.cat([text_pos, content_pos], dim=0)  # (l_v,)

        tgt_fps = get_single_value(data["target_fps"].tolist())
        tgt_pos = torch.arange(data["target"].shape[1], device=device) / tgt_fps / self.dt  # (l_q,)

        controls = {
            "c": 0.,  # (b, 1, d)
            "seq": seq,  # (b, l_v, d)
            "self_attn_mask": self_mask.unsqueeze(1),  # (b, 1, l_q, l_q)
            "cross_attn_mask": cross_mask.unsqueeze(1),  # (b, 1, l_q, l_v)
            "input_pos": in_pos,  # (b, l_v)
            "target_pos": tgt_pos,  # (b, l_q)
        }

        return controls 